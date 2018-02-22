import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from lib.nms_wrapper import nms
from .torch_rcnn_utils import torch_apply_box_deltas, torch_clip_boxes


############################################################
#  Proposal Layer
############################################################
class RPNHead(nn.Module):
    def __init__(self, input_dims, anchors_per_location, anchor_stride):
        super(RPNHead, self).__init__()

        # Setup layers
        self.rpn_conv_shared = nn.Conv2d(
            input_dims, 512, kernel_size=3, stride=anchor_stride, padding=1)
        self.rpn_class_raw = nn.Conv2d(
            512, 2 * anchors_per_location, kernel_size=1)
        self.rpn_bbox_pred = nn.Conv2d(
            512, 4 * anchors_per_location, kernel_size=1)

    #[1,256,16,16]*[
    def forward(self, x):
        shared = F.relu(self.rpn_conv_shared(x), True)
        x = self.rpn_class_raw(shared)
        rpn_class_logits = x.permute(
            0, 2, 3, 1).contiguous().view(x.size(0), -1, 2)
        rpn_probs = F.softmax(rpn_class_logits, dim=-1)
        x = self.rpn_bbox_pred(shared)

        rpn_bbox = x.permute(0, 2, 3, 1).contiguous().view(
            x.size(0), -1, 4)  # reshape to (N, 4)

        return rpn_class_logits, rpn_probs, rpn_bbox



class RPNRefine (nn.Module):
    def __init__(self, anchors, config, mode='inference'):
        super(RPNRefine, self).__init__()

        self.anchors = anchors
        self.config = config
        self.proposal_count = self.config.POST_NMS_ROIS_TRAINING if mode == "training"\
            else self.config.POST_NMS_ROIS_INFERENCE


    def forward(self, rpn_class, rpn_bbox):
        # handling proposals
        scores = rpn_class[:, :, 1]
        # Box deltas [batch, num_rois, 4]
        deltas_mul = torch.autograd.Variable(torch.from_numpy(np.reshape(
            self.config.RPN_BBOX_STD_DEV, [1, 1, 4]).astype(np.float32))).cuda()
        deltas = rpn_bbox * deltas_mul

        pre_nms_limit = min(6000, self.anchors.shape[0])

        scores, ix = torch.topk(scores, pre_nms_limit, dim=-1,
                                largest=True, sorted=True)

        ix = torch.unsqueeze(ix, 2)
        ix = torch.cat([ix, ix, ix, ix], dim=2)
        deltas = torch.gather(deltas, 1, ix)

        scores = torch.unsqueeze(scores, 2)

        _anchors = []
        for i in range(self.config.IMAGES_PER_GPU):
            anchors = torch.autograd.Variable(torch.from_numpy(
                self.anchors.astype(np.float32))).cuda()
            _anchors.append(anchors)
        anchors = torch.stack(_anchors, 0)

        pre_nms_anchors = torch.gather(anchors, 1, ix)
        refined_anchors = torch_apply_box_deltas(pre_nms_anchors, deltas)

        # Clip to image boundaries. [batch, N, (y1, x1, y2, x2)]
        height, width = self.config.IMAGE_SHAPE[:2]
        window = np.array([0, 0, height, width]).astype(np.float32)
        window = torch.autograd.Variable(torch.from_numpy(window)).cuda()

        refined_anchors_clipped = torch_clip_boxes(refined_anchors, window)

        refined_proposals = []
        for i in range(self.config.IMAGES_PER_GPU):
            indices = nms(
                torch.cat([refined_anchors_clipped.data[i], scores.data[i]], 1), 0.7)
            indices = indices[:self.proposal_count]
            indices = torch.stack([indices, indices, indices, indices], dim=1)
            indices = torch.autograd.Variable(indices).cuda()
            proposals = torch.gather(refined_anchors_clipped[i], 0, indices)
            padding = self.proposal_count - proposals.size()[0]
            proposals = torch.cat(
                [proposals, torch.autograd.Variable(torch.zeros([padding, 4])).cuda()], 0)
            refined_proposals.append(proposals)

        rpn_rois = torch.stack(refined_proposals, 0)

        return rpn_rois


################
# Loss functions#
################
# region proposal network confidence loss
def rpn_class_loss(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
    """
    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = torch.eq(rpn_match, 1)
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = torch.ne(rpn_match, 0.)

    rpn_class_logits = torch.masked_select(rpn_class_logits, indices)
    anchor_class = torch.masked_select(anchor_class, indices)

    rpn_class_logits = rpn_class_logits.contiguous().view(-1, 2)

    anchor_class = anchor_class.contiguous().view(-1).type(torch.cuda.LongTensor)
    loss = F.cross_entropy(rpn_class_logits, anchor_class, weight=None)
    return loss


# region proposal bounding bbox loss
def rpn_bbox_loss(target_bbox, rpn_match, rpn_bbox, config):
    """Return the RPN bounding box loss graph.

    config: the model config object.
    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """
    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    indices = torch.eq(rpn_match, 1)
    rpn_bbox = torch.masked_select(rpn_bbox, indices)
    batch_counts = torch.sum(indices.float(), dim=1)

    outputs = []
    for i in range(config.IMAGES_PER_GPU):
        #        print(batch_counts[i].cpu().data.numpy()[0])
        outputs.append(
            target_bbox[i, torch.arange(int(batch_counts[i].cpu().data.numpy()[0])).type(torch.cuda.LongTensor)])

    target_bbox = torch.cat(outputs, dim=0)

    loss = F.smooth_l1_loss(rpn_bbox, target_bbox, size_average=True)
    return loss
