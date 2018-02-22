import time

import torch, torch.nn as nn, torch.nn.functional as F

from lib.roi_align.roi_align import CropAndResize, RoIAlign

from . import torch_rcnn_utils



def ROIAlign(feature_maps, rois, config, pool_size, mode='bilinear'):
    """Implements ROI Align on the features.

    Params:
    - pool_shape: [height, width] of the output pooled regions. Usually [7, 7]
    - image_shape: [height, width, chanells]. Shape of input image in pixels

    Inputs:
    - boxes: [batch, num_boxes, (x1, y1, x2, y2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, channels, height, width]

    Output:
    Pooled regions in the shape: [batch, num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """
    """
    [  x2-x1             x1 + x2 - W + 1  ]
    [  -----      0      ---------------  ]
    [  W - 1                  W - 1       ]
    [                                     ]
    [           y2-y1    y1 + y2 - H + 1  ]
    [    0      -----    ---------------  ]
    [           H - 1         H - 1      ]
    """
    # feature_maps= [P2, P3, P4, P5]
    rois = rois.detach()
    crop_resize = CropAndResize(pool_size, pool_size, 0)

    roi_number = rois.size()[1]

    pooled = rois.data.new(
        config.IMAGES_PER_GPU * rois.size(
            1), 256, pool_size, pool_size).zero_()

    rois = rois.view(
        config.IMAGES_PER_GPU * rois.size(1),
        4)

    # Loop through levels and apply ROI pooling to each. P2 to P5.
    x_1 = rois[:, 0]
    y_1 = rois[:, 1]
    x_2 = rois[:, 2]
    y_2 = rois[:, 3]

    roi_level = torch_rcnn_utils.log2_graph(
        torch.div(torch.sqrt((y_2 - y_1) * (x_2 - x_1)), 224.0))

    roi_level = torch.clamp(torch.clamp(
        torch.add(torch.round(roi_level), 4), min=2), max=5)

    # P2 is 256x256, P3 is 128x128, P4 is 64x64, P5 is 32x32
    # P2 is 4, P3 is 8, P4 is 16, P5 is 32
    for i, level in enumerate(range(2, 6)):

        scaling_ratio = 2 ** level

        height = float(config.IMAGE_MAX_DIM) / scaling_ratio
        width = float(config.IMAGE_MAX_DIM) / scaling_ratio

        ixx = torch.eq(roi_level, level)

        box_indices = ixx.view(-1).int() * 0
        ix = torch.unsqueeze(ixx, 1)
        level_boxes = torch.masked_select(rois, ix)
        if len(level_boxes) == 0 or level_boxes.size()[0] == 0:
            continue
        level_boxes = level_boxes.view(-1, 4)

        crops = crop_resize(feature_maps[i], torch.div(
            level_boxes, float(config.IMAGE_MAX_DIM)
        )[:, [1, 0, 3, 2]], box_indices)

        indices_pooled = ixx.nonzero()[:, 0]
        pooled[indices_pooled.data, :, :, :] = crops.data

    pooled = pooled.view(config.IMAGES_PER_GPU, roi_number,
                         256, pool_size, pool_size)
    pooled = torch.autograd.Variable(pooled).cuda()
    return pooled


############################################################
#  Bbox Layer
############################################################
class RCNNHead(nn.Module):
    def __init__(self, num_classes, config):

        super(RCNNHead, self).__init__()
        self.num_classes = num_classes
        self.config = config
        # Setup layers
        self.mrcnn_class_conv1 = nn.Conv2d(
            256, 1024, kernel_size=self.config.POOL_SIZE, stride=1, padding=0)
        self.mrcnn_class_bn1 = nn.BatchNorm2d(1024, eps=0.001)

#        self.dropout = nn.Dropout(p=0.5, inplace=True)

        self.mrcnn_class_conv2 = nn.Conv2d(
            1024, 1024, kernel_size=1, stride=1, padding=0)
        self.mrcnn_class_bn2 = nn.BatchNorm2d(1024, eps=0.001)

        # Classifier head
        self.mrcnn_class_logits = nn.Linear(1024, self.num_classes)
        self.mrcnn_bbox_fc = nn.Linear(1024, self.num_classes * 4)

    def forward(self, x, rpn_rois):
        start = time.time()
        x = ROIAlign(x, rpn_rois, self.config, self.config.POOL_SIZE)

        spend = time.time()-start
        print('first roalign', spend)
        roi_number = x.size()[1]

        x = x.view(self.config.IMAGES_PER_GPU * roi_number,
                   256, self.config.POOL_SIZE,
                   self.config.POOL_SIZE)

        x = self.mrcnn_class_conv1(x)
        x = self.mrcnn_class_bn1(x)
        x = F.relu(x, inplace=True)
#        x = self.dropout(x)
        x = self.mrcnn_class_conv2(x)
        x = self.mrcnn_class_bn2(x)
        x = F.relu(x, inplace=True)

        shared = x.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1)
        # Classifier head
        mrcnn_class_logits = self.mrcnn_class_logits(shared)
        mrcnn_probs = F.softmax(mrcnn_class_logits, dim=-1)

        x = self.mrcnn_bbox_fc(shared)
        mrcnn_bbox = x.view(x.size()[0], self.num_classes, 4)

        mrcnn_class_logits = mrcnn_class_logits.view(self.config.IMAGES_PER_GPU,
                                                     roi_number,
                                                     mrcnn_class_logits.size()[-1])
        mrcnn_probs = mrcnn_probs.view(self.config.IMAGES_PER_GPU,
                                       roi_number,
                                       mrcnn_probs.size()[-1])
        # BBox head
        # [batch, boxes, num_classes , (dy, dx, log(dh), log(dw))]
        mrcnn_bbox = mrcnn_bbox.view(self.config.IMAGES_PER_GPU,
                                     roi_number,
                                     self.config.NUM_CLASSES,
                                     4)

        return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


# rcnn head confidence loss
def rcnn_class_loss(target_class_ids, pred_class_logits, active_class_ids, config):
    """Loss for the classifier head of Mask RCNN.
    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    active_class_ids: [batch, num_classes]. Has a value of 1 for
        classes that are in the dataset of the image, and 0
        for classes that are not in the dataset.
    """

    # Find predictions of classes that are not in the dataset.
    pred_class_logits = pred_class_logits.contiguous().view(-1, config.NUM_CLASSES)

    target_class_ids = target_class_ids.contiguous().view(-1).type(torch.cuda.LongTensor)
    # Loss
    loss = F.cross_entropy(
        pred_class_logits, target_class_ids, weight=None, size_average=True)

    # Erase losses of predictions of classes that are not in the active
    # classes of the image.
    #    loss = loss * pred_active

    # Computer loss mean. Use only predictions that contribute
    # to the loss to get a correct mean.
    #    loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
    return loss


# rcnn head bbox loss
def rcnn_bbox_loss(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """
    # Reshape to merge batch and roi dimensions for simplicity.
    target_class_ids = target_class_ids.contiguous().view(-1)
    target_bbox = target_bbox.contiguous().view(-1, 4)
    pred_bbox = pred_bbox.contiguous().view(-1, pred_bbox.size()[2], 4)
    #    print(target_class_ids)

    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indicies.
    positive_roi_ix = torch.gt(target_class_ids, 0)
    #    print(positive_roi_ix)
    positive_roi_class_ids = torch.masked_select(target_class_ids, positive_roi_ix)

    indices = target_class_ids
    #    indices = torch.stack([positive_roi_ix, positive_roi_class_ids], dim=1)
    #    print(indices)
    # Gather the deltas (predicted and true) that contribute to loss
    #    target_bbox = torch.gather(target_bbox, positive_roi_ix)
    #    pred_bbox = torch.gather(pred_bbox, indices)

    loss = F.smooth_l1_loss(pred_bbox, target_bbox, size_average=True)
    return loss

