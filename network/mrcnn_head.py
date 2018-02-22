import torch, torch.nn as nn, torch.nn.functional as F

from . import rcnn_head


############################################################
#  Mask Layer
############################################################
class MaskRCNNHead(nn.Module):

    def __init__(self, num_classes, config):
        """Builds the computation graph of the mask head of Feature Pyramid Network.

        rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
              coordinates.
        feature_maps: List of feature maps from diffent layers of the pyramid,
                      [P2, P3, P4, P5]. Each has a different resolution.
        image_shape: [height, width, depth]
        pool_size: The width of the square feature map generated from ROI Pooling.
        num_classes: number of classes, which determines the depth of the results

        Returns: Masks [batch, roi_count, height, width, num_classes]
        """
        # ROI Pooling
        # Shape: [batch, boxes, pool_height, pool_width, channels]
        super(MaskRCNNHead, self).__init__()
        self.num_classes = num_classes
        self.config = config
        # Setup layers
        self.mrcnn_mask_conv1 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1)
        self.mrcnn_mask_bn1 = nn.BatchNorm2d(256, eps=0.001)

        self.mrcnn_mask_conv2 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1)
        self.mrcnn_mask_bn2 = nn.BatchNorm2d(256, eps=0.001)

        self.mrcnn_mask_conv3 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1)
        self.mrcnn_mask_bn3 = nn.BatchNorm2d(256, eps=0.001)

        self.mrcnn_mask_conv4 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1)
        self.mrcnn_mask_bn4 = nn.BatchNorm2d(256, eps=0.001)

        self.mrcnn_mask_deconv = nn.ConvTranspose2d(
            256, 256, kernel_size=2, stride=2)

        self.mrcnn_mask = nn.Conv2d(
            256, self.num_classes, kernel_size=1, stride=1)

    def forward(self, x, rpn_rois):

        x = rcnn_head.ROIAlign(x, rpn_rois, self.config, self.config.MASK_POOL_SIZE)

        roi_number = x.size()[1]

        # merge batch and roi number together
        x = x.view(self.config.IMAGES_PER_GPU * roi_number,
                   256, self.config.MASK_POOL_SIZE,
                   self.config.MASK_POOL_SIZE)

        x = self.mrcnn_mask_conv1(x)
        x = self.mrcnn_mask_bn1(x)
        x = F.relu(x, inplace=True)

        x = self.mrcnn_mask_conv2(x)
        x = self.mrcnn_mask_bn2(x)
        x = F.relu(x, inplace=True)

        x = self.mrcnn_mask_conv3(x)
        x = self.mrcnn_mask_bn3(x)
        x = F.relu(x, inplace=True)

        x = self.mrcnn_mask_conv4(x)
        x = self.mrcnn_mask_bn4(x)
        x = F.relu(x, inplace=True)

        x = self.mrcnn_mask_deconv(x)

        x = F.relu(x, inplace=True)
        x = self.mrcnn_mask(x)

        # resize to add the batch dim
        x = x.view(self.config.IMAGES_PER_GPU,
                   roi_number,
                   self.config.NUM_CLASSES,
                   self.config.MASK_POOL_SIZE * 2,
                   self.config.MASK_POOL_SIZE * 2)
        return x



# rcnn head mask loss
def mrcnn_mask_loss(target_masks, target_class_ids, pred_masks_logits):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    # Reshape for simplicity. Merge first two dimensions into one.
    target_class_ids = target_class_ids.view(-1)

    loss = F.binary_cross_entropy_with_logits(pred_masks_logits, target_masks)
    return loss
