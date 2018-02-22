"""
Mask R-CNN
The main Mask R-CNN model implemenetation.
"""
import datetime
import glob
import itertools
import json
import logging
import math
import os
import random
import re
import time
import sys
from collections import OrderedDict

import numpy as np
import scipy.misc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable

from tasks.bbox.generate_anchors import generate_pyramid_anchors
from tasks.merge_task import build_detection_targets

from . import resnet_backbone, rpn_head, rcnn_head, mrcnn_head

def to_variable(numpy_data, volatile=False):
    numpy_data = numpy_data.astype(np.float32)
    torch_data = torch.from_numpy(numpy_data).float()
    variable = Variable(torch_data, volatile=volatile)
    return variable
    

       


############################################################
#  Main Class of MASK-RCNN
############################################################
class MaskRCNN(nn.Module):
    """
    Encapsulates the Mask RCNN model functionality.
    
    """
    def __init__(self, config, mode='inference'):
        super(MaskRCNN, self).__init__()
        """
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        self.config = config
        self.mode = mode
        self.resnet_graph = resnet_backbone.ResNetBackbone(
            resnet_backbone.Bottleneck, [3, 4, 23, 3], stage5=True)

        # feature pyramid layers:
        self.fpn_c5p5 = nn.Conv2d(
            512 * 4, 256, kernel_size=1, stride=1, padding=0)
        self.fpn_c4p4 = nn.Conv2d(
            256 * 4, 256, kernel_size=1, stride=1, padding=0)
        self.fpn_c3p3 = nn.Conv2d(
            128 * 4, 256, kernel_size=1, stride=1, padding=0)
        self.fpn_c2p2 = nn.Conv2d(
            64 * 4, 256, kernel_size=1, stride=1,  padding=0)

        self.fpn_p2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.fpn_p3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.fpn_p4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.fpn_p5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.scale_ratios = [4, 8, 16, 32]
        self.fpn_p6 = nn.MaxPool2d(
            kernel_size=1, stride=2, padding=0, ceil_mode=False)

        self.anchors = generate_pyramid_anchors(self.config.RPN_ANCHOR_SCALES,
                                                self.config.RPN_ANCHOR_RATIOS,
                                                self.config.BACKBONE_SHAPES,
                                                self.config.BACKBONE_STRIDES,
                                                self.config.RPN_ANCHOR_STRIDE)
        self.anchors = self.anchors.astype(np.float32)


        # RPN Model
        self.rpn = rpn_head.RPNHead(
            256, len(self.config.RPN_ANCHOR_RATIOS), self.config.RPN_ANCHOR_STRIDE)

        self.rpn_class = rcnn_head.RCNNHead(config.NUM_CLASSES, config)

        self.rpn_mask = mrcnn_head.MaskRCNNHead(config.NUM_CLASSES, config)


        self.rpn_refine = rpn_head.RPNRefine(self.anchors, config, mode=mode)

        self._initialize_weights()

    def forward(self, x):

        start = time.time()
        saved_for_loss = []
        
        C1, C2, C3, C4, C5 = self.resnet_graph(x)
        
        resnet_time = time.time()
       
        print('resnet spend', resnet_time-start)
        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the thead (stage 5), so we pick the 4th item in the list.
        P5 = self.fpn_c5p5(C5)
        P4 = self.fpn_c4p4(C4) + F.upsample(P5,
                                            scale_factor=2, mode='bilinear')
        P3 = self.fpn_c3p3(C3) + F.upsample(P4,
                                            scale_factor=2, mode='bilinear')
        P2 = self.fpn_c2p2(C2) + F.upsample(P3,
                                            scale_factor=2, mode='bilinear')

        # Attach 3x3 conv to all P layers to get the final feature maps.
        # P2 is 256, P3 is 128, P4 is 64, P5 is 32
        P2 = self.fpn_p2(P2)
        P3 = self.fpn_p3(P3)
        P4 = self.fpn_p4(P4)
        P5 = self.fpn_p5(P5)
        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        P6 = self.fpn_p6(P5)

        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [P2, P3, P4, P5, P6]

        self.mrcnn_feature_maps = [P2, P3, P4, P5]

        
        # Loop through pyramid layers
        rpn_class_logits_outputs = []
        rpn_class_outputs = []
        rpn_bbox_outputs = []

        for p in rpn_feature_maps:
            rpn_class_logits, rpn_probs, rpn_bbox = self.rpn(p)
            rpn_class_logits_outputs.append(rpn_class_logits)
            rpn_class_outputs.append(rpn_probs)
            rpn_bbox_outputs.append(rpn_bbox)

        rpn_class_logits = torch.cat(rpn_class_logits_outputs, dim=1)
        rpn_class = torch.cat(rpn_class_outputs, dim=1)
    
        rpn_bbox = torch.cat(rpn_bbox_outputs, dim=1)

        rpn_rois = self.rpn_refine(rpn_class, rpn_bbox)
 
 
        spend = time.time()-resnet_time
        
        print('fpn spend 1', spend)
 
        rcnn_class_logits, rcnn_class, rcnn_bbox = self.rpn_class(
            self.mrcnn_feature_maps, rpn_rois)

        mrcnn_masks_logits = self.rpn_mask(self.mrcnn_feature_maps, rpn_rois)


        if self.mode == 'training':
            return [rpn_class_logits, rpn_class, rpn_bbox, rpn_rois, 
                    rcnn_class_logits, rcnn_class, rcnn_bbox,
                    mrcnn_masks_logits],\
                    [rpn_class_logits, rpn_class, rpn_bbox, rpn_rois, 
                    rcnn_class_logits, rcnn_class, rcnn_bbox,
                    mrcnn_masks_logits]
        else:
            return [rpn_class_logits, rpn_class, rpn_bbox, rpn_rois, 
                    rcnn_class_logits, rcnn_class, rcnn_bbox,
                    mrcnn_masks_logits]

            
        
    @staticmethod
    def build_loss(saved_for_loss, ground_truths, config):
        #create dict to save loss for visualization
        saved_for_log = OrderedDict()
        #unpack saved log
        predict_rpn_class_logits, predict_rpn_class,\
        predict_rpn_bbox, predict_rpn_rois,\
        predict_mrcnn_class_logits, predict_mrcnn_class,\
        predict_mrcnn_bbox, predict_mrcnn_masks_logits = saved_for_loss

        batch_rpn_match, batch_rpn_bbox, \
        batch_gt_class_ids, batch_gt_boxes,\
        batch_gt_masks, active_class_ids = ground_truths
        

        rpn_rois = predict_rpn_rois.cpu().data.numpy() 
        rpn_rois = rpn_rois[:, :, [1, 0, 3, 2]]
        batch_rois, batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_mask = stage2_target(rpn_rois, batch_gt_class_ids, batch_gt_boxes, batch_gt_masks, config)

#        print(np.sum(batch_mrcnn_class_ids))
        batch_mrcnn_mask = batch_mrcnn_mask.transpose(0, 1, 4, 2, 3)
        batch_mrcnn_class_ids = to_variable(
            batch_mrcnn_class_ids).cuda()
        batch_mrcnn_bbox = to_variable(batch_mrcnn_bbox).cuda()
        batch_mrcnn_mask = to_variable(batch_mrcnn_mask).cuda()   
             
#        print(batch_mrcnn_class_ids)
        # RPN branch loss->classification
        rpn_cls_loss = rpn_head.rpn_class_loss(
            batch_rpn_match, predict_rpn_class_logits)
        
        # RPN branch loss->bbox            
        rpn_box_loss = rpn_head.rpn_bbox_loss(
            batch_rpn_bbox, batch_rpn_match, predict_rpn_bbox, config)

        # bbox branch loss->classification
        rcnn_box_loss = rcnn_head.rcnn_bbox_loss(
            batch_mrcnn_bbox, batch_mrcnn_class_ids, predict_mrcnn_bbox)

        # bbox branch loss->bbox
        rcnn_cls_loss = rcnn_head.rcnn_class_loss(
            batch_mrcnn_class_ids, predict_mrcnn_class_logits, active_class_ids, config)
            
        # mask branch loss
        mrcnn_mask_loss = mrcnn_head.mrcnn_mask_loss(
            batch_mrcnn_mask, batch_mrcnn_class_ids, predict_mrcnn_masks_logits)                           

        total_loss = rpn_cls_loss + rpn_box_loss + rcnn_cls_loss + rcnn_box_loss + mrcnn_mask_loss
        saved_for_log['rpn_cls_loss'] = rpn_cls_loss.data[0]
        saved_for_log['rpn_box_loss'] = rpn_box_loss.data[0]
        saved_for_log['rcnn_cls_loss'] = rcnn_cls_loss.data[0]
        saved_for_log['rcnn_box_loss'] = rcnn_box_loss.data[0]
        saved_for_log['mrcnn_mask_loss'] = mrcnn_mask_loss.data[0]
        saved_for_log['total_loss'] = total_loss.data[0]

        return total_loss, saved_for_log
            
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def stage2_target(rpn_rois, gt_class_ids, gt_boxes, gt_masks, config):
     
    batch_rois = []
    batch_mrcnn_class_ids = []
    batch_mrcnn_bbox = []
    batch_mrcnn_mask = []
                                
    for i in range(config.IMAGES_PER_GPU):
        rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask =\
        build_detection_targets(
        rpn_rois[i], gt_class_ids[i], gt_boxes[i], gt_masks[i], config)
    
        batch_rois.append(rois)
        batch_mrcnn_class_ids.append(mrcnn_class_ids)
        batch_mrcnn_bbox.append(mrcnn_bbox)
        batch_mrcnn_mask.append(mrcnn_mask)
        
    batch_rois = np.array(batch_rois)
    batch_mrcnn_class_ids = np.array(batch_mrcnn_class_ids)
    batch_mrcnn_bbox = np.array(batch_mrcnn_bbox)
    batch_mrcnn_mask = np.array(batch_mrcnn_mask)
    return batch_rois, batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_mask
                        
