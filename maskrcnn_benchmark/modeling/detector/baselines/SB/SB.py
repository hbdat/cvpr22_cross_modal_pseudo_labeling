# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from maskrcnn_benchmark.modeling.backbone import build_backbone
from maskrcnn_benchmark.modeling.rpn.rpn import build_rpn
from maskrcnn_benchmark.modeling.roi_heads.roi_heads import build_roi_heads


class SBBaseline(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(SBBaseline, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
        self.fix_rpn = cfg.MODEL.RPN.DONT_TRAIN
        if self.fix_rpn:
            for p in self.rpn.parameters():
                p.requires_grad = False
        
        self.loss_name = ['loss_box_reg', 'loss_classifier', 'loss_mask']

    def compute_dummy_loss(self):
        loss = 0.0
        for p in self.parameters():
            loss += torch.sum(p)*0.0

        return loss

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)
                [or (list[ndarray]) with image-level labels in weakly supervised settings]

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.fix_rpn:
            self.rpn.eval()
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)

        if self.roi_heads:
            try:
                x, result, detector_losses = self.roi_heads(features, proposals, targets)
            except Exception as e:
                dummy_loss = self.compute_dummy_loss()
                detector_losses = {}
                for k in self.loss_name:
                    detector_losses[k] = dummy_loss
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result
