# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import ConvTranspose2d
from maskrcnn_benchmark.modeling import registry
import torch


@registry.ROI_MASK_PREDICTOR.register("OMPMaskRCNNC4Predictor")        # << default mask head
class OMPMaskRCNNC4Predictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(OMPMaskRCNNC4Predictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        dim_reduced = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1]
        num_inputs = in_channels
        
        if cfg.MODEL.CLS_AGNOSTIC_MASK:
            num_classes = 2
        
        self.conv5_mask = ConvTranspose2d(num_inputs, dim_reduced, 2, 2, 0)
        self.mask_fcn_logits = Conv2d(dim_reduced, num_classes, 1, 1, 0)

        self.deconv_upscale = ConvTranspose2d(1, 1, 2, 2, 0)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x, att_map):
        x_ = F.relu(self.conv5_mask(x))
        mask_logits =  self.mask_fcn_logits(x_)

        mask_logits += self.deconv_upscale(att_map[:,None,:,:])
        return mask_logits

def make_roi_mask_predictor(cfg, in_channels):
    func = registry.ROI_MASK_PREDICTOR[cfg.MODEL.ROI_MASK_HEAD.PREDICTOR]
    return func(cfg, in_channels)
