# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import ConvTranspose2d
from maskrcnn_benchmark.modeling import registry
import torch

@registry.ROI_MASK_PREDICTOR.register("MaskRCNNC4Predictor")        # << default mask head
class MaskRCNNC4Predictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(MaskRCNNC4Predictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        dim_reduced = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1]
        num_inputs = in_channels
        
        if cfg.MODEL.CLS_AGNOSTIC_MASK:
            num_classes = 2
        
        self.conv5_mask = ConvTranspose2d(num_inputs, dim_reduced, 2, 2, 0)
        self.mask_fcn_logits = Conv2d(dim_reduced, num_classes, 1, 1, 0)
        
        self.uncertainty = cfg.MODEL.UNCERTAINTY

        if self.uncertainty:
            self.uncertain_pred = Conv2d(dim_reduced, 1, 1, 1, 0)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

        if self.uncertainty:
            nn.init.normal_(self.uncertain_pred.weight, mean=0, std=0.001)
            nn.init.constant_(self.uncertain_pred.bias, 1)

    def reparameterize(self, mu, std):  #mu [p,1,w,h]; std [p,1,w,h]
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        n_samples = 1
        eps = torch.randn(n_samples,*std.shape).to(mu.device)    # [n_s,p,1,w,h]
        mu = mu[None]                       # [1,p,1,w,h]
        std = std[None]                     # [1,p,1,w,h]
        sample = mu + (eps * std)             # sampling as if coming from the input space
        return sample                               # [n_s,p,1,w,h]

    def forward(self, x, compute_uncertain=False):
        x_ = F.relu(self.conv5_mask(x))
        mask_logits =  self.mask_fcn_logits(x_)
        if self.uncertainty and compute_uncertain:
            #x_ = F.sigmoid(self.uncertain_conv5_mask(x))
            scale = self.uncertain_pred(x_.detach())          # p1 <-- pd
            scale = torch.exp(0.5*scale) # standard deviation
            #scale = torch.clamp(scale, min=0.0001,max=10)
            if self.training:
                mask_logits = self.reparameterize(mask_logits,mask_logits*0.0+scale)
            return mask_logits, scale
        else:
            return mask_logits


@registry.ROI_MASK_PREDICTOR.register("MaskRCNNConv1x1Predictor")
class MaskRCNNConv1x1Predictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(MaskRCNNConv1x1Predictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        
        if cfg.MODEL.CLS_AGNOSTIC_MASK:
            num_classes = 2
        
        num_inputs = in_channels

        self.mask_fcn_logits = Conv2d(num_inputs, num_classes, 1, 1, 0)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.mask_fcn_logits(x)


def make_roi_mask_predictor(cfg, in_channels):
    func = registry.ROI_MASK_PREDICTOR[cfg.MODEL.ROI_MASK_HEAD.PREDICTOR]
    return func(cfg, in_channels)



### other uncertainty variants ###
@registry.ROI_MASK_PREDICTOR.register("DropOut_MaskRCNNC4Predictor")        # << default mask head
class DropOut_MaskRCNNC4Predictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(DropOut_MaskRCNNC4Predictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        dim_reduced = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1]
        num_inputs = in_channels
        
        if cfg.MODEL.CLS_AGNOSTIC_MASK:
            num_classes = 2
        
        self.conv5_mask = ConvTranspose2d(num_inputs, dim_reduced, 2, 2, 0)
        self.mask_fcn_logits = Conv2d(dim_reduced, num_classes, 1, 1, 0)
        
        self.uncertainty = cfg.MODEL.UNCERTAINTY

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x, compute_uncertain=False):
        x_ = F.relu(self.conv5_mask(x))
        mask_logits =  self.mask_fcn_logits(x_)
        if self.uncertainty and compute_uncertain:
            x_dropout = x_[None].expand(10,-1,-1,-1,-1)
            x_dropout = F.dropout(x_dropout, 0.5, self.training).split(1,dim=0)

            mask_logits_dropout = [self.mask_fcn_logits(x[0])[None] for x in x_dropout]
            mask_logits_dropout = torch.cat(mask_logits_dropout,dim=0)
            prop = torch.sigmoid(mask_logits_dropout).mean(dim=0)    #[p,1,w,h]
            eps = 1e-8
            entropy = -prop*torch.log2(prop+eps) - (1-prop)*torch.log2(1-prop+eps)
            scale = 1 - entropy.mean(dim=[1,2,3])
            return mask_logits, scale
        else:
            return mask_logits

@registry.ROI_MASK_PREDICTOR.register("PixelScore_MaskRCNNC4Predictor")        # << default mask head
class PixelScore_MaskRCNNC4Predictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(PixelScore_MaskRCNNC4Predictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        dim_reduced = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1]
        num_inputs = in_channels
        
        if cfg.MODEL.CLS_AGNOSTIC_MASK:
            num_classes = 2
        
        self.conv5_mask = ConvTranspose2d(num_inputs, dim_reduced, 2, 2, 0)
        self.mask_fcn_logits = Conv2d(dim_reduced, num_classes, 1, 1, 0)
        
        self.uncertainty = cfg.MODEL.UNCERTAINTY

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x, compute_uncertain=False):
        x_ = F.relu(self.conv5_mask(x))
        mask_logits =  self.mask_fcn_logits(x_)
        if self.uncertainty and compute_uncertain:
            prob = torch.sigmoid(mask_logits[:,1,:,:])      #[p,w,h]
            hcm =  prob > 0.2
            eps = 1e-8
            scale = (prob*hcm).sum(dim=[1,2])/(hcm.sum(dim=[1,2])+eps)
            return mask_logits, scale
        else:
            return mask_logits