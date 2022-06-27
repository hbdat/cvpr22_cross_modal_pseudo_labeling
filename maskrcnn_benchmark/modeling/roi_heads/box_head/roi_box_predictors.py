# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.modeling import registry
import torch
from torch import nn


@registry.ROI_BOX_PREDICTOR.register("FastRCNNPredictor")
class FastRCNNPredictor(nn.Module):
    def __init__(self, config, in_channels, is_teacher):
        super(FastRCNNPredictor, self).__init__()
        assert in_channels is not None

        num_inputs = in_channels
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        

        self.embedding_based = config.MODEL.ROI_BOX_HEAD.EMBEDDING_BASED
        if self.embedding_based:
            self.emb_dim = config.MODEL.ROI_BOX_HEAD.EMB_DIM
            self.emb_pred = nn.Linear(num_inputs, self.emb_dim)
            nn.init.normal_(self.emb_pred.weight, mean=0, std=0.01)
            nn.init.constant_(self.emb_pred.bias, 0)
            assert config.MODEL.CLS_AGNOSTIC_BBOX_REG
            num_bbox_reg_classes = 2
            
            # __forward__() can't be used until these are initialized, AFTER the optimizer is made.
            self.num_classes = None
            self.cls_score = None
            if config.MODEL.ROI_BOX_HEAD.FREEZE_EMB_PRED:
                self.emb_pred.weight.requires_grad = False
                self.emb_pred.bias.requires_grad = False
        else:
            self.num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
            num_bbox_reg_classes = 2 if config.MODEL.CLS_AGNOSTIC_BBOX_REG \
                                     else self.num_classes
            self.cls_score = nn.Linear(num_inputs, self.num_classes)

            nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
            nn.init.constant_(self.cls_score.bias, 0)

        self.bbox_pred = nn.Linear(num_inputs, num_bbox_reg_classes * 4)

        self.uncertainty = False

        if self.uncertainty:
            self.uncertain_pred = nn.Linear(num_inputs, 1)
            nn.init.normal_(self.uncertain_pred.weight, mean=0, std=0.001)
            nn.init.constant_(self.uncertain_pred.bias, 0)

        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)
    
    def reparameterize(self, mu, std):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample
        
    def forward(self, x, compute_uncertain=False):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.embedding_based:
            cls_emb = self.emb_pred(x)
            cls_logit = torch.einsum('pe,ce->pc',cls_emb,self.cls_score)
            #cls_logit = self.cls_score(cls_emb)
        else:
            cls_logit = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)

        if self.uncertainty and compute_uncertain:
            scale = self.uncertain_pred(x)          # p1 <-- pd
            scale = torch.exp(0.5*scale) # standard deviation
            if self.training:
                # TODO: Need to sample more !!!
                cls_logit = self.reparameterize(cls_logit,cls_logit*0.0+scale)
            return cls_logit, bbox_pred, scale

        return cls_logit, bbox_pred

    
    def set_class_embeddings(self, embs):
        # if not torch.all(torch.norm(embs[1:,:],dim=-1) - 1.0 < 1e-6):       # <--  check for unit norm excluding the bg class
        #     print('debug embs:',embs)
        
        # assert torch.all(torch.norm(embs[1:,:],dim=-1) - 1.0 < 1e-6)

        device = self.emb_pred.weight.device
        self.num_classes = embs.shape[0]
        self.cls_score = embs.to(device)

        
@registry.ROI_BOX_PREDICTOR.register("FPNPredictor")
class FPNPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FPNPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        representation_size = in_channels

        if cfg.MODEL.ROI_BOX_HEAD.EMBEDDING_BASED:
            raise NotImplementedError
        
        self.cls_score = nn.Linear(representation_size, num_classes)
        num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


def make_roi_box_predictor(cfg, in_channels, is_teacher):
    func = registry.ROI_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR]
    return func(cfg, in_channels, is_teacher)
