# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.modeling import registry
import torch
from torch import nn


@registry.ROI_BOX_PREDICTOR.register("OMPFastRCNNPredictor")
class OMPFastRCNNPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(OMPFastRCNNPredictor, self).__init__()
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

        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)
        
    def forward(self, x):
        x_map = x                   #[p,d,w,h]
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.embedding_based:
            cls_emb = self.emb_pred(x)
            cls_logit = torch.einsum('pe,ce->pc',cls_emb,self.cls_score)
            #cls_logit = self.cls_score(cls_emb)
        else:
            cls_logit = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        
        idx_cls = torch.argmax(cls_logit,dim=-1)                            #[p]

        x_map_flatten = x_map.permute([0,2,3,1])        # [p,w,h,d] <- [p,d,w,h]
        att_map = self.emb_pred(x_map_flatten)          # [p,w,h,e] <- [p,w,h,d]
        att_map = torch.einsum('pwhe,pe->pwh',att_map,self.cls_score[idx_cls])         # [p,w,h]
        return cls_logit, bbox_pred, att_map

    
    def set_class_embeddings(self, embs):
        device = self.emb_pred.weight.device
        self.num_classes = embs.shape[0]
        self.cls_score = embs.to(device)

def make_roi_box_predictor(cfg, in_channels):
    func = registry.ROI_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR]
    return func(cfg, in_channels)
