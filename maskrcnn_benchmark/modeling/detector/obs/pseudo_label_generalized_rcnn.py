# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
from maskrcnn_benchmark.modeling.language_backbone.transformers import BERT
import torch.nn.functional as F
import numpy as np

class PseudoLabelGeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(PseudoLabelGeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels, no_filter=True)
        self.bert = BERT(cfg)

        self.fix_rpn = cfg.MODEL.RPN.DONT_TRAIN
        if self.fix_rpn:
            for p in self.rpn.parameters():
                p.requires_grad = False

        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def extract_emb(self, words):

        encoded_word_list = self.bert(words)
        mask = (1 - encoded_word_list['special_tokens_mask']).to(torch.float32)
        embeddings = (encoded_word_list['input_embeddings'] * mask[:, :, None]).sum(1) / mask.sum(1)[:, None]  # this summation ensures that there is always a vector output

        embeddings = F.normalize(embeddings, dim=-1)   #[n_class,dim_emb]

        return embeddings

    def generate_pseudo_label(self, features, proposals, caps, targets):
        
        self.roi_heads['box'].predictor.set_class_embeddings(np.zeros((1,768),dtype=np.float32))     # << set dummy values

        self.roi_heads.eval()   # >> if set at training mode, the model would use gt to subsample regions
        package_x, results, _ = self.roi_heads(features, proposals, None)

        assert len(results[0]) == len(proposals[0])

        x = package_x['bbox']
        f_regions = self.avgpool(x)
        f_regions = f_regions.view(f_regions.size(0), -1)                               #[P,dim_v]
        cls_embs = self.roi_heads.box.predictor.emb_pred(f_regions)                     #[P,dim_emb]
        cls_embs = cls_embs.split([len(p) for p in proposals])                          #list([p,dim_emb])
        
        '''
        alignment
        '''
        zip_package = zip(cls_embs, caps, results, targets)
        pseudo_labels = []
        for emb_img, words, result_img, target_img in zip_package:
            # label = target_img.get_field('labels').long()
            # seen_idxs = target_img.get_field('seen_idxs').long()
            # bbox = target_img.bbox
            
            if len(words) == 0:      # cannot find any noun phrase
                dummy_box = BoxList(np.zeros((0,4)), result_img.size, mode=result_img.mode)
                pseudo_labels.append(dummy_box)
                continue
            
            w_cap_img = self.extract_emb(words)
            
            fragment_score = torch.einsum('pd,wd->pw',emb_img,w_cap_img)
            
            align_fragment_score, idx_align_fragment = torch.max(fragment_score,dim=0)      #[w],[w]
            
            pseudo_label_img = result_img[idx_align_fragment]
             
            pseudo_label_img.add_field('joined_words', '/'.join(words))
            pseudo_labels.append(pseudo_label_img)
        
        return pseudo_labels

    def forward(self, images, targets):
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
        
        assert self.training == False
        assert self.roi_heads

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.fix_rpn:
            self.rpn.eval()
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        proposals, _ = self.rpn(images, features, targets)
        import pdb; pdb.set_trace()

        caps = targets#[t.get_field('nn_caption') for t in targets]
        result = self.generate_pseudo_label(features, proposals, caps, targets)

        return result
