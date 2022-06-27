# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList

from maskrcnn_benchmark.modeling.backbone import build_backbone
from maskrcnn_benchmark.modeling.rpn.rpn import build_rpn
from .roi_heads.roi_heads import build_roi_heads
from maskrcnn_benchmark.modeling.language_backbone.transformers import BERT
import torch.nn.functional as F
import numpy as np
from maskrcnn_benchmark.data.datasets.helper.lvis_v1_categories import LVIS_CATEGORIES
from maskrcnn_benchmark.data.datasets.helper.parser import normalize_class_names
import copy

from maskrcnn_benchmark.utils.comm import get_rank, get_world_size
import pickle
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker

class SoftTeacherBaseline(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(SoftTeacherBaseline, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels, is_teacher = True)
        self.roi_heads_student = build_roi_heads(cfg, self.backbone.out_channels)
        self.bert = BERT(cfg)

        self.fix_rpn = cfg.MODEL.RPN.DONT_TRAIN
        assert self.fix_rpn == True

        self.output_dir = cfg.OUTPUT_DIR
        self.class_names = None
        self.mask_on = cfg.MODEL.MASK_ON

        self.consistency_transform = None

        '''freeze parameters'''
        if self.fix_rpn:
            for p in self.rpn.parameters():
                p.requires_grad = False
        
        for p in self.backbone.parameters():
            p.requires_grad = False
            
        for p in self.roi_heads.parameters():
            p.requires_grad = False

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.loss_name = ['loss_box_reg', 'loss_classifier', 'loss_mask']
        cap_vocab = ['']*len(LVIS_CATEGORIES)
        for item in LVIS_CATEGORIES:
            id = item['id']-1       # convert to 0 base
            cap_vocab[id] = item['name']
        self.cap_vocab = normalize_class_names(cap_vocab)
        self.is_first_run = True
        self.iter = torch.zeros(1).long()

        self.masker = Masker(threshold=0.5, padding=1)
        self.lambda_pseudo_label = cfg.MODEL.LAMBDA_PSEUDO_LABEL
        self.resume = cfg.MODEL.RESUME

    def is_same_cls_score(self):
        if self.roi_heads_student.box.predictor.cls_score is None:
            return False
        if len(self.roi_heads_student.box.predictor.cls_score) != len(self.roi_heads.box.predictor.cls_score):
            return False
        if torch.any(self.roi_heads_student.box.predictor.cls_score != self.roi_heads.box.predictor.cls_score):
            return False
        
        return True

    def prepare_model(self):
        self.cap_embs = self.extract_emb(self.cap_vocab)

        if not self.is_same_cls_score():
            self.roi_heads_student.box.predictor.cls_score = self.roi_heads.box.predictor.cls_score
            ### the code switch this head everytime it is evaluated on a new dataset !!!
        
        if self.iter == 0 and not self.resume:
            self.roi_heads_student.load_state_dict(copy.deepcopy(self.roi_heads.state_dict()),strict=False)
            self.iter += 1      # might double count once
            print('copy teacher parameters')

    def extract_emb(self, words):
        encoded_word_list = self.bert(words)
        mask = (1 - encoded_word_list['special_tokens_mask']).to(torch.float32)
        embeddings = (encoded_word_list['input_embeddings'] * mask[:, :, None]).sum(1) / mask.sum(1)[:, None]  # this summation ensures that there is always a vector output

        embeddings = F.normalize(embeddings, dim=-1)   #[n_class,dim_emb]

        return embeddings

    def generate_pseudo_label(self, features, proposals, caps, targets):
        class_embs = self.roi_heads['box'].predictor.cls_score
        self.roi_heads['box'].predictor.set_class_embeddings(torch.zeros((1,768)))     # << set dummy values
        self.roi_heads.eval()   # >> if set at training mode, the model would use gt to subsample regions
        package_x, results, _ = self.roi_heads(features, proposals, None, bbox_only = True)
        assert len(results[0]) == len(proposals[0])

        x = package_x['bbox']  
        f_regions = self.avgpool(x)                                                     #[P,dim_v,1]
        f_regions = f_regions.view(f_regions.size(0), -1)                               #[P,dim_v]
        cls_embs = self.roi_heads.box.predictor.emb_pred(f_regions)                     #[P,dim_emb]
        cls_embs = cls_embs.split([len(p) for p in proposals])                          #list([p,dim_emb])
        
        '''
        alignment
        '''
        zip_package = zip(cls_embs, caps, results, targets)
        pseudo_labels = []
        for emb_img, words, result_img, target_img in zip_package:
            
            if len(words) == 0:      # cannot find any noun phrase
                dummy_box = BoxList(np.zeros((0,4)), result_img.size, mode=result_img.mode)
                pseudo_labels.append(dummy_box)
                continue
            
            w_cap_img = self.extract_emb(words)
            
            region_scores = torch.einsum('pd,wd->pw',emb_img,w_cap_img)
            region_prop = F.softmax(region_scores,dim=-1)
            vs, cls_idxs = torch.max(region_prop,dim=-1)            #[p]

            selected_region = torch.argsort(vs,descending=True)[:2] # get two most confident regions
            selected_cls = cls_idxs[selected_region]
            
            aligned_region_scores, idx_aligned_regions = vs[selected_region], selected_region      #[w]<-[pw]
            
            pseudo_label_img = result_img[idx_aligned_regions]
            
            #pseudo_label_img.add_field('joined_words', '/'.join([words[selected_cls]]))

            weights = torch.sigmoid(aligned_region_scores)
            embs = emb_img[idx_aligned_regions]
            aligned_region_scores = torch.sigmoid(aligned_region_scores)        # << sigmoid normalization
            
            ids_cap = selected_cls
            pseudo_label_img.add_field("labels",ids_cap)
            pseudo_label_img.add_field("scores",aligned_region_scores)
            pseudo_label_img.add_field("weights",weights)
            pseudo_label_img.add_field("embs",embs)
            pseudo_labels.append(pseudo_label_img)      #this is the similar format with target tensor
        
        ### mask pass ###
        if self.mask_on:
            package_x, results, _ = self.roi_heads(features, pseudo_labels, None, bbox_only = False)
            for results_img, pseudo_label_img in zip(results,pseudo_labels):
                masks = results_img.get_field('mask')                   #[p,1,14,14] , already binary (boolean)
                masks = self.masker([masks], [pseudo_label_img])[0]     #[p,1,w,h]
                masks = masks[:,0]                                      #[p,w,h] 
                masks = SegmentationMask(masks, pseudo_label_img.size, mode='mask')
                pseudo_label_img.add_field('masks',masks)

        self.roi_heads['box'].predictor.set_class_embeddings(class_embs)
        return pseudo_labels

    def compute_dummy_loss(self):
        loss = 0.0
        for p in self.roi_heads_student.parameters():
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
        assert self.roi_heads

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        
        images = to_image_list(images)
        features = self.backbone(images.tensors)

        if self.training:
            self.prepare_model()
            dummy_loss = self.compute_dummy_loss()

            idxs_cap = [idx for idx,t in enumerate(targets) if t.get_field('nn_caption')!='']

            if len(idxs_cap) > 0:
                self.rpn.eval()
                proposals, _ = self.rpn(images, features, None)

                cap_features = [features[idx] for idx in idxs_cap]
                cap_proposals = [proposals[idx] for idx in idxs_cap]
                cap_targets = [targets[idx] for idx in idxs_cap]
            
                caps = [self.cap_vocab]   #[t.get_field('nn_caption').split('/') for t in cap_targets]

                with torch.no_grad():
                    pseudo_targets = self.generate_pseudo_label(cap_features, cap_proposals, caps, cap_targets)

                ### set vocab to lvis ###
                embs = F.normalize(self.cap_embs,dim=-1) #self.cap_embs#
                self.roi_heads_student['box'].predictor.set_class_embeddings(embs)

                try:
                    _,_, loss_pseudo = self.roi_heads_student(cap_features, cap_proposals, pseudo_targets)

                    for k in loss_pseudo:
                        if 'mask' not in k:
                            loss_pseudo[k] *= self.lambda_pseudo_label       
                    
                except Exception as e:
                    print(e)
                    loss_pseudo = {}
            else:
                loss_pseudo = {}
            
            ### attach pseudo keyword in loss_dict ###
            new_loss_pseudo = {}
            for k in self.loss_name:
                if k in loss_pseudo:
                    new_loss_pseudo['{}_pseudo'.format(k)] = loss_pseudo[k]
                else:
                    new_loss_pseudo['{}_pseudo'.format(k)] = dummy_loss
            loss_pseudo = new_loss_pseudo

            ### seen class training ###
            idxs_gt = [idx for idx,t in enumerate(targets) if t.get_field('is_det') == 'Yes']

            if len(idxs_gt) > 0:
                self.rpn.train()
                proposals_target, _ = self.rpn(images, features, targets)

                gt_features = [features[idx] for idx in idxs_gt]
                gt_proposals = [proposals_target[idx] for idx in idxs_gt]
                gt_targets = [targets[idx] for idx in idxs_gt]

                ### set back vocab to mscoco seen ###
                class_embs = self.roi_heads['box'].predictor.cls_score
                assert len(self.class_names) == len(class_embs)
                embs = F.normalize(class_embs,dim=-1)#class_embs#

                self.roi_heads_student['box'].predictor.set_class_embeddings(embs)
                try:
                    _,_, loss_gt = self.roi_heads_student(gt_features, gt_proposals, gt_targets)
                except Exception as e:
                    print(e)
                    loss_gt = {}
            else:
                loss_gt = {}

            for k in self.loss_name:
                if k not in loss_gt:
                    loss_gt[k] = dummy_loss

            losses = {}
            losses.update(loss_pseudo)
            losses.update(loss_gt)
            
            self.iter += 1

            return losses
        else:
            self.rpn.eval()
            proposals, _ = self.rpn(images, features, None)

            class_embs = self.roi_heads['box'].predictor.cls_score 
            embs = F.normalize(class_embs,dim=-1)#class_embs#
            self.roi_heads_student['box'].predictor.set_class_embeddings(embs)
            x, result_student, _ = self.roi_heads_student(features, proposals, targets)

            return result_student
