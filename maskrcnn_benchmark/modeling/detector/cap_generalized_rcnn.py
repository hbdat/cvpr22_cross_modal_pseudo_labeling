# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 14:37:51 2021

@author: badat
"""

import torch
from torch import nn
import torch.nn.functional as F

from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.bounding_box import BoxList

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads

from maskrcnn_benchmark.modeling.language_backbone.transformers import BERT
import copy

class CapGeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(CapGeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.rpn_student = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
        self.roi_heads_student = build_roi_heads(cfg, self.backbone.out_channels)
        self.fix_rpn = True#cfg.MODEL.RPN.DONT_TRAIN
        self.fix_backbone = True
        self.bert = BERT(cfg)
        
        self.is_first_run = True
        self.lambda_fragment = 0.01
        # self.dummy_parameter = nn.Parameter(torch.tensor(0.0),requires_grad = True)
        
        self.hit = 0
        self.hit_unseen = 0
        self.all = 0
        self.all_unseen = 0
        
        '''freeze parameters'''
        if self.fix_rpn:
            for p in self.rpn.parameters():
                p.requires_grad = False
        
        # if self.backbone:
        for p in self.backbone.parameters():
            p.requires_grad = False
        
        for p in self.roi_heads.box.parameters():
            p.requires_grad = False
            
        for p in self.roi_heads.parameters():
            p.requires_grad = False
        
        for p in self.rpn_student.parameters():
            p.requires_grad = False
        
        # for p in self.roi_heads_student.mask.parameters():
        #     p.requires_grad = False
        
        '''freeze parameters'''
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def extract_emb(self, words):

        encoded_word_list = self.bert(words)
        mask = (1 - encoded_word_list['special_tokens_mask']).to(torch.float32)
        embeddings = (encoded_word_list['input_embeddings'] * mask[:, :, None]).sum(1) / mask.sum(1)[:, None]  # this summation ensures that there is always a vector output

        embeddings = F.normalize(embeddings, dim=-1)   #[n_class,dim_emb]

        return embeddings

    def pass_teacher(self, features, proposals, w_caps, targets,special_tokens_masks):
        
        ret_alignments = []
        ret_alignment_scores = []
        
        self.roi_heads.eval()   # >> if set at training mode, the model would use gt to subsample regions
        package_x, result_teacher, _ = self.roi_heads(features, proposals, None)
        x = package_x['bbox']
        f_regions = self.avgpool(x)
        f_regions = f_regions.view(f_regions.size(0), -1)
        cls_embs = self.roi_heads.box.predictor.emb_pred(f_regions)                     #[P,dim_emb]
        cls_embs = cls_embs.split([len(p) for p in proposals])          #list([p,dim_emb])
        
        '''
        alignment
        '''
        zip_package = zip(cls_embs,w_caps,proposals,targets,special_tokens_masks)
        new_targets = []
        for emb_img, w_cap_img, proposal_img, target_img, special_tokens_mask_img in zip_package:
            label = target_img.get_field('labels').long()
            seen_idxs = target_img.get_field('seen_idxs').long()
            bbox = target_img.bbox
            
            idxs_valid_token = torch.where(special_tokens_mask_img!=1)[0]
            
            if len(idxs_valid_token) == 0:      # cannot find any noun phrase
                ret_alignments.append(None)
                ret_alignment_scores.append(None)
                continue
            
            w_cap_img = w_cap_img[idxs_valid_token]
            
            # cap_class_alignment = self.roi_heads.box.predictor.cls_score(w_cap_img)
            # cap2class = torch.argmax(cap_class_alignment,dim=1)
            
            fragment_score = torch.einsum('pd,wd->pw',emb_img,w_cap_img)
            
            align_fragment_score, idx_align_fragment = torch.max(fragment_score,dim=0)      #[w],[w]
            
            ret_alignments.append(idx_align_fragment)
            ret_alignment_scores.append(align_fragment_score)
            
            new_bbox = [bbox[seen_idxs]]
            new_label = [label[seen_idxs]]
            
            new_bbox = torch.cat(new_bbox, dim = 0)
            new_label = torch.cat(new_label, dim = 0)
            
            new_target_img = BoxList(new_bbox, target_img.size, mode=target_img.mode) 
            new_target_img.add_field('labels', new_label)
            new_targets.append(new_target_img)
        
        return ret_alignments, ret_alignment_scores, new_targets
            
    def pass_student(self, features, proposals, w_caps, targets,special_tokens_masks, teacher_alignments, teacher_alignment_scores, new_targets):
        
        self.roi_heads_student.eval()   # >> if set at training mode, the model would use gt to subsample regions
        package_x, result_student, _ = self.roi_heads_student(features, proposals, None)
        x_bbox = package_x['bbox']
        x_mask = package_x['mask']
        self.roi_heads_student.train()
        
        f_regions = self.avgpool(x_bbox)
        f_regions = f_regions.view(f_regions.size(0), -1)
        
        loss_dummy = torch.sum(self.roi_heads_student.box.predictor.emb_pred(f_regions))*0.0
        loss_dummy += torch.sum(self.roi_heads_student.box.predictor.bbox_pred(f_regions))*0.0
        loss_dummy += torch.sum(self.roi_heads_student.mask.predictor(x_mask))*0.0
        
        
        cls_embs = self.roi_heads_student.box.predictor.emb_pred(f_regions)                     #[P,dim_emb]
        cls_embs = cls_embs.split([len(p) for p in proposals])          #list([p,dim_emb])
        
        zip_package = zip(cls_embs,w_caps,proposals,targets,special_tokens_masks, teacher_alignments, teacher_alignment_scores)
        
        loss_fragment = loss_dummy
        
        for emb_img, w_cap_img, proposal_img, target_img, special_tokens_mask_img, alignment_img, alignment_score_img in zip_package:
            
            idxs_valid_token = torch.where(special_tokens_mask_img!=1)[0]
            
            if len(idxs_valid_token) == 0:      # cannot find any noun phrase
                continue
            
            w_cap_img = w_cap_img[idxs_valid_token]
            
            fragment_score = torch.einsum('pd,wd->pw',emb_img,w_cap_img)
            
            align_fragment_score_student = fragment_score.gather(0, alignment_img.view(1,-1))[0] #torch.index_select(fragment_score, 0, alignment_img)
            # if align_fragment_score_student.shape != alignment_score_img.shape:
            #     print(align_fragment_score_student.shape, alignment_score_img.shape)
            assert align_fragment_score_student.shape == alignment_score_img.shape

            
            align_fragment_att = F.softmax(alignment_score_img,dim=0)
            align_fragment_log_score_student = torch.log(torch.sigmoid(align_fragment_score_student))
            loss_fragment_img = -torch.sum(align_fragment_att*align_fragment_log_score_student)
            loss_fragment += loss_fragment_img/len(proposals)
        
        filter_features = []
        filter_proposals = []
        filter_new_targets = []
        zip_package = zip(features, proposals, new_targets)
        
        for feature_img, proposal_img, new_target_img in zip_package:
            if len(new_target_img) != 0:
                filter_features.append(feature_img)
                filter_proposals.append(proposal_img)
                filter_new_targets.append(new_target_img)
        
        if len(filter_new_targets) != 0:
            self.roi_heads_student.train()
            x, result_student, detector_losses_student = self.roi_heads_student(filter_features, filter_proposals, filter_new_targets, dummy_value = loss_dummy)
        else:
            detector_losses_student = {'loss_classifier':loss_dummy, 'loss_box_reg':loss_dummy, 'loss_mask':loss_dummy}
            
        return {'loss_fragment':loss_fragment*self.lambda_fragment}, {'loss_dummy':loss_dummy}, detector_losses_student
    
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
        
        if self.roi_heads_student.box.predictor.cls_score != self.roi_heads.box.predictor.cls_score:
            self.roi_heads_student.box.predictor.cls_score = self.roi_heads.box.predictor.cls_score
            ### the code switch this head everytime it is evaluated on a new dataset !!!
        
        if self.is_first_run:
            self.roi_heads_student.load_state_dict(copy.deepcopy(self.roi_heads.state_dict()))
            self.rpn_student.load_state_dict(copy.deepcopy(self.rpn.state_dict()))
            self.is_first_run = False
            print('copy teacher parameters')
        
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.fix_rpn:
            self.rpn.eval()
        images = to_image_list(images)
        
        with torch.no_grad():
            features = self.backbone(images.tensors)
        
        # if self.roi_heads:
        #     x, result, detector_losses = self.roi_heads(features, proposals, targets)
        # else:
        #     # RPN-only models don't have roi_heads
        #     x = features
        #     result = proposals
        #     detector_losses = {}

        with torch.no_grad():
            self.rpn.eval()
            proposals, proposal_losses = self.rpn(images, features, targets)
            self.rpn.train()

        '''
            embed between caps and proposals
        '''
        if self.training:
            caps = [t.get_field('nn_caption') for t in targets]
            input_caption = self.bert(caps)
            special_tokens_masks = [e[0] for e in input_caption['special_tokens_mask'].split(1)]
            w_caps = input_caption['input_embeddings']
            
            with torch.no_grad():
                teacher_alignments, teacher_alignment_scores, _ = self.pass_teacher(features, proposals, w_caps, targets, special_tokens_masks)

            loss_fragment, loss_dummy, detector_losses_student = self.pass_student(features, proposals, w_caps, targets, special_tokens_masks, 
                                                                                   teacher_alignments, teacher_alignment_scores, targets)
            
            # with torch.no_grad():
            #     _, _, detector_losses_ref = self.roi_heads_student(features, proposals, targets)
            #     info_ref = {}
            #     for k in detector_losses_ref.keys():
            #         info_ref[k+'_info'] = detector_losses_ref[k]
        else:
            
            # proposals, proposal_losses = self.rpn_student(images, features, targets)
            
            x, result_student, _ = self.roi_heads_student(features, proposals, None)
            
        if self.training:
            losses = {}
            losses.update(loss_fragment)
            losses.update(detector_losses_student)
            # info = {}
            # info.update(info_ref)
            # losses.update(proposal_losses)
            
            # losses['loss_mask'] *= 0.0
            
            return losses#(info,losses)

        return result_student
