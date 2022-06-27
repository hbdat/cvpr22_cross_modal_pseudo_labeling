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
from maskrcnn_benchmark.data.datasets.helper.lvis_v1_categories import LVIS_CATEGORIES
from maskrcnn_benchmark.data.datasets.helper.parser import normalize_class_names
import copy

from maskrcnn_benchmark.utils.comm import get_rank, get_world_size
import pickle
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker

class UncertaintyGeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(UncertaintyGeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads_student = build_roi_heads(cfg, self.backbone.out_channels, is_teacher = True)
        self.bert = BERT(cfg)

        self.fix_rpn = cfg.MODEL.RPN.DONT_TRAIN
        assert self.fix_rpn == True

        self.exemplar_type = 'SINGLE'
        self.exemplars = {}
        self.output_dir = cfg.OUTPUT_DIR
        self.class_names = None
        self.lambda_exemplar = nn.Parameter(torch.zeros(1).float(),requires_grad=True)
        self.mask_on = cfg.MODEL.MASK_ON

        self.consistency_transform = None

        '''freeze parameters'''
        if self.fix_rpn:
            for p in self.rpn.parameters():
                p.requires_grad = False
        
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.loss_name = ['loss_box_reg', 'loss_classifier', 'loss_mask']
        cap_vocab = ['']*len(LVIS_CATEGORIES)
        for item in LVIS_CATEGORIES:
            id = item['id']-1       # convert to 0 base
            cap_vocab[id] = item['name']
        self.cap_vocab = normalize_class_names(cap_vocab)
        self.is_first_run = True
        self.lambda_pseudo_label = cfg.MODEL.LAMBDA_PSEUDO_LABEL
        self.iter = torch.zeros(1).long()

        weight_dir = '/'.join(cfg.MODEL.WEIGHT.split('/')[:-1])
        self.load_exemplars(output_dir=weight_dir)
        self.masker = Masker(threshold=0.5, padding=1)

    def update_exemplars(self, pseudo_label_img, info=None):
        nns = pseudo_label_img.get_field('joined_words').split('/')
        assert len(nns) == len(pseudo_label_img)
        
        scores = pseudo_label_img.get_field('scores')
        consistencies = pseudo_label_img.get_field('consistencies')
        embs = pseudo_label_img.get_field('embs')

        qualities = scores*consistencies
        embs = F.normalize(embs, dim=-1 )
        for idx_nn, nn in enumerate(nns):
            if self.exemplar_type == 'SINGLE':
                package = {'emb':embs[idx_nn].cpu(),'quality':qualities[idx_nn].cpu(),
                        'score':scores[idx_nn].cpu(),'info':info}
                if nn not in self.exemplars or self.exemplars[nn]['quality'] < package['quality']:
                    self.exemplars[nn] = package
                    #print('exemplar: {} | consistency {} | score {} | coverage {} | lambda_exemplar {}'.format(nn,consistencies[idx_nn].cpu(),scores[idx_nn].cpu(),len(self.exemplars),self.lambda_exemplar.item()))
            elif self.exemplar_type == 'ACCUM':
                package = {'emb':embs[idx_nn].cpu(),'accum_quality':qualities[idx_nn].cpu(),
                        'accum_score':scores[idx_nn].cpu()}
                if nn not in self.exemplars:
                    self.exemplars[nn] = package
                else:
                    self.exemplars[nn] = self.combine_exemplar(self.exemplars[nn],package)
            else:
                raise Exception('Unknown exemplar type')

    def save_exemplars(self):                   #--> framework agnostic
        rank = get_rank()
        with open(self.output_dir+'/exemplars_{}_{}.pkl'.format(rank,self.exemplar_type),'wb') as f:
            pickle.dump(self.exemplars,f)

    def load_exemplars(self, output_dir=None):  #--> framework agnostic
        if output_dir is None:
            output_dir = self.output_dir
            world_size = get_world_size()
        else:
            world_size = 1                 # the content of every partition should be synchronized at the end of training, thus loading any partition is the same
        for rank in range(world_size):
            try:
                with open(output_dir+'/exemplars_{}_{}.pkl'.format(rank,self.exemplar_type),'rb') as f:
                    print('load exemplars_{}'.format(rank))
                    dict = pickle.load(f)
                    for k in dict.keys():
                        if self.exemplar_type == 'SINGLE':
                            if k not in self.exemplars or self.exemplars[k]['quality'] < dict[k]['quality']:
                                self.exemplars[k] = dict[k]
                        elif self.exemplar_type == 'ACCUM':
                            if k not in self.exemplars:
                                self.exemplars[k] = dict[k]
                            else:
                                self.exemplars[k] = self.combine_exemplar(self.exemplars[k],dict[k])
                        else:
                            raise Exception('Unknown exemplar type')
            except:
                print('Cannot load exemplars_{}'.format(rank))

    def combine_embs(self, nns, embs):     #--> framework agnostic
        word_embs = torch.clone(embs).detach()
        device = embs.device
        for idx_nn, nn in enumerate(nns):
            if nn in self.exemplars:
                #word_embs[idx_nn] *= 1-self.lambda_exemplar
                word_embs[idx_nn] += self.lambda_exemplar*self.exemplars[nn]['emb'].to(device)
            else:
                word_embs[idx_nn] += self.lambda_exemplar*0.0       #--> make sure lambda_exemplar is always included in the computational graph to avoid backprop error
        word_embs = F.normalize(word_embs, dim=-1)
        return word_embs
    ### helper func for exemplars ###

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
        
        self.roi_heads.load_state_dict(copy.deepcopy(self.roi_heads_student.state_dict()),strict=False)
        print('copy teacher parameters')

    def extract_emb(self, words):
        encoded_word_list = self.bert(words)
        mask = (1 - encoded_word_list['special_tokens_mask']).to(torch.float32)
        embeddings = (encoded_word_list['input_embeddings'] * mask[:, :, None]).sum(1) / mask.sum(1)[:, None]  # this summation ensures that there is always a vector output

        embeddings = F.normalize(embeddings, dim=-1)   #[n_class,dim_emb]

        return embeddings

    def add_bg_emb(self, embs):     # --> bg class is at 0 idx
        device = embs.device
        dim_emb = embs.shape[1]
        bg_embs = torch.zeros((1,dim_emb),device = device)
        return torch.cat([bg_embs,embs],dim=0)


    def generate_pseudo_label(self, features, proposals, caps, targets):
        #class_embs = self.roi_heads_student['box'].predictor.cls_score
        self.roi_heads_student['box'].predictor.set_class_embeddings(torch.zeros((1,768)))     # << set dummy values
        self.roi_heads_student.eval()   # >> if set at training mode, the model would use gt to subsample regions
        package_x, results, _ = self.roi_heads_student(features, proposals, None, bbox_only = True)
        assert len(results[0]) == len(proposals[0])

        x = package_x['bbox']  
        f_regions = self.avgpool(x)                                                     #[P,dim_v,1]
        f_regions = f_regions.view(f_regions.size(0), -1)                               #[P,dim_v]
        cls_embs = self.roi_heads_student.box.predictor.emb_pred(f_regions)                     #[P,dim_emb]
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
            
            aligned_region_scores, idx_aligned_regions = torch.max(region_scores,dim=0)      #[w]<-[pw]
            
            pseudo_label_img = result_img[idx_aligned_regions]
            
            pseudo_label_img.add_field('joined_words', '/'.join(words))

            consistencies = aligned_region_scores*0.0+1.0
            embs = emb_img[idx_aligned_regions]
            aligned_region_scores = torch.sigmoid(aligned_region_scores)        # << sigmoid normalization
            
            ids_cap = target_img.get_field('ids_cap')
            pseudo_label_img.add_field("labels",ids_cap)
            pseudo_label_img.add_field("scores",aligned_region_scores)
            pseudo_label_img.add_field("consistencies",consistencies)
            pseudo_label_img.add_field("embs",embs)
            pseudo_labels.append(pseudo_label_img)      #this is the similar format with target tensor
        
        ### mask pass ###
        if self.mask_on:
            package_x, results, _ = self.roi_heads_student(features, pseudo_labels, None, bbox_only = False, compute_uncertain = True)
            for results_img, pseudo_label_img in zip(results,pseudo_labels):
                masks = results_img.get_field('mask')                   #[p,1,14,14] , already binary (boolean)
                masks = self.masker([masks], [pseudo_label_img])[0]     #[p,1,w,h]
                masks = masks[:,0]                                      #[p,w,h] 
                masks = SegmentationMask(masks, pseudo_label_img.size, mode='mask')
                pseudo_label_img.add_field('masks',masks)
                pseudo_label_img.add_field('class_uncertainty',results_img.get_field('class_uncertainty'))
                pseudo_label_img.add_field('mask_uncertainty',results_img.get_field('mask_uncertainty'))

        #self.roi_heads_student['box'].predictor.set_class_embeddings(class_embs)
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
        assert self.roi_heads_student

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        
        images = to_image_list(images)
        features = self.backbone(images.tensors)

        if self.training:
            raise Exception('No training mode')
        else:
            self.rpn.eval()
            proposals, _ = self.rpn(images, features, None)
            caps = [t.get_field('nn_caption').split('/') for t in targets]
            pseudo_targets = self.generate_pseudo_label(features, proposals, caps, targets)

            return pseudo_targets
