# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 16:37:08 2021

@author: badat
"""

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import json
import numpy as np
import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints
from maskrcnn_benchmark.data.datasets.helper.parser import normalize_class_names
from .helper.parser import LVISParser

from pycocotools.coco import COCO

min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False

def has_unseen_labels(anno,unseen_classes):
    classes = [obj["category_id"] for obj in anno]
    unseen_labels = [c for c in classes if c in unseen_classes]
    return len(unseen_labels) > 0

class COCOCapDetDataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, ann_file_cap, root, remove_images_without_annotations, 
        transforms=None, extra_args=None,
    ):
        super(COCOCapDetDataset, self).__init__(root, ann_file)
        
        ## Dat modification ##
        self.coco_cap = COCO(ann_file_cap)
        self.coco = COCO(ann_file)
        self._image_root = root
        
        cat_ids = self.coco.getCatIds()
        cats = self.coco.loadCats(cat_ids)
        self.cat2id = {cat['name']:cat['id'] for cat in cats}
        self.id2cat = {cat['id']:cat['name'] for cat in cats}
        ## Dat modification ##
        
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids
        
        self.categories = {cat['id']: cat['name'] for cat in self.coco.cats.values()}

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

        self.class_splits = {}
        if getattr(extra_args, 'LOAD_EMBEDDINGS', False):
            self.class_embeddings = {}
            with open(ann_file, 'r') as fin:
                ann_data = json.load(fin)
                for item in ann_data['categories']:
                    emb = item['embedding'][extra_args['EMB_KEY']]
                    self.class_embeddings[item['id']] = np.asarray(emb, dtype=np.float32)
                    if 'split' in item:
                        if item['split'] not in self.class_splits:
                            self.class_splits[item['split']] = []
                        self.class_splits[item['split']].append(item['id'])
            self.class_emb_mtx = np.zeros(
                (len(self.contiguous_category_id_to_json_id) + 1, extra_args['EMB_DIM']),
                dtype=np.float32)
            for i, cid in self.contiguous_category_id_to_json_id.items():
                self.class_emb_mtx[i, :] = self.class_embeddings[cid]

            self.class_emb_mtx = torch.tensor(self.class_emb_mtx)
        
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        
        self.parser = LVISParser()

        ### get cat names ###
        class_names = ['']*(len(self.categories)+1) #--> add background class
        for json_id, name in self.categories.items():
                cont_id = self.json_category_id_to_contiguous_id[json_id]
                class_names[cont_id] = name
        class_names[0] = 'bg'

        self.class_names = normalize_class_names(class_names)
        ### get cat names ###
            
    def extract_obj(self, sentence_list):
        dict_ret = {}
        for sent in sentence_list:
            nns, category_ids = self.parser.parse(sent)
            for n,i in zip(nns,category_ids):
                dict_ret[n] = i
        
        unique_nns = list(dict_ret.keys())
        unique_ids = [dict_ret[e] for e in unique_nns]
        return unique_nns, unique_ids

    def __getitem__(self, idx):
        img, anno = super(COCOCapDetDataset, self).__getitem__(idx)
        
        img_id = anno[0]['image_id']
        
        annId_cap = self.coco_cap.getAnnIds(imgIds=img_id)
        anno_cap = self.coco_cap.loadAnns(annId_cap)
        anno_cap = [cap['caption'] for cap in anno_cap]
        
        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]
        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")
        
        classes = [obj["category_id"] for obj in anno]
        # cat_tokens = [self.id2cat[ID] for ID in classes]
        
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)
        
        target.add_field("caption",'/'.join(anno_cap))
        nns_cap, ids_cap = self.extract_obj(anno_cap)
        ids_cap = torch.tensor(ids_cap)
        target.add_field("nn_caption",'/'.join(nns_cap))
        target.add_field("ids_cap",ids_cap)
        target.add_field("dataset_name",'MSCOCO')
        target.add_field("is_det","Yes")
        
        if anno and "segmentation" in anno[0]:
            masks = [obj["segmentation"] for obj in anno]
            masks = SegmentationMask(masks, img.size, mode='poly')
            target.add_field("masks", masks)

        target = target.clip_to_image(remove_empty=False)       #<< change here is accomodate string fields

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
