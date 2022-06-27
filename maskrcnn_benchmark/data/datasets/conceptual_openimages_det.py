import os
import json
import numpy as np
import torch
import torchvision
from PIL import Image
import pickle

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from .conceptual_cap_det import ConCapDetDataset
from .openimages import OpenImagesDataset
import copy

class ConceptualOpenImagesDetDataset:
    def __init__(
        self, 
        openimages_ann_file, openimage_img_dir, openimages_imagelevel_file,
        conceptual_split, conceptual_img_dir, conceptual_anno_dir,
        transforms=None, extra_args=None,
    ):
        print('loading openimages ...')
        self.openimages = OpenImagesDataset(openimages_ann_file, openimage_img_dir, openimages_imagelevel_file, remove_images_without_annotations=True, 
        transforms=transforms, extra_args=extra_args,is_repeat_sampling=True)
        print('done')

        print('loading conceptual caption ....')
        self.conceptualcap = ConCapDetDataset(conceptual_split, conceptual_img_dir, conceptual_anno_dir, transforms=transforms, extra_args=extra_args)
        print('done')

        self._transforms = transforms

        self.categories = self.openimages.categories

        self.json_category_id_to_contiguous_id = self.openimages.json_category_id_to_contiguous_id
        self.contiguous_category_id_to_json_id = self.openimages.contiguous_category_id_to_json_id
        #self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.class_splits = self.openimages.class_splits
        self.class_emb_mtx = self.openimages.class_emb_mtx
        self.class_embeddings = self.openimages.class_embeddings
        self.class_names = self.openimages.class_names

        ## create global ids
        openimages_global_ids = ['openimages_{}'.format(idx) for idx in range(len(self.openimages))]
        conceptual_global_ids = ['conceptual_{}'.format(idx) for idx in range(len(self.conceptualcap))]

        balance_factor = int(max(len(conceptual_global_ids)//len(openimages_global_ids),1))
        openimages_global_ids = openimages_global_ids*balance_factor

        self.ids = openimages_global_ids + conceptual_global_ids
        self.ids = np.random.permutation(self.ids)

        print('#samples {}: #openimages {} --> {} #conceptual_cap {}'.format(len(self.ids),len(self.openimages),len(openimages_global_ids),len(conceptual_global_ids)))

        self.switch = None

    def __getitem__(self, idx):
        if self.switch is not None:
            if self.switch == 0:
                idx_coco = idx%len(self.conceptualcap)
                img, target, _ = self.conceptualcap.__getitem__(idx_coco)
            else:
                idx_openimages = idx%len(self.openimages)
                img, target, _ = self.openimages.__getitem__(idx_openimages)
        else:
            global_id = self.ids[idx]
            if 'openimages_' in global_id:
                #print('Det')
                idx_openimages = int(global_id.split('_')[-1])
                img, target, _ = self.openimages.__getitem__(idx_openimages)
            elif 'conceptual_' in global_id:
                #print('Cap')
                idx_conceptual = int(global_id.split('_')[-1])
                img, target, _ = self.conceptualcap.__getitem__(idx_conceptual)

        return img, target, idx

    def get_img_info(self, index):
        if self.switch is not None:
            if self.switch == 0:
                idx_coco = index%len(self.conceptualcap)
                return self.cococap.get_img_info(idx_coco)
            else:
                idx_openimages = index%len(self.openimages)
                return self.conceptualcap.get_img_info(idx_openimages)
        else:
            global_id = self.ids[index]
            if 'openimages_' in global_id:
                idx_openimages = int(global_id.split('_')[-1])
                return self.openimages.get_img_info(idx_openimages)
            elif 'conceptual_' in global_id:
                idx_conceptual = int(global_id.split('_')[-1])
                return self.conceptualcap.get_img_info(idx_conceptual)

    def __len__(self):
        return len(self.ids)