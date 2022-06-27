import os
import json
import numpy as np
import torch
import torchvision
from PIL import Image
import pickle
import spacy

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from .helper.parser import LVISParser

class ConCapDetDataset:
    def __init__(
        self, split, img_dir, anno_dir, transforms=None, extra_args=None, size = -1
    ):
        self._transforms = transforms
        self.split = split
        self.img_dir = img_dir
        self._image_root = img_dir
        
        index_file = os.path.join(anno_dir,'{}_index.json'.format(split))
        cap_file = os.path.join(anno_dir,'{}_caption.json'.format(split))
        
        with open(index_file, 'r') as fin:
            self.index_meta = json.load(fin)

        with open(cap_file, 'r') as fin:
            self.cap_meta = json.load(fin)

        with open('./datasets/conceptual/all_meta_{}.pkl'.format(split),'rb') as f:
            self.all_meta_dic = pickle.load(f)

        if size != -1:
            self.index_meta = self.index_meta[:size]

        self.parser = LVISParser()

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
        fname = self.index_meta[idx]
        anno_cap = self.cap_meta[idx]
        img = Image.open(os.path.join(self._image_root, fname)).convert('RGB')

        boxes = [1,2,3,4]       #dummy value to avoid error
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        target.add_field("caption",anno_cap)
        nns_cap, ids_cap = self.extract_obj([anno_cap])
        ids_cap = torch.tensor(ids_cap)
        target.add_field("nn_caption",'/'.join(nns_cap))
        target.add_field("ids_cap",ids_cap)
        target.add_field("dataset_name",'Conceptual')
        target.add_field("is_det","No")

        if self._transforms is not None:
            img, _ = self._transforms(img, target)
        return img, target, idx

    def get_img_info(self, index):
        ret_dic = {'file_name': self.index_meta[index], 'caption': self.cap_meta[index]}
        #img = Image.open(os.path.join(self._image_root, ret_dic['fname'])).convert('RGB')
        #width, height = 100, 100#img.size
        ret_dic['width'], ret_dic['height'] = self.all_meta_dic[index]['width'], self.all_meta_dic[index]['height']
        return ret_dic

    def __len__(self):
        return len(self.index_meta)