# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import json
import numpy as np
import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints
import pandas as pd
from typing import Any, Callable, Optional, Tuple, List
import os
import pickle
import pycococreatortools.pycococreatortools as pycococreatortools
import copy
from tqdm import tqdm
from maskrcnn_benchmark.data.datasets.helper.parser import normalize_class_names


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


### only applicable for torchvision 0.8.2+cu110 ###
from PIL import Image
import os
import os.path

assert torchvision.__version__ == '0.8.2+cu110'
def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = [self.ids[index]]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

torchvision.datasets.coco.CocoDetection.__getitem__ = __getitem__
### only applicable for pytorch 1.4.0 ###


class OpenImagesDataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, imagelevel_file, seg_ann_file=None, remove_images_without_annotations=True, 
        transforms=None, extra_args=None,is_repeat_sampling=True
    ):
        super(OpenImagesDataset, self).__init__(root, ann_file)
        self._image_root = root
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
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

        self.freebase_id_2_cont_id = {cat['freebase_id']:  self.json_category_id_to_contiguous_id[cat['id']] for cat in self.coco.cats.values()}

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
        
        if imagelevel_file is not None:
            self.imagelevel = True
            self.prepare_imagelevel_info(root, imagelevel_file)
        else:
            self.imagelevel = False

        # LabelName to int id
        self.map_class_id_to_class_name = {0:'__background__'}
        for cont_id in self.contiguous_category_id_to_json_id.keys():
            self.map_class_id_to_class_name[cont_id] = self.categories[self.contiguous_category_id_to_json_id[cont_id]]

        ### load segmentation info ###
        self.seg_ann_file = seg_ann_file

        if self.seg_ann_file is not None:
            with open(self.seg_ann_file, 'rb') as f:
                self.dict_seg = pickle.load(f)
        ### load segmentation info ###

        if 'train' in root and is_repeat_sampling:
            ### perform repeat sampling ###
            t = 0.1#t = 0.01#
            rf_id_cache = './datasets/openimages/'+ ann_file.split('/')[-1].split('.')[0]+'_t_{}.pkl'.format(t)

            if os.path.isfile(rf_id_cache):
                with open(rf_id_cache, 'rb') as f:
                    new_ids = pickle.load(f)
            else:
                print('not found {}'.format(rf_id_cache))
                print('compute freq')
                frequency = self.compute_freq()
                print('compute repeat factor')
                repeat_factor_img = self.compute_repeat_factor_img(frequency, t)
                print('generate new ids')
                new_ids = self.compute_new_ids(repeat_factor_img)

                with open(rf_id_cache, 'wb') as f:
                    pickle.dump(new_ids, f)

            print('applying repeat factor sampling {}: {} --> {}'.format(t, len(self.ids), len(new_ids)))
            
            self.ids = new_ids
            print('random_permute')
            self.ids = np.random.permutation(self.ids)
            self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}     ## recomputing id_to_img_map after permutation

        print("datasize {}".format(len(self.ids)))

        ### get cat names ###
        class_names = ['']*(len(self.categories)+1) #--> add background class
        for json_id, name in self.categories.items():
                cont_id = self.json_category_id_to_contiguous_id[json_id]
                class_names[cont_id] = name
        class_names[0] = 'bg'

        self.class_names = normalize_class_names(class_names)
        ### get cat names ###



    ### repeat factor sampling ###
    def compute_freq(self):
        dic_stats = {}
        for idx, img_id in tqdm(enumerate(self.ids)):
            target = self.get_groundtruth(idx)
            for class_id in target.get_field('labels').numpy().tolist():
                if class_id not in dic_stats:
                    dic_stats[class_id] = []
                dic_stats[class_id].append(idx)

        frequency = {}
        for class_id in dic_stats:
            frequency[class_id] = len(dic_stats[class_id])*1.0/len(self.ids)

        return frequency

    def compute_repeat_factor(self, freqs, t):
        factor = [max(1,(t/f)**0.5) for f in freqs]
        return int(max(factor))

    def compute_repeat_factor_img(self, frequency, t = 0.02):
        repeat_factor_img = {}

        for idx, img_id in tqdm(enumerate(self.ids)):
            target = self.get_groundtruth(idx)
            freqs = []

            for class_id in target.get_field('labels').numpy().tolist():
                freqs.append(frequency[class_id])

            repeat_factor_img[img_id] = self.compute_repeat_factor(freqs, t)
        return repeat_factor_img

    def compute_new_ids(self, repeat_factor_img):
        new_ids = []
        for img_id in repeat_factor_img.keys():
            new_ids.extend([img_id]*repeat_factor_img[img_id])
        
        return new_ids
    ### repeat factor sampling ###

    def prepare_imagelevel_info(self,root, imagelevel_file):
        imagelevel_ann = pd.read_csv(imagelevel_file)
        ### caution unknown implementation ###
        imagelevel_ann = imagelevel_ann[imagelevel_ann.Confidence==0]
        ### caution unknown implementation ###
        self.imagelevel_groups = imagelevel_ann.groupby("ImageID")

    def get_groundtruth(self, idx, imagelevel=False, isgroup=False):
        image_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=[image_id], iscrowd=None)
        anno = self.coco.loadAnns(ann_ids)
        
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        info = self.get_img_info(idx)
        target = BoxList(boxes, (info['width'],info['height']), mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        target.add_field("caption","")
        target.add_field("nn_caption","")
        target.add_field("is_det","Yes")

        ### load mask on-the-fly ###
        if "iseg_file_name" in anno[0]:
            masks = []
            for j in range(len(anno)):
                annotation_info = None
                try:
                    iseg_file_name = anno[j]["iseg_file_name"]
                    # if "root_dir" in dataset_dict:
                    #     iseg_file_name = os.path.join(dataset_dict["root_dir"], iseg_file_name)

                    if os.path.isfile(iseg_file_name):
                        binary_mask = np.asarray(Image.open(iseg_file_name).convert('1')).astype(np.uint8)
                        category_info = {'id': anno[j]['category_id'], 'is_crowd': False}
                        image_id = image_id
                        img_size = (info["width"], info["height"])
                        annotation_info = pycococreatortools.create_annotation_info(
                                        -1, image_id, category_info, binary_mask,
                                        img_size, tolerance=2)   # return list of list
                    else:
                        print('not found {}'.format(iseg_file_name))
                    
                except Exception as e:
                    print(e)

                if annotation_info is None:
                    print('none annotation {}'.format(iseg_file_name))
                    annotation_info = {'segmentation': [[0.0]*10]}          # must create dummy values because the structures of dicts in the same batch should be the same
                masks.append(annotation_info['segmentation'])
            
            masks = SegmentationMask(masks, target.size, mode='poly')
            target.add_field("masks", masks)
        ### load mask on-the-fly ###

        if anno and "segmentation" in anno[0]:
            masks = [obj["segmentation"] for obj in anno]
            masks = SegmentationMask(masks, target.size, mode='poly')
            target.add_field("masks", masks)

        # if anno and "keypoints" in anno[0]:
        #     keypoints = [obj["keypoints"] for obj in anno]
        #     keypoints = PersonKeypoints(keypoints, img.size)
        #     target.add_field("keypoints", keypoints)

        if isgroup:
            ### ignore isgroup
            isgroup_column = torch.Tensor([0]*len(boxes)).byte()
            target.add_field("isgroup", isgroup_column)

        target = target.clip_to_image(remove_empty=False)  # << set to false to account for str field

        boxlist = target

        if self.imagelevel and imagelevel:
            try:
                imagelevel_classID = []
                imagelevel_lns = np.unique(self.imagelevel_groups.get_group(image_id).LabelName.values)
                for l in imagelevel_lns:
                    if l in self.freebase_id_2_cont_id:   
                        imagelevel_classID.append(self.freebase_id_2_cont_id[l])
                return boxlist, np.unique(imagelevel_classID)
            except:
                return boxlist, np.array([])
        else:
            return boxlist

    def __getitem__(self, idx):
        img, anno = super(OpenImagesDataset, self).__getitem__(idx)

        target = self.get_groundtruth(idx)

        assert img.size[0] == target.size[0]
        assert img.size[1] == target.size[1]

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
