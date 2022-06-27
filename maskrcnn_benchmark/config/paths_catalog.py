# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os
from copy import deepcopy

class DatasetCatalog(object):
    DATA_DIR = "datasets"
    DATASETS = {
        "conceptual_captions_train": {
            "img_dir": "concap/images",
            "ann_file": "concap/train.json"
        },
        "conceptual_captions_val": {
            "img_dir": "concap/images",
            "ann_file": "concap/validation.json"
        },

        "coco_captions_train": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/annotations/captions_train2017.json"
        },
        "coco_captions_val": {
            "img_dir": "coco/val2017",
            "ann_file": "coco/annotations/captions_val2017.json"
        },

        "coco_cap_det_train": {
            "img_dir": "coco/train2017",
            "ann_file_cap": "coco/annotations/captions_train2017.json",
            "ann_file": "coco/zero-shot/instances_train2017_seen_2.json"
        },

        "coco_zeroshot_train": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/zero-shot/instances_train2017_seen_2.json"
        },
        "coco_zeroshot_val": {
            "img_dir": "coco/val2017",
            "ann_file": "coco/zero-shot/instances_val2017_unseen_2.json"
        },
        "coco_generalized_zeroshot_val": {
            "img_dir": "coco/val2017",
            "ann_file": "coco/zero-shot/instances_val2017_all_2.json"
        },
        "coco_not_zeroshot_val": {
            "img_dir": "coco/val2017",
            "ann_file": "coco/zero-shot/instances_val2017_seen_2.json"
        },
        "coco_zeroshot_plus_unseen_train": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/zero-shot/instances_train2017_all_2.json"
        },

        "coco_2017_train": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/zero-shot/instances_train2017_full.json"
        },
        "coco_2017_val": {
            "img_dir": "coco/val2017",
            "ann_file": "coco/zero-shot/instances_val2017_full.json"
        },
        "coco_2014_train": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/instances_train2014.json"
        },
        "coco_2014_val": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_val2014.json"
        },
        "coco_2014_minival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_minival2014.json"
        },
        "coco_2014_valminusminival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_valminusminival2014.json"
        },
        "keypoints_coco_2014_train": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/person_keypoints_train2014.json",
        },
        "keypoints_coco_2014_val": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/person_keypoints_val2014.json"
        },
        "keypoints_coco_2014_minival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/person_keypoints_minival2014.json",
        },
        "keypoints_coco_2014_valminusminival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/person_keypoints_valminusminival2014.json",
        },
        "voc_2007_train": {
            "data_dir": "voc/VOC2007",
            "split": "train"
        },
        "voc_2007_train_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_train2007.json"
        },
        "voc_2007_val": {
            "data_dir": "voc/VOC2007",
            "split": "val"
        },
        "voc_2007_val_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_val2007.json"
        },
        "voc_2007_test": {
            "data_dir": "voc/VOC2007",
            "split": "test"
        },
        "voc_2007_test_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_test2007.json"
        },
        "voc_2012_train": {
            "data_dir": "voc/VOC2012",
            "split": "train"
        },
        "voc_2012_train_cocostyle": {
            "img_dir": "voc/VOC2012/JPEGImages",
            "ann_file": "voc/VOC2012/Annotations/pascal_train2012.json"
        },
        "voc_2012_val": {
            "data_dir": "voc/VOC2012",
            "split": "val"
        },
        "voc_2012_val_cocostyle": {
            "img_dir": "voc/VOC2012/JPEGImages",
            "ann_file": "voc/VOC2012/Annotations/pascal_val2012.json"
        },
        "voc_2012_test": {
            "data_dir": "voc/VOC2012",
            "split": "test"
            # PASCAL VOC2012 doesn't made the test annotations available, so there's no json annotation
        },

        ##############################################
        # These ones are deprecated, should be removed
        "cityscapes_fine_instanceonly_seg_train_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_train.json"
        },
        "cityscapes_fine_instanceonly_seg_val_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_val.json"
        },
        "cityscapes_fine_instanceonly_seg_test_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_test.json"
        },
        ##############################################

        "cityscapes_poly_instance_train": {
            "img_dir": "cityscapes/leftImg8bit/",
            "ann_dir": "cityscapes/gtFine/",
            "split": "train",
            "mode": "poly",
        },
        "cityscapes_poly_instance_val": {
            "img_dir": "cityscapes/leftImg8bit",
            "ann_dir": "cityscapes/gtFine",
            "split": "val",
            "mode": "poly",
        },
        "cityscapes_poly_instance_minival": {
            "img_dir": "cityscapes/leftImg8bit",
            "ann_dir": "cityscapes/gtFine",
            "split": "val",
            "mode": "poly",
            "mini": 10,
        },
        "cityscapes_mask_instance_train": {
            "img_dir": "cityscapes/leftImg8bit/",
            "ann_dir": "cityscapes/gtFine/",
            "split": "train",
            "mode": "mask",
        },
        "cityscapes_mask_instance_val": {
            "img_dir": "cityscapes/leftImg8bit",
            "ann_dir": "cityscapes/gtFine",
            "split": "val",
            "mode": "mask",
        },
        "cityscapes_mask_instance_minival": {
            "img_dir": "cityscapes/leftImg8bit",
            "ann_dir": "cityscapes/gtFine",
            "split": "val",
            "mode": "mask",
            "mini": 10,
        },

        ### combined openimages + conceptual cap ###
        "conceptual_openimages_mask": {
            "openimage_img_dir": "/mnt/session_space/datasets/openimages/train",
            "openimages_ann_file": "/openimages/zero-shot/instances_train2019_mask_seen_100.json",
            "openimages_imagelevel_file": None,

            "conceptual_img_dir": "/trainman-mount/trainman-storage-add87537-4c73-4d36-9774-0fddd6d5fd4f/datasets/conceptual/images/",
            "conceptual_anno_dir": "/trainman-mount/trainman-storage-add87537-4c73-4d36-9774-0fddd6d5fd4f/datasets/conceptual/img_lists/",
            "conceptual_split": "train"
        },

        "dummy_conceptual_openimages_mask": {
            "openimage_img_dir": "/mnt/session_space/datasets/openimages/validation",
            "openimages_ann_file": "/openimages/zero-shot/instances_val2019_seg_unseen_100.json",
            "openimages_imagelevel_file": None,

            "conceptual_img_dir": "/trainman-mount/trainman-storage-add87537-4c73-4d36-9774-0fddd6d5fd4f/datasets/conceptual/images/",
            "conceptual_anno_dir": "/trainman-mount/trainman-storage-add87537-4c73-4d36-9774-0fddd6d5fd4f/datasets/conceptual/img_lists/",
            "conceptual_split": "train"
        },

        ### zs openimages mask ###
        'openimages_zeroshot_train_100_mask':{"ann_file": 'openimages/zero-shot/instances_train2019_mask_seen_100.json',
                                    "img_dir": '/mnt/session_space/datasets/openimages/train',
                                    "imagelevel_file": None},

        'openimages_zeroshot_val_100_mask':{"ann_file": 'openimages/zero-shot/instances_val2019_seg_unseen_100.json',
                                        "img_dir": '/mnt/session_space/datasets/openimages/validation',
                                        "imagelevel_file":'./datasets/openimages/annotations/challenge-2019-validation-segmentation-labels_expand.csv'},
    
        'openimages_not_zeroshot_val_100_mask':{"ann_file": 'openimages/zero-shot/instances_val2019_seg_seen_100.json',
                                            "img_dir": '/mnt/session_space/datasets/openimages/validation',
                                            "imagelevel_file":'./datasets/openimages/annotations/challenge-2019-validation-segmentation-labels_expand.csv'},
                                            
        'openimages_generalized_zeroshot_val_100_mask':{"ann_file": 'openimages/zero-shot/instances_val2019_seg_all_100.json',
                                                    "img_dir": '/mnt/session_space/datasets/openimages/validation',
                                                    "imagelevel_file":'./datasets/openimages/annotations/challenge-2019-validation-segmentation-labels_expand.csv'},

        ### conceptual cap ###
        "conceptual_cap_train": {
            "img_dir": "/trainman-mount/trainman-storage-add87537-4c73-4d36-9774-0fddd6d5fd4f/datasets/conceptual/images/",
            "anno_dir": "/trainman-mount/trainman-storage-add87537-4c73-4d36-9774-0fddd6d5fd4f/datasets/conceptual/img_lists/",
            "split": "train",
            "size":100,
        },
    }

    @staticmethod
    def get(name):
        if "conceptual_openimages_mask" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                openimage_img_dir = attrs["openimage_img_dir"],
                openimages_ann_file = data_dir + attrs["openimages_ann_file"],
                openimages_imagelevel_file = attrs["openimages_imagelevel_file"],

                conceptual_split=attrs["conceptual_split"],
                conceptual_img_dir=attrs["conceptual_img_dir"],
                conceptual_anno_dir=attrs["conceptual_anno_dir"],
            )
            return dict(
                factory="ConceptualOpenImagesDetDataset",
                args=args,
            )
        elif 'openimages' in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root= attrs["img_dir"],
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
                imagelevel_file=attrs["imagelevel_file"],
            )
            return dict(
                factory="OpenImagesDataset",
                args=args,
            )
        elif 'conceptual_cap' in name:
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                split=attrs["split"],
                img_dir=attrs["img_dir"],
                anno_dir=attrs["anno_dir"],
                size=attrs["size"]
            )
            return dict(
                factory="ConCapDetDataset",
                args=args
            )
        elif "coco_cap_det" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
                ann_file_cap=os.path.join(data_dir, attrs["ann_file_cap"]),
                remove_images_without_annotations=False,
            )
            return dict(
                factory="COCOCapDetDataset",
                args=args,
            )
        elif "coco" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="COCOCaptionsDataset" if "captions" in name else "COCODataset",
                args=args,
            )
        elif "voc" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(
                factory="PascalVOCDataset",
                args=args,
            )
        elif "cityscapes" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = deepcopy(DatasetCatalog.DATASETS[name])
            attrs["img_dir"] = os.path.join(data_dir, attrs["img_dir"])
            attrs["ann_dir"] = os.path.join(data_dir, attrs["ann_dir"])
            return dict(factory="CityScapesDataset", args=attrs)
        elif "conceptual" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="ConCapDataset",
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog(object):
    S3_C2_DETECTRON_URL = "https://dl.fbaipublicfiles.com/detectron"
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-50-GN": "ImageNetPretrained/47261647/R-50-GN.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "MSRA/R-101-GN": "ImageNetPretrained/47592356/R-101-GN.pkl",
        "FAIR/20171220/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
    }

    C2_DETECTRON_SUFFIX = "output/train/{}coco_2014_train%3A{}coco_2014_valminusminival/generalized_rcnn/model_final.pkl"
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "01_33_49.iAX0mXvW",
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "01_36_30.cUF7QR7I",
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "01_38_50.sNxI7sX7",
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "06_31_39.5MIHi1fZ",
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "01_45_57.ZgkA7hPB",
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "01_48_14.DzEQe4wC",
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "02_31_37.KqyEK4tT",
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "06_35_59.RZotkLKI",
        "37129812/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x": "09_35_36.8pzTQKYK",
        # keypoints
        "37697547/e2e_keypoint_rcnn_R-50-FPN_1x": "08_42_54.kdzV35ao"
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog.get_c2_detectron_12_2017_baselines(name)
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog {}".format(name))

    @staticmethod
    def get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        name = name[len("ImageNetPretrained/"):]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        url = "/".join([prefix, name])
        return url

    @staticmethod
    def get_c2_detectron_12_2017_baselines(name):
        # Detectron C2 models are stored following the structure
        # prefix/<model_id>/2012_2017_baselines/<model_name>.yaml.<signature>/suffix
        # we use as identifiers in the catalog Caffe2Detectron/COCO/<model_id>/<model_name>
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        dataset_tag = "keypoints_" if "keypoint" in name else ""
        suffix = ModelCatalog.C2_DETECTRON_SUFFIX.format(dataset_tag, dataset_tag)
        # remove identification prefix
        name = name[len("Caffe2Detectron/COCO/"):]
        # split in <model_id> and <model_name>
        model_id, model_name = name.split("/")
        # parsing to make it match the url address from the Caffe2 models
        model_name = "{}.yaml".format(model_name)
        signature = ModelCatalog.C2_DETECTRON_MODELS[name]
        unique_name = ".".join([model_name, signature])
        url = "/".join([prefix, model_id, "12_2017_baselines", unique_name, suffix])
        return url
