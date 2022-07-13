# Open-Vocabulary Instance Segmentation via Robust Cross-Modal Pseudo-Labeling

## Overview
This repository contains the implementation of [Open-Vocabulary Instance Segmentation via Robust Cross-Modal Pseudo-Labeling](https://openaccess.thecvf.com/content/CVPR2022/papers/Huynh_Open-Vocabulary_Instance_Segmentation_via_Robust_Cross-Modal_Pseudo-Labeling_CVPR_2022_paper.pdf).
> In this work, we address open-vocabulary instance segmentation, which learn to segment novel objects without any mask annotation during training by generating pseudo masks based on captioned images.

![Image](https://raw.githubusercontent.com/hbdat/cvpr22_cross_modal_pseudo_labeling/main/fig/schematic_figure.png)

---
## Installation
Our code is based upon [OVR](https://github.com/alirezazareian/ovr-cnn), which is built upon [mask-rcnn benchmark] (https://github.com/facebookresearch/maskrcnn-benchmark).
To setup the code, please follow the instruction within [INSTALL.md](https://github.com/hbdat/cvpr22_cross_modal_pseudo_labeling/blob/main/INSTALL.md).

---
## Datasets
To download the datasets, please follow the below instructions.
For more information the data directory is structured, please refer to [maskrcnn_benchmark/config/paths_catalog.py](https://github.com/hbdat/cvpr22_cross_modal_pseudo_labeling/blob/main/maskrcnn_benchmark/config/paths_catalog.py)

### MS-COCO
+ Please download the MS-COCO 2017 dataset into `./datasets/coco/` folder from its official website: [https://cocodataset.org/#download](https://cocodataset.org/#download).
+ Following prior works, the data is partitioned into base and target classes via:

```
python ./preprocess/coco/construct_coco_json.py
```  

### Open Images & Conceptual Captions
+ Please download the correspond datasets from:
	+ Open Images:			[https://github.com/cvdfoundation/open-images-dataset#download-images-with-bounding-boxes-annotations](https://github.com/cvdfoundation/open-images-dataset#download-images-with-bounding-boxes-annotations)
	+ Conceptual Captions:	[https://ai.google.com/research/ConceptualCaptions/download](https://ai.google.com/research/ConceptualCaptions/download)
+ Please extract the corresponding datasets into `./datasets/conceptual/` and `./datasets/openimages/` folder.

#### Annotations for Open Images
+ Convert annotation format of Open Images to COCO format (json):
```
cd ./preprocess/openimages/openimages2coco
python convert_annotations.py -p ../../../datasets/openimages/ --version challenge_2019 --task mask --subsets train
python convert_annotations.py -p ../../../datasets/openimages/ --version challenge_2019 --task mask --subsets val
```
+ Partition all classes into base and target classes:
```
python ./preprocess/openimages/construct_openimages_json.py
```

#### Annotations for Conceptual Captions
```
Coming Soon!
```

---
## Experiments
To reproduce the main experiments in the paper, we provide the script to train the teacher and the student models on both MS-COCO and Open Images & Conceptual Captions below.
Please notice that the teacher must be trained first in order to produce pseudo labels/masks to train the student models.

### MS-COCO

+ Caption pretraining:
  + Please download the pretrained backbone model from [here](https://drive.google.com/file/d/1mFnAZVnn2NT2Ys841EPOMaQ6jnvFXPWJ/view?usp=sharing) into the folder `./model_weights`. This model is from the [OVR](https://github.com/alirezazareian/ovr-cnn) code base.

+ Teacher training:
```
python -m torch.distributed.launch --nproc_per_node=8 tools/train_net.py --config-file configs/coco_cap_det/zeroshot_mask.yaml OUTPUT_DIR ./checkpoint/mscoco_teacher/ MODEL.WEIGHT ./model_weights/model_pretrained.pth
```

+ Student training:
```
python -m torch.distributed.launch --nproc_per_node=8 tools/train_net.py --config-file configs/coco_cap_det/student_teacher_mask_rcnn_uncertainty.yaml OUTPUT_DIR ./checkpoint/mscoco_student/ MODEL.WEIGHT ./checkpoint/mscoco_student/model_final.pth
```

+ Evaluation
  + To quickly evaluate the performance, we provide pretrained student/teacher model in [Pretrained Models](https://github.com/hbdat/cvpr22_cross_modal_pseudo_labeling/edit/main/README.md#pretrained-models). Please download them into `./pretrained_model/` folder and run the following script:
```
python -m torch.distributed.launch --nproc_per_node=8 tools/test_net.py --config-file configs/coco_cap_det/student_teacher_mask_rcnn_uncertainty.yaml OUTPUT_DIR ./results/mscoco_student MODEL.WEIGHT ./pretrained_model/coco_student/model_final.pth
```

### Open Images & Conceptual Captions

+ Caption pretraining:
```
Coming Soon!
```

+ Teacher training:
```
Coming Soon!
```

+ Student training:
```
Coming Soon!
```

+ Evaluation
```
Coming Soon!
```

## Pretrained Models

| Dataset                       | Teacher | Student |
|-------------------------------|---------|---------|
| MS-COCO                       |  [model](https://drive.google.com/file/d/1KGnURlIlZfkW1N2_TMHrY81YN5WMzO_J/view?usp=sharing)  |  [model](https://drive.google.com/file/d/12BGwgV1wPyO_2xeAhLGxN2elBqc8v247/view?usp=sharing)  |
| Conceptual Caps + Open Images |  [model]()  |  [model]()  |

---
## Citation
If this code is helpful for your research, we would appreciate if you cite the work:
```
@article{Huynh:CVPR22,
  author = {D.~Huynh and J.~Kuen and Z.~Lin and J.~Gu and E.~Elhamifar},
  title = {Open-Vocabulary Instance Segmentation via Robust Cross-Modal Pseudo-Labeling},
  journal = {{IEEE} Conference on Computer Vision and Pattern Recognition},
  year = {2022}}
```
