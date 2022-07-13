# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN
from .mmss_gcnn import MMSSGridModel
from .st_generalized_rcnn import STGeneralizedRCNN


_DETECTION_META_ARCHITECTURES = {
    "GeneralizedRCNN": GeneralizedRCNN,
    'STGeneralizedRCNN': STGeneralizedRCNN,
    "MMSS-GCNN": MMSSGridModel, # MMSS stands for multimedia self-supervised
}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
