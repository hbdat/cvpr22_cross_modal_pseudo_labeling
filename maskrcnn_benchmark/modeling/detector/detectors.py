# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN
from .cap_generalized_rcnn import CapGeneralizedRCNN
#from .mmss_gcnn import MMSSGridModel
from .st_generalized_rcnn import STGeneralizedRCNN
from .uncertainty_generalized_rcnn import UncertaintyGeneralizedRCNN

from .baselines.unbiased_teacher.unbiased_teacher import UnbiasedTeacherBaseline
from .baselines.soft_teacher.soft_teacher import SoftTeacherBaseline
from .baselines.SB.SB import SBBaseline
from .baselines.BA_RPN.BA_RPN import BA_RPNBaseline
from .baselines.OMP.OMP import OMPBaseline
from .st_generalized_rcnn_teacher_uncertainty import TeacherUncertaintySTGeneralizedRCNN    #this is for testing different uncertainty variant
from .pseudo_mask_generalized_rcnn import PseudoMaskGeneralizedRCNN
from .st_generalized_rcnn_teacher_cls_score_reweight import ClsScoreReweightSTGeneralizedRCNN


_DETECTION_META_ARCHITECTURES = {
    "GeneralizedRCNN": GeneralizedRCNN,
    "CapGeneralizedRCNN": CapGeneralizedRCNN,
    'STGeneralizedRCNN': STGeneralizedRCNN,
    'UncertaintyGeneralizedRCNN':UncertaintyGeneralizedRCNN,

    'UnbiasedTeacherBaseline':UnbiasedTeacherBaseline,
    'SoftTeacherBaseline':SoftTeacherBaseline,
    'SBBaseline':SBBaseline,
    'BA_RPNBaseline':BA_RPNBaseline,
    'OMPBaseline':OMPBaseline,
    'TeacherUncertaintySTGeneralizedRCNN': TeacherUncertaintySTGeneralizedRCNN,
    'PseudoMaskGeneralizedRCNN': PseudoMaskGeneralizedRCNN,
    'ClsScoreReweightSTGeneralizedRCNN':ClsScoreReweightSTGeneralizedRCNN
}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
