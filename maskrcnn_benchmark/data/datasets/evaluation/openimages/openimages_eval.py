# A modification version from chainercv repository.
# (See https://github.com/chainer/chainercv/blob/master/chainercv/evaluations/eval_detection_voc.py)
from __future__ import division

import os
from collections import defaultdict
import numpy as np
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
import pickle
import pdb
import torch

def do_openimages_evaluation(dataset, predictions, output_folder, logger, box_only):
    # TODO need to make the use_07_metric format available
    # for the user to choose
    pred_boxlists = []
    gt_boxlists = []
    list_image_names = []
    for image_id, prediction in enumerate(predictions):
        img_info = dataset.get_img_info(image_id)
        if len(prediction) == 0:
            continue
        image_width = img_info["width"]
        image_height = img_info["height"]
        
        prediction = prediction.resize((image_width, image_height))
        pred_boxlists.append(prediction)

        image_name = dataset.id_to_img_map[image_id]
        list_image_names.append(image_name)

        if box_only:
            dummy_labels = prediction.get_field('objectness').float()*0+1
            prediction.add_field('labels',dummy_labels.int())
            prediction.add_field('scores',prediction.get_field('objectness'))

        if dataset.imagelevel:
            gt_boxlist, imagelevel_classes = dataset.get_groundtruth(image_id, imagelevel=True, isgroup=True)
            
            if box_only:
                gt_boxlist.extra_fields["labels"] = gt_boxlist.extra_fields["labels"].float()*0+1
                imagelevel_classes = imagelevel_classes*0+1

            labeled_classes = np.unique(imagelevel_classes.tolist() + gt_boxlist.extra_fields["labels"].tolist())
            valid_inds = np.nonzero(np.isin(pred_boxlists[-1].get_field("labels").cpu().numpy(), labeled_classes))[0].tolist()
            pred_boxlists[-1].bbox = pred_boxlists[-1].bbox[valid_inds,...]
            pred_boxlists[-1].extra_fields["labels"] = pred_boxlists[-1].extra_fields["labels"][valid_inds]
            pred_boxlists[-1].extra_fields["scores"] = pred_boxlists[-1].extra_fields["scores"][valid_inds]
        else:
            gt_boxlist = dataset.get_groundtruth(image_id, isgroup=True)
        gt_boxlists.append(gt_boxlist)

    ### iou 0.5 ###
    iou_thresh=0.5
    result = eval_detection_voc(
        pred_boxlists=pred_boxlists,
        gt_boxlists=gt_boxlists,
        iou_thresh=iou_thresh,
    )
    result_str = "mAP_{}: {:.4f}\n".format(iou_thresh,result["map"])
    for i, ap in enumerate(result["ap"]):
        if i == 0:  # skip background
            continue
        result_str += "{:<16}: {:.4f}\n".format(
            dataset.map_class_id_to_class_name[i], ap
        )
    result_str += "\n\nmAR_{}: {:.4f}\n".format(iou_thresh,result["mar"])
    ### iou 0.75 ###
    iou_thresh=0.75
    result = eval_detection_voc(
        pred_boxlists=pred_boxlists,
        gt_boxlists=gt_boxlists,
        iou_thresh=iou_thresh,
    )
    result_str += "\n\n\n mAP_{}: {:.4f}\n".format(iou_thresh,result["map"])
    for i, ap in enumerate(result["ap"]):
        if i == 0:  # skip background
            continue
        result_str += "{:<16}: {:.4f}\n".format(
            dataset.map_class_id_to_class_name[i], ap
        )
    result_str += "\n\nmAR_{}: {:.4f}\n".format(iou_thresh,result["mar"])   

    if box_only:
        ### iou 0.95 ###
        iou_thresh=0.95
        result = eval_detection_voc(
            pred_boxlists=pred_boxlists,
            gt_boxlists=gt_boxlists,
            iou_thresh=iou_thresh,
        )
        result_str += "\n\n\n mAP_{}: {:.4f}\n".format(iou_thresh,result["map"])
        for i, ap in enumerate(result["ap"]):
            if i == 0:  # skip background
                continue
            result_str += "{:<16}: {:.4f}\n".format(
                dataset.map_class_id_to_class_name[i], ap
            )
        result_str += "\n\nmAR_{}: {:.4f}\n".format(iou_thresh,result["mar"])    

    logger.info(result_str)
    if output_folder:
        '''save the output dictionary here'''
        ret_dic = package_visualization_result(pred_boxlists,gt_boxlists,list_image_names)
        with open(os.path.join(output_folder, "visualization_package.pkl"), 'wb') as f:
            pickle.dump({'ret_dic':ret_dic, 'map_class_id_to_class_name':dataset.map_class_id_to_class_name}, f)

        with open(os.path.join(output_folder, "result.txt"), "w") as fid:
            fid.write(result_str)
    return result


### Dat code ###
def match_pred_with_gt(pred_dic,gt_dic, iou_thresh = 0.5):
    pred_label = pred_dic['pred_label']
    pred_bbox = pred_dic['pred_bbox']
    pred_score = pred_dic['pred_score']

    gt_label = gt_dic['gt_label']
    gt_bbox = gt_dic['gt_bbox']
    gt_size = gt_dic['gt_size']

    match_dic = {}

    for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
        pred_mask_l = pred_label == l
        pred_bbox_l = pred_bbox[pred_mask_l]
        pred_score_l = pred_score[pred_mask_l]
        # sort by score
        order = pred_score_l.argsort()[::-1]
        pred_bbox_l = pred_bbox_l[order]
        pred_score_l = pred_score_l[order]

        gt_mask_l = gt_label == l
        gt_bbox_l = gt_bbox[gt_mask_l]

        if len(pred_bbox_l) == 0:
            continue
        if len(gt_bbox_l) == 0:
            #match[l].extend((0,) * pred_bbox_l.shape[0])
            continue

        # VOC evaluation follows integer typed bounding boxes.
        pred_bbox_l = pred_bbox_l.copy()
        pred_bbox_l[:, 2:] += 1
        gt_bbox_l = gt_bbox_l.copy()
        gt_bbox_l[:, 2:] += 1
        iou, iou2 = boxlist_iou(
            BoxList(pred_bbox_l, gt_size),
            BoxList(gt_bbox_l, gt_size),
            divideFirst=True,
        )
        iou, iou2 = iou.numpy(), iou2.numpy()
        gt_index = iou.argmax(axis=1)
        # set -1 if there is no matching ground truth
        gt_index[iou.max(axis=1) < iou_thresh] = -1

        match_dic[l] = {'pred_bbox_l':pred_bbox_l, 'gt_bbox_l':gt_bbox_l,'gt_index':gt_index}

    return match_dic

def package_visualization_result(pred_boxlists,gt_boxlists,list_image_names):
    zip_package = zip(pred_boxlists,gt_boxlists,list_image_names)
    ret_dic = {}
    for pred, gt, img_name in zip_package:
        pred_bbox = pred.bbox.numpy()
        pred_label = pred.get_field("labels").numpy()
        pred_score = pred.get_field("scores").numpy()
        pred_dic = {'pred_bbox':pred_bbox,
                    'pred_label':pred_label,
                    'pred_score':pred_score}

        gt_bbox = gt.bbox.numpy()
        gt_label = gt.get_field("labels").numpy()
        gt_size = gt.size
        gt_dic = {'gt_bbox':gt_bbox,
                'gt_label':gt_label,
                'gt_size':gt_size}
        
        match_dic_50 = match_pred_with_gt(pred_dic,gt_dic, iou_thresh = 0.5)
        match_dic_75 = match_pred_with_gt(pred_dic,gt_dic, iou_thresh = 0.75)
        match_dic_90 = match_pred_with_gt(pred_dic,gt_dic, iou_thresh = 0.9)
        
        ret_dic[img_name] = {'match_dic_50':match_dic_50,
                            'match_dic_75':match_dic_75,
                            'match_dic_90':match_dic_90
                            }

    return ret_dic

    
### Dat code ###

def eval_detection_voc(pred_boxlists, gt_boxlists, iou_thresh=0.5, use_07_metric=False):
    """Evaluate on voc dataset.
    Args:
        pred_boxlists(list[BoxList]): pred boxlist, has labels and scores fields.
        gt_boxlists(list[BoxList]): ground truth boxlist, has labels field.
        iou_thresh: iou thresh
        use_07_metric: boolean
    Returns:
        dict represents the results
    """
    assert len(gt_boxlists) == len(
        pred_boxlists
    ), "Length of gt and pred lists need to be same."
    prec, rec = calc_detection_voc_prec_rec(
        pred_boxlists=pred_boxlists, gt_boxlists=gt_boxlists, iou_thresh=iou_thresh
    )
    ap = calc_detection_voc_ap(prec, rec, use_07_metric=use_07_metric)
    ar = []
    for r in rec:
        if r is not None:
            if len(r) > 0:
                ar.append(r[-1])
    return {"ap": ap, "map": np.nanmean(ap), "mar": np.nanmean(ar)}


def calc_detection_voc_prec_rec(gt_boxlists, pred_boxlists, iou_thresh=0.5):
    """Calculate precision and recall based on evaluation code of PASCAL VOC.
    This function calculates precision and recall of
    predicted bounding boxes obtained from a dataset which has :math:`N`
    images.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
   """
    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)
    for gt_boxlist, pred_boxlist in zip(gt_boxlists, pred_boxlists):
        pred_bbox = pred_boxlist.bbox.numpy()
        pred_label = pred_boxlist.get_field("labels").numpy()
        pred_score = pred_boxlist.get_field("scores").numpy()
        gt_bbox = gt_boxlist.bbox.numpy()
        gt_label = gt_boxlist.get_field("labels").numpy()
        gt_isgroup = gt_boxlist.get_field("isgroup").numpy()
        gt_difficult = np.copy(gt_label).astype(np.uint8)*0

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            # if l == -100:
            #     continue
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_isgroup_l = gt_isgroup[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]

            n_pos[l] += np.logical_not(gt_difficult_l).sum()
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            # VOC evaluation follows integer typed bounding boxes.
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1
            iou, iou2 = boxlist_iou(
                BoxList(pred_bbox_l, gt_boxlist.size),
                BoxList(gt_bbox_l, gt_boxlist.size),
                divideFirst=True,
            )
            iou, iou2 = iou.numpy(), iou2.numpy()
            gt_index = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            # manage isgroup
            if gt_isgroup_l.max() == 1:
                isgroup_inds = np.nonzero(gt_isgroup_l)[0]
                iou2 = iou2[:,isgroup_inds]
                iou2_argmax = iou2.argmax(axis=1)
                bin_mask = (iou2.max(axis=1)>=0.5)
                bin_mask = np.logical_and(bin_mask, (gt_index==-1))
                gt_index[bin_mask] = isgroup_inds[iou2_argmax[bin_mask]]
            del iou, iou2

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if gt_difficult_l[gt_idx]:
                        match[l].append(-1)
                    else:
                        if not selec[gt_idx]:
                            match[l].append(1)
                        else:
                            match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)

    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]

    return prec, rec


def calc_detection_voc_ap(prec, rec, use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC.
    This function calculates average precisions
    from given precisions and recalls.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
    Args:
        prec (list of numpy.array): A list of arrays.
            :obj:`prec[l]` indicates precision for class :math:`l`.
            If :obj:`prec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        rec (list of numpy.array): A list of arrays.
            :obj:`rec[l]` indicates recall for class :math:`l`.
            If :obj:`rec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.
    Returns:
        ~numpy.ndarray:
        This function returns an array of average precisions.
        The :math:`l`-th value corresponds to the average precision
        for class :math:`l`. If :obj:`prec[l]` or :obj:`rec[l]` is
        :obj:`None`, the corresponding value is set to :obj:`numpy.nan`.
    """

    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            for t in np.arange(0.0, 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap
