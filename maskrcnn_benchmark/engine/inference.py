# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os

import torch
from tqdm import tqdm

from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from .bbox_aug import im_detect_bbox_aug
import cv2
from maskrcnn_benchmark.utils import cv2_util
import numpy as np
import matplotlib.pyplot as plt

from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from collections import defaultdict
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.structures.bounding_box import BoxList

def compute_on_dataset(model, data_loader, device, bbox_aug, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        with torch.no_grad():
            if timer:
                timer.tic()
            if bbox_aug:
                output = im_detect_bbox_aug(model, images, device)
            else:
                output = model(images.to(device), targets)
            if timer:
                if not device.type == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    return results_dict


def compute_on_dataset_custom(model, data_loader, device, bbox_aug, timer=None, size = -1):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for idx, batch in enumerate(tqdm(data_loader)):
        if size != -1 and idx >= size:
            break
        images, targets, image_ids = batch

        with torch.no_grad():
            if timer:
                timer.tic()
            try:
                if bbox_aug:
                    output = im_detect_bbox_aug(model, images, device)
                else:
                    output = model(images.to(device), targets)
            except:
                output = [BoxList(bbox = torch.zeros(0,4), image_size=(1,1))]
            if timer:
                if not device.type == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]

        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    if size != -1:
        assert len(results_dict) == size
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        bbox_aug=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    
    # zero-shot models should have class embeddings for inference to map predicted embeddings to classes
    module = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    if hasattr(module, 'roi_heads') and 'box' in module.roi_heads:
        if module.roi_heads['box'].predictor.embedding_based:
            module.roi_heads['box'].predictor.set_class_embeddings(
                data_loader.dataset.class_emb_mtx)
            
            module.class_names = data_loader.dataset.class_names

    if hasattr(module,'exemplars'):
        module.load_exemplars()

    
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, device, bbox_aug, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)

def select_top_predictions(predictions, confidence_threshold = 0.5):
    """
    Select only predictions which have a `score` > self.confidence_threshold,
    and returns the predictions in descending order of score

    Arguments:
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores`.

    Returns:
        prediction (BoxList): the detected objects. Additional information
            of the detection properties can be found in the fields of
            the BoxList via `prediction.fields()`
    """
    scores = predictions.get_field("scores")
    keep = torch.nonzero(scores > confidence_threshold).squeeze(1)
    predictions = predictions[keep]
    scores = predictions.get_field("scores")
    _, idx = scores.sort(0, descending=True)
    return predictions[idx]

def nms(pred):
    labels = pred.get_field('labels')
    unique_labels = labels.unique()
    list_preds = []
    #selected_idxs = []
    for l in unique_labels:
        idxs = torch.where(labels==l)[0]
        # pred_l = pred[idxs]
        # scores = pred_l.get_field('scores')
        idx_max = idxs[:1]#torch.argmax(scores,keepdim=True)
        list_preds.append(pred[idx_max])

    return list_preds

def visualization_uncertainty(
        model,
        data_loader,
        dataset_name,
        bbox_aug=False,
        device="cuda",
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    
    # zero-shot models should have class embeddings for inference to map predicted embeddings to classes
    module = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    if hasattr(module, 'roi_heads') and 'box' in module.roi_heads:
        if module.roi_heads['box'].predictor.embedding_based:
            module.roi_heads['box'].predictor.set_class_embeddings(
                data_loader.dataset.class_emb_mtx)
    
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset_custom(model, data_loader, device, bbox_aug, inference_timer, size=400)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )
    
    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

        mask_dir = output_folder+'/mask_img/'
        if not os.path.isdir(mask_dir):
            os.mkdir(mask_dir)

        uncertainty_dir = output_folder+'/uncertainty_img/'
        if not os.path.isdir(uncertainty_dir):
            os.mkdir(uncertainty_dir)

        combine_dir = output_folder+'/combine_dir/'
        if not os.path.isdir(combine_dir):
            os.mkdir(combine_dir)

        vis_stats = defaultdict(lambda: 0)
        CATEGORIES = dataset.class_names
        mask_threshold = -1
        masker = Masker(threshold=mask_threshold, padding=1)
        cap_vocab = module.cap_vocab

        for image_id in range(len(predictions)):
            img_info = dataset.get_img_info(image_id)
            image_width = img_info["width"]
            image_height = img_info["height"]
            
            file_name = img_info['file_name']
            img_path = os.path.join(dataset._image_root, file_name)

            pred = predictions[image_id].resize((image_width, image_height))
            if len(pred) == 0:
                continue
            
            labels = pred.get_field("labels").cpu()
            mask_uncertainty = pred.get_field("mask_uncertainty")
            pred.add_field("mask_uncertainty", mask_uncertainty/mask_uncertainty.max())
            pred.add_field("scores", 0.01/mask_uncertainty.mean(dim=[1,2,3]))

            joined_class_names = pred.get_field("joined_class_names")
            class_names = joined_class_names.split('/')

            is_save = True

            if is_save:
                print('saved', file_name)
                list_preds = nms(pred)
                
                img = cv2.imread(img_path)
                for pred in list_preds:
                    target_masks = pred.get_field("target_masks")
                    masks = target_masks.get_mask_tensor().long()[None]#target_masks.resize((image_width, image_height)).get_mask_tensor()
                    pred.add_field("teacher_mask", masks)
                    pred.add_field("mask", masks)
                    #img = overlay_boxes(img,pred,[])
                    img = overlay_filled_mask(img, pred)
                    #img = overlay_class_names(img, pred, cap_vocab, [])
                cv2.imwrite(mask_dir+file_name,img)

                img = cv2.imread(img_path)
                for pred in list_preds:
                    mask_uncertainty = pred.get_field("mask_uncertainty")
                    # always single image is passed at a time
                    mask_uncertainty = masker([mask_uncertainty], [pred])[0]
                    pred.add_field("uncertainty", mask_uncertainty)
                    pred.add_field("mask", mask_uncertainty)
                    #img = overlay_boxes(img,pred,[])
                    img = overlay_uncertainty_mask(img, pred)
                    img = overlay_class_names(img, pred, cap_vocab, [])
                cv2.imwrite(uncertainty_dir+file_name,img)

                img = cv2.imread(img_path)
                for pred in list_preds:
                    pred.add_field("mask", mask_uncertainty)

                    #img = overlay_boxes(img,pred,[])

                    pred.add_field("mask", pred.get_field('teacher_mask'))
                    img = overlay_filled_mask(img, pred)

                    pred.add_field("mask", pred.get_field('uncertainty'))
                    img = overlay_uncertainty_mask(img, pred)

                    img = overlay_class_names(img, pred, cap_vocab, [])
                cv2.imwrite(combine_dir+file_name,img)

    return None

def visualization_mask(
        model,
        data_loader,
        dataset_name,
        bbox_aug=False,
        device="cuda",
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    
    # zero-shot models should have class embeddings for inference to map predicted embeddings to classes
    module = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    if hasattr(module, 'roi_heads') and 'box' in module.roi_heads:
        if module.roi_heads['box'].predictor.embedding_based:
            module.roi_heads['box'].predictor.set_class_embeddings(
                data_loader.dataset.class_emb_mtx)
    
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset_custom(model, data_loader, device, bbox_aug, inference_timer, size=-1)
    print(len(predictions))
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

        mask_dir = output_folder+'/mask_img/'
        if not os.path.isdir(mask_dir):
            os.mkdir(mask_dir)

        vis_stats = defaultdict(lambda: 0)
        CATEGORIES = dataset.class_names
        mask_threshold = 0.5
        masker = Masker(threshold=mask_threshold, padding=1)
        class_splits = dataset.class_splits

        unseen_class = [dataset.json_category_id_to_contiguous_id[l] for l in class_splits['unseen']]
        target_files = ['000000025394.jpg','000000020247.jpg','000000000632.jpg'] #['000000087476.jpg','000000088462.jpg','000000090108.jpg','000000097230.jpg','000000575357.jpg','000000581100.jpg']
        for image_id in range(len(predictions)):
            img_info = dataset.get_img_info(image_id)
            image_width = img_info["width"]
            image_height = img_info["height"]
            
            file_name = img_info['file_name']
            img_path = os.path.join(dataset._image_root, file_name)
            img = cv2.imread(img_path)

            pred = predictions[image_id].resize((image_width, image_height))
            pred = select_top_predictions(pred, confidence_threshold = 0.5)
            labels = pred.get_field("labels").cpu()

            is_save = False
            for l in labels:
                l_name = CATEGORIES[l]
                if (l in unseen_class and vis_stats[l] < 20) or (file_name in target_files):
                    is_save = True
                    vis_stats[l] += 1

            if is_save:
                print('saved', file_name)
                masks = pred.get_field("mask")
                # always single image is passed at a time
                masks = masker([masks], [pred])[0]
                pred.add_field("mask", masks)
                boxed_img = overlay_boxes(img,pred,unseen_class)
                masked_img = overlay_filled_mask(boxed_img, pred)
                masked_img = overlay_class_names(masked_img, pred, CATEGORIES, unseen_class,display_score=False)
                cv2.imwrite(mask_dir+file_name,masked_img)

    return None

def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=3,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h), font, font_scale, text_color, font_thickness)

    return text_size

def overlay_class_names(image, predictions, CATEGORIES, unseen_split, display_score = True):
    """
    Adds detected class names and scores in the positions defined by the
    top-left corner of the predicted bounding box

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores` and `labels`.
    """
    scores = predictions.get_field("scores").tolist()
    labels = predictions.get_field("labels").tolist()
    label_names = [CATEGORIES[i] for i in labels]
    label_names = [l.upper() if i in unseen_split else l 
                    for i, l in zip(labels, label_names)]
    boxes = predictions.bbox.cpu().long().numpy()

    img_size = predictions.size
    max_area = np.max((boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]))/(img_size[0]*img_size[1])

    if display_score:
        template = "{} {:.2f}"
    else:
        template = "{}"
    for box, score, lidx, lname in zip(boxes, scores, labels, label_names):
        x1, y1 = box[:2]
        x2, y2 = box[2:]

        area = (x2-x1)*(y2-y1)/(img_size[0]*img_size[1])
        # if area/max_area < 0.01:
        #     continue
        text_size = area if area > 0.4 else 0.4
        if display_score:
            s = template.format(lname,score)
        else:
            s = template.format(lname)
        color = (0, 0, 255) if lidx in unseen_split else (0, 0, 0)
        pos = (x1,y1)#((x1+x2)//2, (y1+y2)//2)
        draw_text(img=image, text=s,
          font=cv2.FONT_HERSHEY_SIMPLEX,
          pos=pos,
          font_scale=text_size,
          font_thickness=1,
          text_color=color,
          text_color_bg=(255, 255, 255)
        )

    return image

def compute_colors_for_labels(labels):
    """
    Simple function that adds fixed colors depending on the class
    """
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = labels[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    return colors

def overlay_boxes(image, predictions,unseen_split):
    """
    Adds the predicted boxes on top of the image

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    """
    labels = predictions.get_field("labels")
    boxes = predictions.bbox

    for box, label in zip(boxes, labels):
        thickness = 2
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        color = (0,0,0) if label not in unseen_split else (0,0,255)
        image = cv2.rectangle(
            image, tuple(top_left), tuple(bottom_right), tuple(color), thickness
        )

    return image

def random_color(rgb=False, maximum=255):
    """
    Args:
        rgb (bool): whether to return RGB colors or BGR colors.
        maximum (int): either 255 or 1

    Returns:
        ndarray: a vector of 3 numbers
    """
    idx = np.random.randint(0, len(_COLORS))
    ret = _COLORS[idx] * maximum
    if not rgb:
        ret = ret[::-1]
    return ret

def overlay_filled_mask(image, predictions):
    masks = predictions.get_field("mask").numpy()
    labels = predictions.get_field("labels")
    #colors = compute_colors_for_labels(labels).tolist()
    colors = [random_color(rgb=True, maximum=255) for _ in range(labels.shape[0])]
    masked_image=image.astype(np.uint8).copy()

    for mask, color in zip(masks, colors):
        for c in range(3):
            masked_image[:, :, c] = np.where(mask == 1, masked_image[:, :, c] * (1 - 0.5) + 0.5 * color[c], masked_image[:, :, c])
    composite = masked_image

    return composite

def overlay_uncertainty_mask(image, predictions):
    masks = predictions.get_field("mask").numpy()
    labels = predictions.get_field("labels")
    #colors = compute_colors_for_labels(labels).tolist()
    colors = [(0,0,255) for _ in range(labels.shape[0])]
    masked_image=image.astype(np.uint8).copy()
    scores = predictions.get_field("scores").tolist()

    for mask, color, s in zip(masks, colors, scores):
        mask_area = np.sum(mask != 0)
        overall_area = mask_area+np.sum(mask == 0)
        area_factor = (mask_area/overall_area)*10
        # mask = np.clip(mask/area_factor,a_min=0.0,a_max=1.0)
        mask = np.clip(mask * (0.2/s),a_min=0.0,a_max=1.0)
        for c in range(3):
            masked_image[:, :, c] = np.where(mask != 0, masked_image[:, :, c] * (1 - mask) + mask * color[c], masked_image[:, :, c])
    composite = masked_image

    return composite

def overlay_mask(image, predictions):
    """
    Adds the instances contours for each predicted object.
    Each label has a different color.

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `mask` and `labels`.
    """
    masks = predictions.get_field("masks")
    labels = predictions.get_field("labels")

    colors = compute_colors_for_labels(labels).tolist()

    for mask, color in zip(masks, colors):
        mask = mask.get_mask_tensor().numpy()
        thresh = mask[:, :, None].astype(np.uint8)
        contours, hierarchy = cv2_util.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        image = cv2.drawContours(image, contours, -1, color, 3)

    composite = image

    return composite

sorted_CATEGORIES = [
        "__background",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

# RGB:
_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.857, 0.857, 0.857,
        1.000, 1.000, 1.000
    ]
).astype(np.float32).reshape(-1, 3)