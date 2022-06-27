import logging

from .openimages_coco_eval import do_openimages_coco_evaluation


def openimages_evaluation(dataset,
    predictions,
    box_only,
    output_folder,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
    ):
    return do_openimages_coco_evaluation(
        dataset=dataset,
        predictions=predictions,
        box_only=box_only,
        output_folder=output_folder,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )
