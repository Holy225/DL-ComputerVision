# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from utils.nms.cpu_nms import cpu_nms
#from utils.nms.gpu_nms import gpu_nms
import numpy as np


# def nms(dets, thresh, force_cpu=False):
#     """Dispatch to either CPU or GPU NMS implementations."""
#
#     if dets.shape[0] == 0:
#         return []
#     if cfg.USE_GPU_NMS and not force_cpu:
#         return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
#     else:
#         return cpu_nms(dets, thresh)


def nms(dets, thresh, force_cpu=False):
    """
    Dispatch NMS to either CPU or GPU
    :param dets: list of detections
    :param thresh: NMS overlap threshold
    :param force_cpu: Boolean to force usage of CPU implementation
    :return:
    """

    if dets.shape[0] == 0:
        return []
    
    return cpu_nms(dets, thresh)
    """
    if force_cpu:
        return cpu_nms(dets, thresh)
    return gpu_nms(dets, thresh)
    """


def nms_detections(pred_boxes, scores, nms_thresh):
    """
    Suppression of non maximum detections
    :param pred_boxes: Predicted boxes
    :param scores: Prediction boxes scores
    :param nms_thresh: NMS overlap threshold
    :return: List of selected detections
    """
    dets = np.hstack((pred_boxes,
                      scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, nms_thresh)
    return keep
