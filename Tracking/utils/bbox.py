import numpy as np
import cv2


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries
    :param boxes: list of boxes
    :param im_shape: image shape
    :return: list of clipped boxes
    """
    boxes = np.asarray(boxes)
    if boxes.shape[0] == 0:
        return boxes
    boxes = np.copy(boxes)
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def clip_box(bbox, im_shape):
    """
    Clip single bounding box
    :param bbox: bounding box
    :param im_shape: Target shape of boundary box
    :return: Clipped boundary box
    """
    h, w = im_shape[:2]
    bbox = np.copy(bbox)
    bbox[0] = max(min(bbox[0], w - 1), 0)
    bbox[1] = max(min(bbox[1], h - 1), 0)
    bbox[2] = max(min(bbox[2], w - 1), 0)
    bbox[3] = max(min(bbox[3], h - 1), 0)

    return bbox


def int_box(box):
    """
    Round box pixel values to integer
    :param box: Bounding Box
    :return: Box with rounded values
    """
    box = np.asarray(box, dtype=np.float)
    box = np.round(box)
    return np.asarray(box, dtype=np.int)


# for display
############################
def _to_color(indx, base):
    """
    Converts an index to a color
    :param indx: Color index
    :param base: Base color
    :return: (b, r, g) tuple
    """
    base2 = base * base
    b = 2 - indx / base2
    r = 2 - (indx % base2) / base
    g = 2 - (indx % base2) % base
    return b * 127, r * 127, g * 127


def get_color(indx, cls_num=1):
    """
    Assigns color to a bounding box
    :param indx: Bounding box index number
    :param cls_num: Minimum index number for assignement
    :return: (b,r,g) color of the bounding box
    """
    if indx >= cls_num:
        return (23 * indx % 255, 47 * indx % 255, 137 * indx % 255)
    base = int(np.ceil(pow(cls_num, 1. / 3)))
    return _to_color(indx, base)


def draw_detection(im, bboxes, scores=None, cls_inds=None, cls_name=None):
    """
    Draw detection boxes in image
    :param im: Input picture
    :param bboxes: List of bounding
    :param scores: Scores of detection
    :param cls_inds: List of detected boxes indexes
    :param cls_name: List of detected boxes names
    :return: img in cv format with bounding boxes
    """
    # draw image
    bboxes = np.round(bboxes).astype(np.int)
    if cls_inds is not None:
        cls_inds = cls_inds.astype(np.int)
    cls_num = len(cls_name) if cls_name is not None else 2

    imgcv = np.copy(im)
    h, w, _ = imgcv.shape
    for i, box in enumerate(bboxes):
        cls_indx = cls_inds[i] if cls_inds is not None else 1
        color = get_color(cls_indx, cls_num)

        thick = int((h + w) / 600)
        cv2.rectangle(imgcv,
                      (box[0], box[1]), (box[2], box[3]),
                      color, thick)

        if cls_indx is not None:
            score = scores[i] if scores is not None else 1
            name = cls_name[cls_indx] if cls_name is not None else str(cls_indx)
            mess = '%s: %.3f' % (name, score) if cls_inds is not None else '%.3f' % (score, )
            cv2.putText(imgcv, mess, (box[0], box[1] - 12),
                        0, 1e-3 * h, color, thick // 3)

    return imgcv
