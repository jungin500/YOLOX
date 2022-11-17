import numpy as np
import torch


def iou(bbox1, bbox2):
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2

    rect1_width, rect1_height = xmax1 - xmin1, ymax1 - ymin1
    rect2_width, rect2_height = xmax2 - xmin2, ymax2 - ymin2
    rect1_vol = rect1_width * rect1_height
    rect2_vol = rect2_width * rect2_height

    inter_xmin, inter_ymin = max(xmin1, xmin2), max(ymin1, ymin2)
    inter_xmax, inter_ymax = min(xmax1, xmax2), min(ymax1, ymax2)
    is_intersected = (inter_xmin < inter_xmax) and (inter_ymin < inter_ymax)
    if is_intersected:
        inter_width = inter_xmax - inter_xmin
        inter_height = inter_ymax - inter_ymin
        inter_vol = inter_width * inter_height
        union_vol = rect1_vol + rect2_vol - inter_vol
        iou = inter_vol / union_vol
    else:
        inter_vol = 0
        iou = 0.0

    return iou


def iou_np(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    w, h = bbox1[2:] - bbox1[:2]
    bbox1_vol = w * h
    w, h = bbox2[2:] - bbox2[:2]
    bbox2_vol = w * h

    b_all = np.vstack([bbox1, bbox2])
    inter_xymin = np.max(b_all[:, :2], axis=0)
    inter_xymax = np.min(b_all[:, 2:], axis=0)
    if np.any(inter_xymax < inter_xymin):
        return .0

    w, h = inter_xymax - inter_xymin
    inter_vol = w * h
    union_vol = bbox1_vol + bbox2_vol - inter_vol
    return inter_vol / union_vol


def iou_torch(bbox1: torch.Tensor, bbox2: torch.Tensor):
    w, h = bbox1[2:] - bbox1[:2]
    bbox1_vol = w * h
    w, h = bbox2[2:] - bbox2[:2]
    bbox2_vol = w * h

    b_all = torch.vstack([bbox1, bbox2])
    inter_xymin = torch.max(b_all[:, :2], dim=0)[0]  # returns values, indices
    inter_xymax = torch.min(b_all[:, 2:], dim=0)[0]  # returns values, indices

    w, h = inter_xymax - inter_xymin
    inter_vol = w * h
    union_vol = bbox1_vol + bbox2_vol - inter_vol

    return torch.where(torch.any(inter_xymax < inter_xymin), 0., inter_vol / union_vol)
