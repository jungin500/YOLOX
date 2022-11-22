'''
    trainval_json과 generated_json간의 evaluation을 진행한다.
    evaluation의 경우 "generated_json"쪽이 model output이라고 생각하고 진행한다.
'''
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from yolox.evaluators.coco_evaluator import per_class_AP_table, per_class_AR_table
import sys
import numpy as np
import contextlib
import io
import json


def reverse_parse(result_str: str):
    l_ap5095, l_ap50, _, l_ap_s, l_ap_m, l_ap_l, *_ = result_str.splitlines()
    ap5095 = float(l_ap5095.split()[-1])
    ap50 = float(l_ap50.split()[-1])
    ap_s = float(l_ap_s.split()[-1])
    ap_m = float(l_ap_m.split()[-1])
    ap_l = float(l_ap_l.split()[-1])
    return ap5095, ap50, ap_s, ap_m, ap_l


# Adopted from yolox.evaluators.coco_evaluator.per_class_AR_table
def per_class_AR(coco_eval, class_names):
    per_class_AR = []
    recalls = coco_eval.eval["recall"]
    # dimension of recalls: [TxKxAxM]
    # recall has dims (iou, cls, area range, max dets)
    assert len(class_names) == recalls.shape[1]

    for idx, _ in enumerate(class_names):
        recall = recalls[:, idx, 0, -1]
        recall = recall[recall > -1]
        ar = np.mean(recall) if recall.size else float("nan")
        per_class_AR.append(float(ar))

    return per_class_AR


# Adopted from yolox.evaluators.coco_evaluator.per_class_AP_table
def per_class_AP(coco_eval, class_names):
    per_class_AP = []
    precisions = coco_eval.eval["precision"]
    # dimension of precisions: [TxRxKxAxM]
    # precision has dims (iou, recall, cls, area range, max dets)
    assert len(class_names) == precisions.shape[2]

    for idx, _ in enumerate(class_names):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        per_class_AP.append(float(ap))

    return per_class_AP


def eval_coco(trainval_json, generated_json):
    with open(generated_json) as f:
        bpdy = json.load(f)
    generated_list = [
        {
            'image_id': item['image_id'],  # [1, ]
            'bbox': item['bbox'],  # [4, ]
            'score': 1.0,
            'category_id': item['category_id']
        } for item in bpdy['annotations']
    ]  # {imageID,x1,y1,w,h,score,class}

    # skip noninformative printouts
    nullout = io.StringIO()
    with contextlib.redirect_stdout(nullout):
        cocoGt = COCO(trainval_json)
        cocoDt = cocoGt.loadRes(generated_list)

        cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
        cocoEval.evaluate()
        cocoEval.accumulate()

    redirect_string = io.StringIO()
    with contextlib.redirect_stdout(redirect_string):
        cocoEval.summarize()
    info = redirect_string.getvalue()
    cat_ids = list(cocoGt.cats.keys())
    cat_names = [cocoGt.cats[catId]['name'] for catId in sorted(cat_ids)]

    ap5095, ap50, ap_s, ap_m, ap_l = reverse_parse(info)

    AP_table = per_class_AP_table(cocoEval, class_names=cat_names)
    AP_perclass = per_class_AP(cocoEval, class_names=cat_names)
    info += "per class AP(IoU=0.5~0.95):\n" + AP_table + "\n"

    AR_table = per_class_AR_table(cocoEval, class_names=cat_names)
    AR_perclass = per_class_AR(cocoEval, class_names=cat_names)
    info += "per class AR(IoU=0.5~0.95):\n" + AR_table + "\n"

    # Additional evaluation for per_class_AP IoU=0.5
    with contextlib.redirect_stdout(nullout):
        cocoEval_50 = COCOeval(cocoGt, cocoDt, "bbox")
        cocoEval_50.params.iouThrs = np.array([0.5])  # Only 0.5
        cocoEval_50.evaluate()
        cocoEval_50.accumulate()

    AP50_table = per_class_AP_table(cocoEval_50, class_names=cat_names)
    AP50_perclass = per_class_AP(cocoEval_50, class_names=cat_names)
    info += "per class AP(IoU=0.5):\n" + AP50_table + "\n"

    print(info)
    return ap5095, ap50, ap_s, ap_m, ap_l, AP_perclass, AR_perclass, AP50_perclass


if __name__ == '__main__':
    if len(sys.argv) == 1:
        # 같은 놈으로 하면 100%가 나와야 한다.
        target_json = '/home/jungin500/workspace/paper/sparse-label-assignment/yolox-datasets/construction-safety/annotations/annotations_test.json'
        eval_coco(target_json, target_json)
    elif len(sys.argv) == 2:
        target_json = sys.argv[1]
        eval_coco(target_json, target_json)
    elif len(sys.argv) == 3:
        eval_coco(sys.argv[1], sys.argv[2])
    else:
        print("Usage: {} [source_json [target_json]]".format(sys.argv[0]))