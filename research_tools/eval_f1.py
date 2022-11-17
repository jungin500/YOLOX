#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import (configure_module, configure_nccl, fuse_model, get_local_rank, get_model_info, setup_logger)

import numpy as np
from generator.util import iou_np
import itertools

from matplotlib import pyplot as plt
import datetime


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("-d", "--devices", default=None, type=int, help="device for training")
    parser.add_argument("--num_machines", default=1, type=int, help="num of node for training")
    parser.add_argument("--machine_rank", default=0, type=int, help="node rank for multi-node training")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--test",
        dest="test",
        default=False,
        action="store_true",
        help="Evaluating on test-dev set.",
    )
    parser.add_argument(
        "--speed",
        dest="speed",
        default=False,
        action="store_true",
        help="speed test only.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def main(exp, args, num_gpu):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn("You have chosen to seed testing. This will turn on the CUDNN deterministic setting, ")

    is_distributed = num_gpu > 1

    # set environment variables for distributed training
    configure_nccl()
    cudnn.benchmark = True

    rank = get_local_rank()

    file_name = os.path.join(exp.output_dir, args.experiment_name)

    fig_savedir = os.path.join(file_name, datetime.datetime.now().strftime('%y%m%d_%H%M%S'))

    if rank == 0:
        os.makedirs(file_name, exist_ok=True)
        os.makedirs(fig_savedir, exist_ok=True)

    setup_logger(file_name, distributed_rank=rank, filename="val_log.txt", mode="a")
    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    evaluator = exp.get_evaluator(args.batch_size, is_distributed, args.test, args.legacy)
    evaluator.per_class_AP = True
    evaluator.per_class_AR = True

    torch.cuda.set_device(rank)
    model.cuda(rank)
    model.eval()

    if not args.speed and not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint from {}".format(ckpt_file))
        loc = "cuda:{}".format(rank)
        ckpt = torch.load(ckpt_file, map_location=loc)
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if is_distributed:
        model = DDP(model, device_ids=[rank])

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert (not args.fuse and not is_distributed
                and args.batch_size == 1), "TensorRT model is not support model fusing and distributed inferencing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(trt_file), "TensorRT model is not found!\n Run tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
    else:
        trt_file = None
        decoder = None

    # start evaluate f1 score
    from tqdm import tqdm
    from collections import ChainMap, defaultdict
    from yolox.utils import (gather, is_main_process, postprocess, synchronize, time_synchronized, xyxy2xywh)

    # variables for inference
    tensor_type = torch.cuda.HalfTensor if args.fp16 else torch.cuda.FloatTensor
    model = model.eval()
    if args.fp16:
        model = model.half()
    ids = []
    data_list = {}
    progress_bar = tqdm if is_main_process() else iter

    # variables for F1 score
    threshold_list = np.linspace(0.01, 0.99, num=100 - 1)

    for cur_iter, (imgs, _, info_imgs, ids) in enumerate(progress_bar(evaluator.dataloader)):
        with torch.no_grad():
            imgs = imgs.type(tensor_type)

            outputs = model(imgs)

            for threshold in threshold_list:
                p_outputs = postprocess(outputs, evaluator.num_classes, threshold, evaluator.nmsthre)

                data_list_elem = evaluator.convert_to_coco_format(p_outputs, info_imgs, ids, return_outputs=False)
                if threshold not in data_list:
                    data_list[threshold] = []

                data_list[threshold].extend(data_list_elem)

        if is_distributed:
            for threshold in threshold_list:
                data_list[threshold] = gather(data_list[threshold], dst=0)
                data_list[threshold] = list(itertools.chain(*data_list[threshold]))

    # Post-metadata gathering for plotting F1 scores
    # GT
    gt_items = {}
    for key, item in evaluator.dataloader.dataset.coco.anns.items():
        image_id = item['image_id']
        xmin, ymin, width, height = item['bbox']
        category_id = item['category_id']

        if image_id not in gt_items:
            gt_items[image_id] = []
        gt_items[image_id].append([xmin, ymin, xmin + width, ymin + height, category_id])

    # Inferred results
    infer_items = {}  # image_id -> threshold -> results
    for threshold in threshold_list:
        for item in data_list[threshold]:
            image_id = item['image_id']
            xmin, ymin, width, height = item['bbox']
            score = item['score']
            category_id = item['category_id']

            if image_id not in infer_items:
                infer_items[image_id] = {}
            if threshold not in infer_items[image_id]:
                infer_items[image_id][threshold] = []
            infer_items[image_id][threshold].append([xmin, ymin, xmin + width, ymin + height, category_id, score])

    # Generate F1 curve
    perclass_f1_scores = []
    for category_id in sorted(np.unique(np.concatenate(list(gt_items.values()), 0)[:, 4])):
        f1_scores = []
        precision_scores = []
        recall_scores = []

        for threshold in tqdm(threshold_list, desc="Generating F1 score map"):
            tp = []  # Matched IoU over threshold
            fp = []  # Doesn't matched IoU over threshold (Inferred items only)
            fn = []  # Doesn't matched IoU over threshold (GT items only)

            # pylint: disable=consider-using-dict-items
            for key in gt_items.keys():
                for g_xmin, g_ymin, g_xmax, g_ymax, g_cat in filter(lambda d: d[4] == category_id, gt_items[key]):
                    g_bbox = np.array([g_xmin, g_ymin, g_xmax, g_ymax])

                    # Infer 결과가 없는 경우
                    if key not in infer_items or len(infer_items[key]) == 0 \
                        or threshold not in infer_items[key] or len(infer_items[key][threshold]) == 0:
                        fn.append(g_bbox)
                        continue

                    # 최다 IoU 매칭
                    ious = []
                    items = []
                    for item in filter(lambda d: d[4] == category_id, infer_items[key][threshold]):
                        i_xmin, i_ymin, i_xmax, i_ymax, i_cat, i_score = item
                        i_bbox = np.array([i_xmin, i_ymin, i_xmax, i_ymax])
                        ious.append(iou_np(g_bbox, i_bbox))
                        items.append(item)  # Call-By-Reference, 추후 검색 및 삭제용

                    # Infer 결과가 없는 경우
                    if len(ious) == 0:
                        fn.append(g_bbox)
                        continue

                    maxarg = np.argsort(ious)[-1]

                    if ious[maxarg] >= 0.45:
                        # 해당하는 Infer 결과는 지운다.
                        infer_items[key][threshold].remove(items[maxarg])
                        tp.append(g_bbox)
                    else:
                        fn.append(g_bbox)

                # 매칭이 끝나고 남은 인퍼런스 결과들은 FP이다.
                if key in infer_items and threshold in infer_items[key]:
                    for i_xmin, i_ymin, i_xmax, i_ymax, i_cat, i_score in infer_items[key][threshold]:
                        i_bbox = np.array([i_xmin, i_ymin, i_xmax, i_ymax])
                        fp.append(i_bbox)

            if len(tp) + len(fp) == 0:
                precision = 0
                recall = 0
            else:
                precision = len(tp) / (len(tp) + len(fp))
                recall = len(tp) / (len(tp) + len(fn))

            precision_scores.append(precision)
            recall_scores.append(recall)

            if precision + recall == 0:
                f1_scores.append(.0)
            else:
                f1_scores.append(2 * (precision * recall) / (precision + recall))

        if category_id == 1:
            class_name = "helmet_on"
            color = "blue"
        elif category_id == 2:
            class_name = "helmet_off"
            color = "blue"
        elif category_id == 3:
            class_name = "belt_on"
            color = "blue"
        elif category_id == 4:
            class_name = "belt_off"
            color = "blue"

        # Draw F1 score map in each class plot
        logger.info("Drawing F1 score map for class {}".format(class_name))
        with open(os.path.join(fig_savedir, '{}.txt'.format(class_name)), 'w') as f:
            f.write(' '.join(['{:.4f}'.format(i) for i in threshold_list]))
            f.write('\n')
            f.write(' '.join(['{:.4f}'.format(i) for i in f1_scores]))

        max_threshold = threshold_list[np.argmax(f1_scores)]

        plt.figure(figsize=(8, 4))
        plt.title("{} F1 Score by Detection Threshold (NMS={:.02f})".format(class_name, exp.nmsthre))
        f1_line, = plt.plot(threshold_list, f1_scores, color=color, linestyle='dashed', label="F1 Score")
        precision_line, = plt.plot(threshold_list,
                                   precision_scores,
                                   color="#ff7f00",
                                   linestyle='dashed',
                                   label="Precision")
        recall_line, = plt.plot(threshold_list, recall_scores, color="#007fff", linestyle='dashed', label="Recall")
        max_score_line = plt.axvline(max_threshold,
                                     color='r',
                                     linestyle='dashdot',
                                     label="Max score ({:.02f})".format(max_threshold))
        plt.legend(handles=[precision_line, recall_line, f1_line, max_score_line], loc='upper right')
        plt.xlabel("Confidence Threshold")
        plt.ylabel("F1 Score")
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        plt.xticks(np.arange(0, 1, step=0.1))
        plt.yticks(np.arange(0.1, 1, step=0.1))
        plt.savefig(os.path.join(fig_savedir, "{}.svg".format(class_name)), dpi=600, transparent=True)

        perclass_f1_scores.append(f1_scores)

    # Draw and save in single plot
    plt.figure(figsize=(8, 4))
    plt.title("F1 Score by Detection Threshold (NMS={:.02f})".format(exp.nmsthre))
    plt.xlabel("Confidence Threshold")
    plt.ylabel("F1 Score")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xticks(np.arange(0, 1, step=0.1))
    plt.yticks(np.arange(0.1, 1, step=0.1))

    all_lines = []
    thresholds = []
    for class_id, f1_scores in enumerate(perclass_f1_scores):
        category_id = class_id + 1
        if category_id == 1:
            class_name = "helmet_on"
            color = "#6EF018"
            linestyle = 'dashed'
        elif category_id == 2:
            class_name = "helmet_off"
            color = "#18A5F0"
            linestyle = 'dashed'
        elif category_id == 3:
            class_name = "belt_on"
            color = "#E718F0"
            linestyle = 'dashed'
        elif category_id == 4:
            class_name = "belt_off"
            color = "#F08C0C"
            linestyle = 'dashed'

        max_threshold = threshold_list[np.argmax(f1_scores)]
        thresholds.append(max_threshold)

        f1_line, = plt.plot(threshold_list,
                            f1_scores,
                            color=color,
                            linestyle=linestyle,
                            label="F1 Score ({})".format(class_name))
        all_lines.append(f1_line)

    avg_max_threshold = np.mean(thresholds)
    max_score_line = plt.axvline(avg_max_threshold,
                                 color='r',
                                 linestyle='dashdot',
                                 label="Average max score ({:.02f})".format(avg_max_threshold))

    plt.legend(handles=[*all_lines, max_score_line], loc='upper right')
    plt.savefig(os.path.join(fig_savedir, "all.svg"), dpi=600, transparent=True)

    logger.info("F1 plots are drawn into {}".format(fig_savedir))


if __name__ == "__main__":
    configure_module()
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args, num_gpu),
    )
