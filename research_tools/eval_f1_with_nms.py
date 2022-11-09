#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# NMS, Conf-Thresh aware F1 Detection Curve

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
from yolox.utils import (
    configure_module,
    configure_nccl,
    fuse_model,
    get_local_rank,
    get_model_info,
    setup_logger
)

import numpy as np
from generator.util import iou_np
import itertools

from matplotlib import pyplot as plt
import datetime
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D, art3d
from matplotlib.patches import Circle


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
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
        warnings.warn(
            "You have chosen to seed testing. This will turn on the CUDNN deterministic setting, "
        )

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
        assert (
            not args.fuse and not is_distributed and args.batch_size == 1
        ), "TensorRT model is not support model fusing and distributed inferencing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
    else:
        trt_file = None
        decoder = None

    # start evaluate f1 score
    from tqdm import tqdm
    from collections import ChainMap, defaultdict
    from yolox.utils import (
        gather,
        is_main_process,
        postprocess,
        synchronize,
        time_synchronized,
        xyxy2xywh
    )
    
    # variables for inference
    tensor_type = torch.cuda.HalfTensor if args.fp16 else torch.cuda.FloatTensor
    model = model.eval()
    if args.fp16:
        model = model.half()
    ids = []
    data_list = {}
    progress_bar = tqdm if is_main_process() else iter

    # variables for F1 score
    conf_thresh_list = np.linspace(0.01, 0.99, num=100 - 1)
    nms_thresh_list = np.linspace(0.1, 0.9, num=10 - 1)
    
    def thresh2keystring(conf_thresh, nms_thresh):
        return '{:.02f} {:.02f}'.format(conf_thresh, nms_thresh)

    for cur_iter, (imgs, _, info_imgs, ids) in enumerate(progress_bar(evaluator.dataloader)):
        with torch.no_grad():
            imgs = imgs.type(tensor_type)

            outputs = model(imgs)

            for conf_thresh in conf_thresh_list:
                for nms_thresh in nms_thresh_list:
                    p_outputs = postprocess(
                        outputs, evaluator.num_classes, conf_thresh, nms_thresh
                    )

                    data_list_elem = evaluator.convert_to_coco_format(
                        p_outputs, info_imgs, ids, return_outputs=False)
                    
                    key = thresh2keystring(conf_thresh, nms_thresh)
                    if key not in data_list:
                        data_list[key] = []
                    data_list[key].extend(data_list_elem)

        if is_distributed:
            for conf_thresh in conf_thresh_list:
                key = thresh2keystring(conf_thresh, nms_thresh)
                data_list[key] = gather(data_list[key], dst=0)
                data_list[key] = list(itertools.chain(*data_list[key]))

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
    infer_items = {}  # image_id -> key(conf_thresh, nms_thresh) -> results
    for conf_thresh in conf_thresh_list:
        for nms_thresh in nms_thresh_list:
            key = thresh2keystring(conf_thresh, nms_thresh)
            for item in data_list[key]:
                image_id = item['image_id']
                xmin, ymin, width, height = item['bbox']
                score = item['score']
                category_id = item['category_id']

                if image_id not in infer_items:
                    infer_items[image_id] = {}
                if key not in infer_items[image_id]:
                    infer_items[image_id][key] = []
                infer_items[image_id][key].append([xmin, ymin, xmin + width, ymin + height, category_id, score])

    # Generate F1 curve
    all_f1_scores = []
    for category_id in sorted(np.unique(np.concatenate(list(gt_items.values()), 0)[:, 4])):
        f1_scores = np.zeros((len(conf_thresh_list), len(nms_thresh_list)))

        for conf_thresh_idx, conf_thresh in tqdm(enumerate(conf_thresh_list), desc="Generating F1 score map", total=len(conf_thresh_list)):
            for nms_thresh_idx, nms_thresh in enumerate(nms_thresh_list):
                tp = []  # Matched IoU over threshold
                fp = []  # Doesn't matched IoU over threshold (Inferred items only)
                fn = []  # Doesn't matched IoU over threshold (GT items only)
                
                key = thresh2keystring(conf_thresh, nms_thresh)

                for image_id in gt_items.keys():
                    for g_xmin, g_ymin, g_xmax, g_ymax, g_cat in filter(lambda d: d[4] == category_id, gt_items[image_id]):
                        g_bbox = np.array([g_xmin, g_ymin, g_xmax, g_ymax])

                        # Infer 결과가 없는 경우
                        if image_id not in infer_items or len(infer_items[image_id]) == 0 \
                            or key not in infer_items[image_id] or len(infer_items[image_id][key]) == 0:
                            fn.append(g_bbox)
                            continue
                        
                        # 최다 IoU 매칭
                        ious = []
                        items = []
                        for item in filter(lambda d: d[4] == category_id, infer_items[image_id][key]):
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
                            infer_items[image_id][key].remove(items[maxarg])
                            tp.append(g_bbox)
                        else:
                            fn.append(g_bbox)

                    # 매칭이 끝나고 남은 인퍼런스 결과들은 FP이다.
                    if image_id in infer_items and key in infer_items[image_id]:
                        for i_xmin, i_ymin, i_xmax, i_ymax, i_cat, i_score in infer_items[image_id][key]:
                            i_bbox = np.array([i_xmin, i_ymin, i_xmax, i_ymax])
                            fp.append(i_bbox)
                        
                if len(tp) + len(fp) == 0:
                    precision = 0
                    recall = 0
                else:
                    precision = len(tp) / (len(tp) + len(fp))
                    recall = len(tp) / (len(tp) + len(fn))

                if precision + recall == 0:
                    f1_scores[conf_thresh_idx, nms_thresh_idx] = .0
                else:
                    f1_scores[conf_thresh_idx, nms_thresh_idx] = 2 * (precision * recall) / (precision + recall)

        # 파일 저장에 사용하기 위함
        if category_id == 1:
            class_name = "helmet_on"
        elif category_id == 2:
            class_name = "helmet_off"
        elif category_id == 3:
            class_name = "belt_on"
        elif category_id == 4:
            class_name = "belt_off"

        # xs = conf_thresh_list
        # ys = nms_thresh_list
        # zs = f1_scores
        
        xs, ys = np.meshgrid(nms_thresh_list, conf_thresh_list)  # indexing forced to use xy(ij)
        zs = f1_scores
        
        max_f1_flat_index = np.argmax(f1_scores.flatten())  # [99, 9] -> [891]
        max_f1_conf_idx = max_f1_flat_index // len(nms_thresh_list)
        max_f1_nms_idx = max_f1_flat_index % len(nms_thresh_list)
        
        max_f1_conf = conf_thresh_list[max_f1_conf_idx]
        max_f1_nms = nms_thresh_list[max_f1_nms_idx]
        max_f1 = f1_scores[max_f1_conf_idx, max_f1_nms_idx]

        # creating figure
        fig = plt.figure(figsize=(8, 6)) # 800x600 (4:3 ratio)
        ax = Axes3D(fig, auto_add_to_figure=False)
        # ax = fig.gca(projection='3d')

        ax.plot_surface(xs, ys, zs, rstride=1, cstride=1, cmap=cm.twilight_shifted, antialiased=True)
        # scatter_dot = ax.scatter(max_f1_nms, max_f1_conf, max_f1, color="red", label="Max F1 Score ({:.02f})".format(max_f1))
        p = Circle((max_f1_nms, max_f1_conf), 0.02, ec='none', fc="red", label="Max F1 Score (F1={:.02f}, NMS={:.01f}, Conf={:.02f})".format(max_f1, max_f1_nms, max_f1_conf))
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=max_f1, zdir="z")

        # setting title and labels
        # ax.set_title("{} F1 detection confidence threshold curve".format(class_name))
        ax.set_xlabel('NMS IoU threshold')
        ax.set_ylabel('Confidence threshold')
        ax.set_zlabel('F1 Score')

        # displaying the plot
        fig.add_axes(ax)
        # fig.set_label(' ')  # Not a 'Figure 1'
        fig.set_label("{} F1 detection confidence threshold curve".format(class_name))

        plt.legend(handles=[p], loc="upper right")
        plt.savefig(os.path.join(fig_savedir, "{}.svg".format(class_name)), dpi=600, transparent=True)
        
        import pickle
        with open(os.path.join(fig_savedir, "{}.pkl".format(class_name)), 'wb') as f:
            pickle.dump((nms_thresh_list, conf_thresh_list, f1_scores, class_name), f)
        
        all_f1_scores.append((f1_scores, max_f1_nms_idx, max_f1_nms))
        
    # Draw and save in single plot      
    # 이전 perclass_f1_scores로부터 max NMS에 해당하는 값을 가져온다.  
    
    # Max NMS 중에서 Average를 찾는다.
    avg_max_nms_thresh = np.mean([max_f1_nms for f1_scores_all_nms, max_f1_nms_idx, max_f1_nms in all_f1_scores])
    
    # nms_thresh_list안에서 가장 가까운 값으로
    avg_max_nms_thresh_idx = (np.abs(np.array(nms_thresh_list) - avg_max_nms_thresh)).argmin()
    avg_max_nms_thresh = nms_thresh_list[avg_max_nms_thresh_idx]
    
    plt.figure(figsize=(8, 4))
    plt.title("F1 Score by Detection Threshold (NMS Threshold: {:.01f})".format(avg_max_nms_thresh))
    plt.xlabel("Confidence Threshold")
    plt.ylabel("F1 Score")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xticks(np.arange(0, 1, step=0.1))
    plt.yticks(np.arange(0.1, 1, step=0.1))
    
    all_lines = []
    thresholds = []
    for class_id, (f1_scores_all_nms, max_f1_nms_idx, max_f1_nms) in enumerate(all_f1_scores):
        f1_scores = f1_scores_all_nms[:, avg_max_nms_thresh_idx]
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
        
        max_threshold = conf_thresh_list[np.argmax(f1_scores)]
        thresholds.append(max_threshold)
        
        f1_line, = plt.plot(conf_thresh_list, f1_scores, color=color, linestyle=linestyle, label="{}".format(class_name))
        all_lines.append(f1_line)
        
    avg_max_threshold = np.mean(thresholds)
    max_score_line = plt.axvline(avg_max_threshold, color='r', linestyle='dashdot', label="Average Maximum ({:.02f})".format(avg_max_threshold))
    
    l_helmet_on, l_helmet_off, l_belt_on, l_belt_off = all_lines  # 제일 결과 좋은순 (웬만하면 바뀔일은 없겠지)
    plt.legend(handles=[l_belt_off, l_helmet_on, l_belt_on, l_helmet_off, max_score_line], loc='upper left', prop={'size': 8})
    plt.savefig(os.path.join(fig_savedir, "all.svg"), dpi=600, transparent=True)
        
    logger.info("F1 plots are drawn into {}".format(file_name))


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
