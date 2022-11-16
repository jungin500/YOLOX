#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Dataset Generator V1+V2+V3
#
# 각 Training Scale별 결과를 모아서 데이터를 구축한다.
#   Naiive, IoU strategy에서는 argument를 활용하며,
#   Scales strategy에서는 구축할 Scale이 args.scales에 정의되어 있다.
# 
# -i, -ids를 이용하면 주어진 ID만  이미지를 생성한 다음 X11 화면으로 보여준다.
# demo_ds_coco에 있는 visualization 기능을 활용한다.
#

import argparse
from loguru import logger
import cv2
import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from yolox.core.launch import launch

from yolox.exp import get_exp
from yolox.utils import get_model_info

from yolox.utils.dist import get_local_rank, get_local_size
from yolox.utils.setup_env import configure_module, configure_nccl

import warnings
import random
import sys

from generator import NaiiveGenerator, NaiiveAdvancedGenerator, MultiscaleGenerator, IOUGenerator, SimpleMultiscaleGenerator
import tempfile
import json
import os
import numpy as np
from matplotlib import pyplot as plt

from util import merge_coco, eval_coco

def make_parser():
    parser = argparse.ArgumentParser("YOLOX COCO Annotation Generator", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-o", "--output", dest="output_path", default="annotations_generated.json", help="COCO-type Annotation output JSON filename")
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, required=True, type=str, help="ckpt for inference")
    parser.add_argument(
        "--gpus",
        type=int,
        default=0,
        help="number of gpus to generate dataset",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("-b", "--batch-size", default=64, type=int, help="batch size")
    parser.add_argument("--conf", default=0.01, type=float, help="test conf")
    parser.add_argument("--nms", default=0.65, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--generator-conf", default=0.45, type=float, help="generator conf")
    parser.add_argument("--generator-iou", default=0.3, type=float, help="generator iou threshold")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--rematch-thresh",
        dest="rematch_thresh",
        default=2,
        type=int,
        help="Rematch threshold"
    )
    parser.add_argument(
        "--strategy",
        dest="strategy",
        default="naiive",
        type=str,
        choices=["naiive", "naiive-advanced", "iou", "scales", "scales-simple"],
        help="Generation strategy"
    ),
    parser.add_argument(
        "--perclass-conf-ious",
        dest="perclass_conf_ious",
        default="0.01,0.65,0.01,0.65,0.01,0.65,0.01,0.65",
        type=str,
        help="Per-class Conf and NMS IoUs (delimited by comma)"
    )
    parser.add_argument(
        "--scales",
        dest="scales",
        default="640,800,960,1120",
        type=str,
        help="Scales (delimited by comma)"
    )
    parser.add_argument("--seed", default=None, type=int, help="Evaluation seed")
    parser.add_argument(
        "-i",
        "--id",
        dest="image_ids",
        help="Input image ids to generate (separated by comma)",
        default=None,
    )
    parser.add_argument(
        "--eval",
        dest="eval",
        default=False,
        action="store_true",
        help="Evaluate after saving JSON annotations",
    )
    parser.add_argument(
        "--conf-eval",
        dest="conf_eval",
        default=False,
        action="store_true",
        help="Evaluate by conf search space",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def main(exp, args, num_gpu):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn(
            "You have chosen to seed testing. This will turn on the CUDNN deterministic setting, "
        )
    else:
        cudnn.benchmark = True
        print("You're getting ready for the paper! be sure to set seed argument.")
        sys.exit(0)
        
    try:
        scales = [int(i) for i in args.scales.replace(' ','').split(',')]
    except Exception as e:
        print("Error: invalid scale list \"args.scales\":", str(e))
        sys.exit(1)

    is_distributed = num_gpu > 1
    configure_nccl()
    rank = get_local_rank()
    world_size = get_local_size()

    if args.eval or args.conf_eval or args.image_ids:
        file_name = os.path.join(exp.output_dir, exp.exp_name)
        fig_savedir = os.path.join(file_name, datetime.datetime.now().strftime('%y%m%d_%H%M%S'))

        if rank == 0:
            os.makedirs(file_name, exist_ok=True)
            os.makedirs(fig_savedir, exist_ok=True)

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.seed is not None:
        exp.seed = args.seed

    logger.info("Exp:\n{}".format(exp))
    logger.info("Args (Nonused parameters preserved): {}".format(args))

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    torch.cuda.set_device(rank)
    if args.device == "gpu":
        model.cuda(rank)
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    ckpt_file = args.ckpt
    assert ckpt_file is not None
    logger.info("loading checkpoint")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")
    
    oneshot_image_ids = None
    if args.image_ids:
        logger.info("Forcing data_num_workers to 0")
        exp.data_num_workers = 0
    
        assert args.batch_size == 1, "Batch size must be 1 to generate with image_ids."
        oneshot_image_ids = args.image_ids.split(',')
        
        # 혹시나 X11 문제를 나중에 알면 실험이 길어지니까
        vis_window_title = 'Dataset visualization'
        cv2.namedWindow(vis_window_title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)

    if args.conf_eval:
        # Evaluation할 때는 0.01 ~ 0.99까지 전부 사용
        conf_thresh_space = np.linspace(0.1, 0.9, num=10 - 1)
    else:
        # 이 외에는 하나만 Evaluate
        conf_thresh_space = [exp.test_conf]
    
    # 평가 결과 저장
    all_ap5095 = []
    all_ap50 = []
    all_ap_s = []
    all_ap_m = []
    all_ap_l = []
    all_AP_perclass = []
    all_AR_perclass = []
    all_AP50_perclass = []

    for conf_thresh_idx, conf_thresh in enumerate(conf_thresh_space):
        logger.info("Running confidence threshold {} ({}/{})".format(conf_thresh, conf_thresh_idx + 1, len(conf_thresh_space)))
        # Target test confidence
        exp.test_conf = conf_thresh
        args.conf = conf_thresh

        reduce_master_rank = 0
        coco_result = generate(
            exp = exp,
            model = model,
            args = args,
            is_distributed = is_distributed,
            rank = rank,
            world_size = world_size,
            oneshot_image_ids = oneshot_image_ids,
            scales = scales,
            reduce_master_rank = reduce_master_rank
        )

        if rank != reduce_master_rank:
            return  # exit application (wait for master rank to exit)

        if args.image_ids:
            # 화면에 표출 및 저장
            visualize(
                exp = exp,
                args = args,
                coco_result = coco_result,
                oneshot_image_ids = oneshot_image_ids,
                save_dir = fig_savedir,
                image_title = "{}, conf={} nms={} g_conf={} g_iou={} rm_thresh={}".format(
                    args.strategy,
                    args.conf,
                    args.nms,
                    args.generator_conf,
                    args.generator_iou,
                    args.rematch_thresh
                )
            )
            logger.info("Result images are stored into {}".format(fig_savedir))
        else:
            # 결과 저장 (Distributed인 경우 Rank 0만 수행)
            save_result(
                output_path = args.output_path,
                coco_result = coco_result
            )

            if args.conf_eval or args.eval:
                ap5095, ap50, ap_s, ap_m, ap_l, AP_perclass, AR_perclass, AP50_perclass = eval_coco(
                    trainval_json = os.path.join(exp.data_dir, 'annotations', exp.train_ann),
                    generated_json = args.output_path
                )

                all_ap5095.append(ap5095)
                all_ap50.append(ap50)
                all_ap_s.append(ap_s)
                all_ap_m.append(ap_m)
                all_ap_l.append(ap_l)
                all_AP_perclass.append(AP_perclass)
                all_AR_perclass.append(AR_perclass)
                all_AP50_perclass.append(AP50_perclass)
    
    if args.conf_eval or args.eval:
        plot_result(fig_savedir, conf_thresh_space, all_ap5095, all_ap50, all_ap_s, all_ap_m, all_ap_l, all_AP_perclass, all_AR_perclass, all_AP50_perclass)
    logger.info("Done generating annotations, exiting ...")


def plot_result(fig_savedir, conf_thresh_space, all_ap5095, all_ap50, all_ap_s, all_ap_m, all_ap_l, all_AP_perclass, all_AR_perclass, all_AP50_perclass):
    # 2nd Index 접근을 위한 변환
    all_ap5095 = np.array(all_ap5095)
    all_ap50 = np.array(all_ap50)
    all_ap_s = np.array(all_ap_s)
    all_ap_m = np.array(all_ap_m)
    all_ap_l = np.array(all_ap_l)
    all_AP_perclass = np.array(all_AP_perclass)
    all_AR_perclass = np.array(all_AR_perclass)
    all_AP50_perclass = np.array(all_AP50_perclass)

    # 결과를 가지고 Plot을 진행한다.
    CLASSES_COLOR_MAP = {
        "helmet_on": "#6EF018",
        "helmet_off": "#18A5F0",
        "belt_on": "#E718F0",
        "belt_off": "#F08C0C"
    }

    # AP50_95 Per-class and All-class
    plt.figure(figsize=(8, 4))
    plt.title("AP5095 Score by Detection Threshold (NMS={:.02f})".format(exp.nmsthre))
    
    lines = []
    thresholds = []
    for idx, class_name in enumerate(CLASSES_COLOR_MAP.keys()):
        color = CLASSES_COLOR_MAP[class_name]

        max_threshold = conf_thresh_space[np.argmax(all_AP_perclass[:, idx])]
        thresholds.append(max_threshold)

        ap_line, = plt.plot(conf_thresh_space, all_AP_perclass[:, idx], color=color, linestyle='dashed', label=class_name)
        lines.append(ap_line)

    all_ap_line, = plt.plot(conf_thresh_space, all_ap5095, color=color, linestyle='dashed', label=class_name)
    avg_max_threshold = np.mean(thresholds)
    max_score_line = plt.axvline(avg_max_threshold, color='r', linestyle='dashdot', label="Max score ({:.02f})".format(avg_max_threshold))
    plt.legend(handles=[*lines, all_ap_line, max_score_line], loc='upper right')
    plt.xlabel("Confidence Threshold")
    plt.ylabel("AP5095 Score")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xticks(np.arange(0, 1, step=0.1))
    plt.yticks(np.arange(0.1, 1, step=0.1))
    plt.savefig(os.path.join(fig_savedir, "ap5095.svg"), dpi=600, transparent=True)

    # AP50 Per-class and All-class
    plt.figure(figsize=(8, 4))
    plt.title("AP50 Score by Detection Threshold (NMS={:.02f})".format(exp.nmsthre))
    
    lines = []
    thresholds = []
    for idx, class_name in enumerate(CLASSES_COLOR_MAP.keys()):
        color = CLASSES_COLOR_MAP[class_name]

        max_threshold = conf_thresh_space[np.argmax(all_AP50_perclass[:, idx])]
        thresholds.append(max_threshold)

        ap_line, = plt.plot(conf_thresh_space, all_AP50_perclass[:, idx], color=color, linestyle='dashed', label=class_name)
        lines.append(ap_line)

    all_ap_line, = plt.plot(conf_thresh_space, all_ap50, color=color, linestyle='dashed', label=class_name)
    avg_max_threshold = np.mean(thresholds)
    max_score_line = plt.axvline(avg_max_threshold, color='r', linestyle='dashdot', label="Max score ({:.02f})".format(avg_max_threshold))
    plt.legend(handles=[*lines, all_ap_line, max_score_line], loc='upper right')
    plt.xlabel("Confidence Threshold")
    plt.ylabel("AP50 Score")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xticks(np.arange(0, 1, step=0.1))
    plt.yticks(np.arange(0.1, 1, step=0.1))
    plt.savefig(os.path.join(fig_savedir, "ap50.svg"), dpi=600, transparent=True)


def generate(exp, model, args, is_distributed, rank, world_size, oneshot_image_ids, scales, reduce_master_rank = 0):
    generator = build_generator(
        exp = exp,
        model = model,
        args = args,
        is_distributed = is_distributed,
        oneshot_image_ids = oneshot_image_ids,
        scales = scales
    )

    generator.init()
    coco_result = generator.generate_dataset()

    if is_distributed:
        # Rank 0: coco_result, anoher ranks: None
        coco_result = reduce_coco_result(
            rank = rank,
            output_path = args.output_path,
            world_size = world_size,
            coco_result = coco_result,
            to_rank = reduce_master_rank
        )

    return coco_result
    

def build_generator(exp, model, args, is_distributed, oneshot_image_ids, scales):
    # Generate annotation with given model
    if args.strategy == 'naiive':
        print("[naiive] Generating using Confidence {:.2f} and NMS-IoU {:.2f}".format(args.conf, args.nms))
        assert '--scales' not in ' '.join(sys.argv), "scales should not be applied on current strategy."
        assert '--generator-conf' not in ' '.join(sys.argv), "generator confidence should not be applied on current strategy."
        assert '--generator-iou' not in ' '.join(sys.argv), "generator iou should not be applied on current strategy."
        generator = NaiiveGenerator(
            exp = exp,
            model = model,
            device = args.device,
            is_distributed = is_distributed,
            batch_size = args.batch_size,
            half_precision = args.fp16,
            oneshot_image_ids = oneshot_image_ids
        )
        
    elif args.strategy == 'naiive-advanced':
        perclass_conf_ious = [float(v) for v in args.perclass_conf_ious.split(',')]
        assert len(perclass_conf_ious) % 2 == 0, "--perclass-conf-ious only accepts a pair of conf_thresh, iou_thresh."
        perclass_conf_ious = list(zip(perclass_conf_ious[::2], perclass_conf_ious[1::2]))
        assert len(perclass_conf_ious) == exp.num_classes, "Provided perclass_conf_ious doesn't match count of classes"
        
        print("[naiive-advanced] Generating using following Confidence, NMS-IoU parameters")
        for class_id, (conf_thresh, iou_thresh) in enumerate(perclass_conf_ious):
            print("[naiive-advanced] Class {}: conf_thresh {:.02f} iou_thresh {:.02f}".format(class_id, conf_thresh, iou_thresh))
        
        assert '--nms' not in ' '.join(sys.argv), "--nms, --conf should be ignored."
        assert '--conf' not in ' '.join(sys.argv), "--nms, --conf should be ignored."
        assert '--scales' not in ' '.join(sys.argv), "scales should not be applied on current strategy."
        assert '--generator-conf' not in ' '.join(sys.argv), "generator confidence should not be applied on current strategy."
        assert '--generator-iou' not in ' '.join(sys.argv), "generator iou should not be applied on current strategy."
        
        generator = NaiiveAdvancedGenerator(
            exp = exp,
            model = model,
            device = args.device,
            is_distributed = is_distributed,
            batch_size = args.batch_size,
            half_precision = args.fp16,
            perclass_conf_ious = perclass_conf_ious,
            oneshot_image_ids = oneshot_image_ids
        )

    elif args.strategy == 'iou':
        print("[iou] Generating using Confidence {:.2f} and NMS-IoU {:.2f}".format(args.conf, args.nms))
        print("[iou] and Generating using Confidence {:.2f} and IoU Thresh {:.2f}".format(args.generator_conf, args.generator_iou))
        assert '--scales' not in ' '.join(sys.argv), "scales should not be applied on current strategy."
        assert args.eval != True, "IOUStrategy depends on original dataset"
        generator = IOUGenerator(
            exp = exp,
            model = model,
            conf = args.generator_conf,
            iou_thresh = args.generator_iou,
            device = args.device,
            is_distributed = is_distributed,
            batch_size = args.batch_size,
            half_precision = args.fp16,
            oneshot_image_ids = oneshot_image_ids
        )
        
    elif args.strategy == 'scales':
        assert args.tsize == None, "tsize cannot be applied on scale strategy. use scales argument."
        print("[scales] Inferring using Confidence {:.2f} and NMS-IoU {:.2f}".format(args.conf, args.nms))
        print("[scales] and Generating using Confidence {:.2f} and IoU Thresh {:.2f}".format(args.generator_conf, args.generator_iou))
        print("[scales] Scale list: {}".format(str(scales)))
        assert args.eval != True, "MultiscaleGenerator depends on original dataset"
        generator = MultiscaleGenerator(
            exp = exp,
            model = model,
            scales = scales,
            conf = args.generator_conf,
            iou_thresh = args.generator_iou,
            rematch_thresh = args.rematch_thresh,
            device = args.device,
            is_distributed = is_distributed,
            batch_size = args.batch_size,
            half_precision = args.fp16,
            oneshot_image_ids = oneshot_image_ids
        )
        
    elif args.strategy == 'scales-simple':
        assert args.tsize == None, "tsize cannot be applied on scale strategy. use scales argument."
        assert '--generator-conf' not in ' '.join(sys.argv), "generator confidence should not be applied on current strategy."
        assert '--generator-iou' not in ' '.join(sys.argv), "generator iou should not be applied on current strategy."
        print("[scales-simple] Inferring using Confidence {:.2f} and NMS-IoU {:.2f}".format(args.conf, args.nms))
        print("[scales-simple] Scale list: {}".format(str(scales)))
        generator = SimpleMultiscaleGenerator(
            exp = exp,
            model = model,
            scales = scales,
            conf = args.generator_conf,
            device = args.device,
            is_distributed = is_distributed,
            batch_size = args.batch_size,
            half_precision = args.fp16,
            oneshot_image_ids = oneshot_image_ids
        )

    else:
        raise NotImplementedError("Strategy type {} not implemented".format(args.strategy))

    return generator


def visualize(exp, args, coco_result, oneshot_image_ids, save_dir = None, image_title = None):
    # 결과 표출 (임시 JSON 파일 활용)
    fp, fileloc = tempfile.mkstemp(suffix=".json")
    with open(fp, 'w') as f:
        json.dump(coco_result, f)
    
    extra_args = []
    if save_dir is not None:
        extra_args.extend(['--save', '--save-path', save_dir])
    if image_title is not None:
        extra_args.extend(['--image-title', image_title])
    
    # demo_ds_coco 실행
    from demo_ds_coco import make_parser as demo_make_parser
    from demo_ds_coco import main as demo_ds_coco
    demo_ds_args = demo_make_parser().parse_args([
        '--seed', str(args.seed),
        '--image-root', os.path.join(exp.data_dir, 'train2017'),
        *extra_args,
        fileloc,
        *oneshot_image_ids,
    ])
    logger.info("Beginning demo display")
    demo_ds_coco(demo_ds_args)


def save_result(output_path, coco_result):
    with open(output_path, 'w') as f:
        json.dump(coco_result, f)
    logger.info("File saved to {}".format(output_path))


def reduce_coco_result(rank, world_size, output_path, coco_result, to_rank=0):
    # Save each rank into separate file
    with open(output_path + '.r{}.tmp'.format(rank), 'w') as f:
        json.dump(coco_result, f)

    # Merge all rank's result into single json file
    dist.barrier()  # Make sure all rank wrote json file

    if rank == to_rank:
        # Do tasks on master rank
        all_annotations = []
        for another_rank_id in range(world_size):
            with open(output_path + '.r{}.tmp'.format(another_rank_id)) as f:
                all_annotations.append(json.load(f))
            
        for another_rank_id in range(world_size):
            os.unlink(output_path + '.r{}.tmp'.format(another_rank_id))
        
        # Sanity check
        assert len(all_annotations) > 0, "Empty annotation?"
        for idx, annotation in enumerate(all_annotations):
            json_filename = output_path + '.r{}.tmp'.format(idx)
            assert len(annotation["images"]) > 0, "Empty images in file {}".format(json_filename)
            assert len(annotation["annotations"]) > 0, "Empty annotations in file {}".format(json_filename)
        assert len(set([json.dumps(annotation["categories"]) for annotation in all_annotations])) == 1, \
            "Different categories between annotations!"
        
        # Merge annotations
        return merge_coco(all_annotations)
    else:
        return None


if __name__ == "__main__":
    configure_module()
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if args.device == 'gpu':
        num_gpu = torch.cuda.device_count() if args.gpus is None else args.gpus
        assert num_gpu <= torch.cuda.device_count()

        launch(
            main,
            num_gpu,
            num_machines=1,
            machine_rank=0,
            backend='nccl',
            dist_url='auto',
            args=(exp, args, num_gpu)
        )
    else:
        main(exp, args)