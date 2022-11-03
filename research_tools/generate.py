#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Dataset Generator V1+V2+V3
#
# 각 Training Scale별 결과를 모아서 데이터를 구축한다.
#   Naiive, IoU strategy에서는 argument를 활용하며,
#   Scales strategy에서는 구축할 Scale이 args.scales에 정의되어 있다.
#

import argparse
from loguru import logger

import torch
import torch.backends.cudnn as cudnn
from yolox.core.launch import launch

from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info

from yolox.utils.dist import get_local_rank

from yolox.utils.setup_env import configure_module, configure_nccl

import warnings
import random
import sys

from generator import NaiiveGenerator, MultiscaleGenerator, IOUGenerator


def make_parser():
    parser = argparse.ArgumentParser("YOLOX COCO Annotation Generator")
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
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=0,
        help="number of gpus to generate dataset",
    )
    parser.add_argument("--conf", default=0.01, type=float, help="test conf")
    parser.add_argument("--nms", default=0.65, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--generator-conf", default=0.45, type=float, help="generator conf")
    parser.add_argument("--generator-iou", default=0.3, type=float, help="generator iou threshold")
    parser.add_argument("-b", "--batch-size", default=64, type=int, help="batch size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--devel",
        dest="devel",
        default=False,
        action="store_true",
        help="Enable fast-development environment",
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
        choices=["naiive", "iou", "scales"],
        help="Generation strategy"
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

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    if args.nms is not None:
        exp.nmsthre = args.nms
    
    logger.info("Exp:\n{}".format(exp))

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

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    # Generate annotation with given model
    if args.strategy == 'naiive':
        print("[naiive] Generating using Confidence {:.2f} and NMS-IoU {:.2f}".format(args.conf, args.nms))
        assert '--scales' not in ' '.join(sys.argv), "scales should not be applied on current strategy."
        generator = NaiiveGenerator(
            exp = exp,
            model = model,
            device = args.device,
            is_distributed = is_distributed,
            batch_size = args.batch_size,
            output_filename = args.output_path,
            half_precision = args.fp16
        )

    elif args.strategy == 'iou':
        print("[iou] Generating using Confidence {:.2f} and NMS-IoU {:.2f}".format(args.conf, args.nms))
        print("[scales] and Generating using Confidence {:.2f} and IoU Thresh {:.2f}".format(args.generator_conf, args.generator_iou))
        assert '--scales' not in ' '.join(sys.argv), "scales should not be applied on current strategy."
        generator = IOUGenerator(
            exp = exp,
            model = model,
            conf = args.generator_conf,
            iou_thresh = args.generator_iou,
            device = args.device,
            is_distributed = is_distributed,
            batch_size = args.batch_size,
            output_filename = args.output_path,
            half_precision = args.fp16
        )
        
    elif args.strategy == 'scales':
        assert args.tsize == None, "tsize cannot be applied on scale strategy. use scales argument."
        print("[scales] Inferring using Confidence {:.2f} and NMS-IoU {:.2f}".format(args.conf, args.nms))
        print("[scales] and Generating using Confidence {:.2f} and IoU Thresh {:.2f}".format(args.generator_conf, args.generator_iou))
        print("[scales] Scale list: {}".format(str(scales)))
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
            output_filename = args.output_path,
            half_precision = args.fp16
        )

    else:
        raise NotImplementedError("Strategy type {} not implemented".format(args.strategy))

    generator.init()
    generator.generate_dataset()

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

    print("Done generating annotations, exiting ...")