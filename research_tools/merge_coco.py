#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Merge multiple coco annotation into single file
# Fold 결과로 나온 annotation을 합치기 위함
#

import argparse
import os
from loguru import logger
from tqdm.auto import tqdm

import cv2

from pathlib import Path
import random
import warnings
import sys
import json

from util.merge_coco import merge_coco

def make_parser():
    parser = argparse.ArgumentParser("COCO Multiple annotation merger")
    parser.add_argument(
        "-o", "--output",
        dest="output",
        type=str,
        required=True,
        help="Output annotation filename",
    )
    parser.add_argument(
        "multiple_annotations",
        help="COCO annotation items",
        default=None,
        nargs=argparse.REMAINDER
    )
    return parser


def main(args):
    # Sanity check if file exists
    for filename in args.multiple_annotations:
        if not os.path.exists(filename):
            raise FileNotFoundError("File {} not found!".format(filename))

    # Read all annotations
    all_annotations = []
    for filename in args.multiple_annotations:
        logger.info("Reading annotation {} ...".format(filename))
        with open(filename, 'r') as f:
            annotation = json.load(f)
        all_annotations.append(annotation)
        
    # Sanity check
    assert len(all_annotations) > 0, "Empty annotation?"
    for idx, annotation in enumerate(all_annotations):
        json_filename = args.multiple_annotations[idx]
        assert len(annotation["images"]) > 0, "Empty images in file {}".format(json_filename)
        assert len(annotation["annotations"]) > 0, "Empty annotations in file {}".format(json_filename)
    assert len(set([json.dumps(annotation["categories"]) for annotation in all_annotations])) == 1, \
        "Different categories between annotations!"

    # Merge annotations
    result_annotation = merge_coco(all_annotations)

    # Save result annotations
    logger.info("Writing annotation {} ...".format(args.output))
    with open(args.output, 'w') as f:
        json.dump(result_annotation, f)
    logger.info("Done")

if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
