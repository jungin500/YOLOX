#!/usr/bin/env python3
# -*- coding:utf-8 -*-

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


def make_parser():
    parser = argparse.ArgumentParser("COCO Annotation Viewer")
    parser.add_argument(
        "--shuffle",
        default=None,
        action='store_true',
        help="Shuffle dataset (intend not to preserve its order)"
    )
    parser.add_argument("--seed", default=None, type=int, help="Shuffle seed for displaying same image")
    parser.add_argument("--image-root", default="parent", type=str, help="Image root folder of COCO annotation (default: ../val2017 of coco_annotation file)")
    parser.add_argument(
        "coco_annotation",
        help="COCO annotation filename",
        default=None
    )
    return parser


def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        warnings.warn("You have chosen to seed testing.")

    if not os.path.exists(args.coco_annotation):
        logger.error("COCO Annotation path {} not found.".format(args.coco_annotation))
        sys.exit(1)

    if args.image_root == 'parent':
        image_root = os.path.join(str(Path(args.coco_annotation).parent.parent), 'val2017')
    else:
        image_root = args.image_root

    vis_window_title = 'Dataset visualization'
    cv2.namedWindow(vis_window_title)
    
    logger.info("Loading annotation ...")
    with open(args.coco_annotation, 'r') as f:
        annotation = json.load(f)
    logger.info("Loaded {} images and {} annootations".format(len(annotation['images']), len(annotation['annotations'])))

    logger.info("Mapping annotation ...")
    # Slowest implementation
    # bbox_map = { image['id']: list(filter(lambda item: item['image_id'] == image['id'], annotation['annotations'])) for image in tqdm(annotation['images'], "Creating image idmap for faster demo ...") }
    # Faster implementation
    bbox_map = { image['id']: [] for image in annotation['images'] }
    [bbox_map[item['image_id']].append(item) for item in annotation['annotations']]
    idmap = { category['id']: category['name'] for category in annotation['categories'] }
    colormap = {
        # helmet_on
        1: (0, 255, 0),
        # helmet_off
        2: (0, 0, 255),
        # belt_on
        3: (255, 0, 0),
        # belt_off
        4: (255, 255, 0),
    }

    if args.shuffle:
        logger.info("Shuffling dataset")
        random.shuffle(annotation['images'])

    logger.info("Done, beginning visualization ...")
    for image_info in annotation['images']:
        image_path = os.path.join(image_root, image_info['id'] + '.jpg')
        assert os.path.exists(image_path), "Image path \"{}\" not found.".format(image_path)

        image = cv2.imread(image_path)

        for item in bbox_map[image_info['id']]:
            xmin, ymin, width, height = item['bbox']
            xmax, ymax = xmin + width, ymin + height
            xmin, ymin, xmax, ymax = [int(i) for i in [xmin, ymin, xmax, ymax]]
            class_name = idmap[item['category_id']]

            cv2.rectangle(
                image,
                (xmin, ymin),
                (xmax, ymax),
                colormap[item['category_id']], 2
            )

            # y-clip
            if ymin - 14 < 14:
                ydraw = ymax + 27
            else:
                ydraw = ymin - 14
            
            # shadow effect (not for eye-candy!)
            cv2.putText(
                image,
                class_name,
                (xmin + 4, ydraw),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 0), 1
            )
            # real text
            cv2.putText(
                image,
                class_name,
                (xmin + 3, ydraw - 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, colormap[item['category_id']], 1
            )

        cv2.imshow(vis_window_title, image)
        key = cv2.waitKey(0)
        if key == 113:
            logger.info("Stopping viewer")
            break

if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
