#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import os
from loguru import logger

import cv2

from pathlib import Path
import random
import warnings
import sys
import json
import glob
import pickle
import hashlib


def make_parser():
    parser = argparse.ArgumentParser("COCO Annotation Viewer")
    parser.add_argument(
        "--shuffle",
        default=None,
        action='store_true',
        help=
        "Shuffle dataset (intend not to preserve its order and/or comparing multiple datasets)"
    )
    parser.add_argument("--small",
                        default=None,
                        action='store_true',
                        help="Small visualization window (640x360)")
    parser.add_argument("--seed",
                        default=None,
                        type=int,
                        help="Shuffle seed for displaying same image")
    parser.add_argument("--save",
                        default=False,
                        action='store_true',
                        help="Save result image")
    parser.add_argument("--save-path", default="./", help="Save path")
    parser.add_argument("--image-title", default=None, help="Image title")
    parser.add_argument(
        "--image-root",
        default="parent",
        type=str,
        help=
        "Image root folder of COCO annotation (default: ../val2017 of coco_annotation file)"
    )
    parser.add_argument("coco_annotation",
                        help="COCO annotation filename",
                        default=None)
    parser.add_argument("ids",
                        default=None,
                        help="exclusive image id name to display",
                        nargs=argparse.REMAINDER)
    return parser


def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        warnings.warn("You have chosen to seed testing.")

    if not os.path.exists(args.coco_annotation):
        logger.error("COCO Annotation path {} not found.".format(
            args.coco_annotation))
        sys.exit(1)

    if args.image_root == 'parent':
        image_root = os.path.join(
            str(Path(args.coco_annotation).parent.parent), 'val2017')
    else:
        image_root = args.image_root

    if args.save:
        assert os.path.exists(
            args.save_path), "Save path {} not found!".format(args.save_path)
    vis_window_title = 'Dataset visualization'
    cv2.namedWindow(
        vis_window_title,
        cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)

    logger.info("Loading annotation ...")
    with open(args.coco_annotation, 'r') as f:
        annotation = json.load(f)
    logger.info("Loaded {} images and {} annootations".format(
        len(annotation['images']), len(annotation['annotations'])))

    # Normalize image root string for proper cache filename hashing
    image_root = image_root.strip()
    if image_root[-1] == os.sep:
        image_root = image_root[:-1]

    cache_filename = 'imagelist.{}.cache'.format(
        hashlib.sha256(image_root.encode('utf-8')).hexdigest()[:8])
    if not os.path.exists(cache_filename):
        logger.warning(
            "Image list cache file of path \"{}\" does not exist.".format(
                image_root))
        logger.warning("Traversing filesystem for full image list ...")
        image_list = glob.glob(os.path.join(image_root, '*.jpg'))
        logger.info("Filtering out _D.jpg files ...")
        image_list = list(
            sorted(
                filter(
                    lambda filename: not filename.lower().endswith('_d.jpg'),
                    image_list)))

        image_map = {Path(path).stem: path for path in image_list}
        with open(cache_filename, 'wb') as f:
            pickle.dump(image_map, f)

        logger.info("Image map cached")
        logger.info("- Images Path: {}".format(image_root))
        logger.info("- Cache filename: {}".format(cache_filename))
        logger.info(
            "Cache is dependent on image path so it's safe to re-run when image path is changed"
        )
    else:
        logger.info("Loading image list cache {} ...".format(cache_filename))
        with open(cache_filename, 'rb') as f:
            image_map = pickle.load(f)
        image_list = [image_map[key] for key in sorted(image_map.keys())]
    # 어찌되었든 image_list는 sort가 보장된 상태임
    # -> shuffle의 입력으로는 무조건 같은 list가 들어갈 것임

    if args.ids:
        logger.warning("Using image id list for displaying")
        # Sanity check
        for image_id in args.ids:
            assert image_id in image_map, "Image {} not in image list!".format(
                image_id)
        image_map = {image_id: image_map[image_id] for image_id in args.ids}
        image_list = [image_map[key] for key in sorted(image_map.keys())]

    if args.shuffle:
        if args.ids:
            logger.warning("Disabling shuffling while displaying with ids")
        else:
            logger.info("Shuffling items ...")
            random.shuffle(image_list)

    # 정상적인 image_list를 골랐는지 확인 (trainval, test 헷갈림)
    fs_image_id_set = set([Path(image_path).stem for image_path in image_list])
    json_image_id_set = set([item['id'] for item in annotation['images']])
    if len(fs_image_id_set - json_image_id_set) == len(fs_image_id_set):
        # 겹치는 구간이 없음
        logger.error(
            "Filesystem has {} image set and JSON annotation has {} image set!"
            .format(len(fs_image_id_set), len(json_image_id_set)))
        logger.error("No occurance detected, maybe wrong dataset?")
        sys.exit(1)

    logger.info("Mapping annotation ...")
    # Slowest implementation
    # bbox_map = { image['id']: list(filter(lambda item: item['image_id'] == image['id'], annotation['annotations'])) for image in tqdm(annotation['images'], "Creating image idmap for faster demo ...") }
    # Faster implementation
    annotation_map = {}
    for item in annotation['images']:
        annotation_map[item['id']] = item
    bbox_map = {image['id']: [] for image in annotation['images']}
    [
        bbox_map[item['image_id']].append(item)
        for item in annotation['annotations']
    ]
    idmap = {
        category['id']: category['name']
        for category in annotation['categories']
    }
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

    logger.info("Done, beginning visualization ...")
    display_idx = 0
    for image_path in image_list:
        # assert os.path.exists(image_path), "Image path \"{}\" not found.".format(image_path)
        image_id = Path(image_path).stem
        image = cv2.imread(image_path)

        # Box Padding
        px, py = 10, 20
        if args.image_title:
            # Display title
            (txt_width,
             txt_middle_pos), _ = cv2.getTextSize(args.image_title,
                                                  cv2.FONT_HERSHEY_SIMPLEX,
                                                  0.7, 2)
            cv2.rectangle(image, (0, 0),
                          (txt_width + px * 2, txt_middle_pos + py * 2),
                          (0, 0, 0), -1)
            cv2.putText(image, args.image_title, (px, py + txt_middle_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Display image name
            prev_height = txt_middle_pos + py * 2
            (txt_width,
             txt_middle_pos), _ = cv2.getTextSize(image_id,
                                                  cv2.FONT_HERSHEY_SIMPLEX,
                                                  0.7, 2)
            cv2.rectangle(
                image, (0, prev_height),
                (txt_width + px * 2, prev_height + txt_middle_pos + py * 2),
                (0, 0, 0), -1)
            cv2.putText(image, image_id,
                        (px, prev_height + py + txt_middle_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            # Display image name
            (txt_width,
             txt_middle_pos), _ = cv2.getTextSize(image_id,
                                                  cv2.FONT_HERSHEY_SIMPLEX,
                                                  1.5, 2)
            cv2.rectangle(image, (0, 0),
                          (txt_width + px * 2, txt_middle_pos + py * 2),
                          (0, 0, 0), -1)
            cv2.putText(image, image_id, (px, py + txt_middle_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

        # Extra information string
        extras = ''
        if image_id in bbox_map:

            for item in bbox_map[image_id]:
                xmin, ymin, width, height = item['bbox']
                xmax, ymax = xmin + width, ymin + height
                xmin, ymin, xmax, ymax = [
                    int(i) for i in [xmin, ymin, xmax, ymax]
                ]
                class_name = idmap[item['category_id']]
                confidence = None
                if 'det_confidence' in item:
                    confidence = item['det_confidence']

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax),
                              colormap[item['category_id']], 2)

                # y-clip
                if ymin - 14 < 14:
                    ydraw = ymax + 27
                else:
                    ydraw = ymin - 14

                # shadow effect (not for eye-candy!)
                if confidence:
                    class_name_text = '{} ({:.01f}%)'.format(
                        class_name, confidence * 100)
                else:
                    class_name_text = class_name
                cv2.putText(image, class_name_text, (xmin + 4, ydraw),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
                # real text
                cv2.putText(image, class_name_text, (xmin + 3, ydraw - 1),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            colormap[item['category_id']], 2)
        else:
            extras = '(No annotation)'

        if args.save:
            cv2.imwrite(os.path.join(args.save_path, image_id + ".jpg"), image)
        else:
            if args.small:
                # Ratio-perserve resizing
                r_image = image.shape[0] / image.shape[1]  # H / W
                window_size = (1280, int(1280 * r_image))
                logger.debug("[{:04d}] Displaying: {} {}".format(
                    display_idx, image_id, extras))
                cv2.imshow(vis_window_title, cv2.resize(image, window_size))
                cv2.resizeWindow(vis_window_title, window_size)
            else:
                cv2.imshow(vis_window_title, image)

            key = cv2.waitKey(0)
            if key == 113:
                logger.info("Stopping viewer")
                break

        display_idx += 1


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
