#!/bin/sh
# Patch pycocotools for supporting custom dataset (Non-digit naming convention)

PYCOCO_PATH=$(python3 -c 'import pycocotools; print(pycocotools.__path__[0])')
patch $PYCOCO_PATH/coco.py coco.py.patch