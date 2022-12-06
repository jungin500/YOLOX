#!/bin/bash
# Prepare environment for training
# for K8s training server setup
# for ROOT user execution
#
# Version 1.0-root
# CVMIPLab, Jung-In An <jungin500@kangwon.ac.kr>

# Global configurations
# will be read from environment exports
# REPOSITORY_URL
# COMMIT_ID
# ARTIFACT_ROOT
# JPEGIMAGES_PATH
# JPEGIMAEGS_PART_TAR_ROOT
# COCO_TRAINVAL_ANNOTATION
# COCO_TEST_ANNOTATION
# NO_COPY_DATASET
# WANDB_API_KEY

# Execution check
if [[ "$(basename -- "$0")" != "prepare.sh" ]]; then
    >&2 echo "Do not source! instread run current file."
    exit 1
fi

# Check UID first (DO not run this is runuser!)
if [ "$UID" -ne "0" ]; then
    >&2 echo "User is not root!"
    exit 1
fi

if [ -z "${WANDB_API_KEY}" ]; then
    echo "Wandb disabled"
fi

# Check path and file for sanity check
if [ ! -d "${JPEGIMAGES_PATH}" ]; then
    >&2 echo "FATAL: JPEGIMAGES_PATH directory ${JPEGIMAGES_PATH} not found."
    exit 1
elif [ ! -d "${JPEGIMAEGS_PART_TAR_ROOT}" ]; then
    >&2 echo "FATAL: JPEGIMAEGS_PART_TAR_ROOT directory ${JPEGIMAEGS_PART_TAR_ROOT} not found."
    exit 1
elif [ ! -f "${COCO_TRAINVAL_ANNOTATION}" ]; then
    >&2 echo "FATAL: COCO_TRAINVAL_ANNOTATION directory ${COCO_TRAINVAL_ANNOTATION} not found."
    exit 1
elif [ ! -f "${COCO_TEST_ANNOTATION}" ]; then
    >&2 echo "FATAL: COCO_TEST_ANNOTATION directory ${COCO_TEST_ANNOTATION} not found."
    exit 1
elif [ ! -d "${ARTIFACT_ROOT}" ]; then
    >&2 echo "FATAL: ARTIFACT_ROOT directory ${ARTIFACT_ROOT} not found."
    exit 1
fi

# Useful for debugging
# and Fail if some command does not run well
set -ex

# Install dependencies
apt-get -qq update
apt-get -qq install -y pv

# Setup timezone for folder naming, wandb, etc
ln -fs /usr/share/zoneinfo/Asia/Seoul /etc/localtime
DEBIAN_FRONTEND=noninteractive apt-get -qq -y install tzdata

# export Artifact directory for later use (after this script)
ARTIFACT_DIR=${ARTIFACT_ROOT}/$(hostname)_$(date +%Y%m%d_%H%M)
mkdir ${ARTIFACT_DIR}

# Clone repository
git clone ${REPOSITORY_URL} YOLOX

# Note that we changed directory
cd YOLOX
git checkout ${COMMIT_ID}

# Setup repository
mkdir YOLOX_outputs
ln -s ${ARTIFACT_DIR} YOLOX_outputs/yolox_l_aihub_cs_coco

# Bugfix of nvcr.io/nvidia/pytorch:22.02-py3
pip3 uninstall -y numpy opencv-python opencv-python-headless
rm -rf /opt/conda/lib/python3.8/site-packages/cv2

sed -i "s/^opencv_python$/opencv-python-headless/g" requirements.txt
pip3 install -r requirements.txt --progress-bar off
pip3 install -v -e . --progress-bar off

if [ ! -z "${WANDB_API_KEY}" ]; then
    # Install wandb
    pip3 install wandb
    wandb login ${WANDB_API_KEY}
fi

# Copy dataset (JPEGImages only, for compat reason. ALSO with COCO type dataset)
# Dynamic copy - will copy if local storage has some capacity, or just link.
mkdir datasets/construction-safety

if [ "${NO_COPY_DATASET}" -eq "1" ]; then
    echo "Linking dataset folder as requested"
    ln -s ${JPEGIMAGES_PATH} datasets/construction-safety/JPEGImages
else
    DATASET_REQUIRED_BYTES=587202560
    REMAINING_BYTES=$(df / | tail -n 1 | awk '{ print $4 }')
    if [ $DATASET_REQUIRED_BYTES -gt $REMAINING_BYTES ]; then
        echo "Insufficient filesystem space left! required 590GB+ for copying dataset into local storage"
        ln -s ${JPEGIMAGES_PATH} datasets/construction-safety/JPEGImages
    else
        echo "Checked storage - Will copy data into current storage"
        mkdir -p datasets/construction-safety/JPEGImages
        if [ -t 1 ]; then
            echo "Running inside TTY container, attach to current container to continue!"
            echo -e '\a'
            seq 1 4 | xargs -P 4 -I {} sh -c "pv -c -N JPEGImages.part{}.tar ${JPEGIMAEGS_PART_TAR_ROOT}/JPEGImages.part{}.tar | tar xf - --strip-components=1 -C datasets/construction-safety/JPEGImages"
        else
            echo "TTY unavailable, continuing normal copy. Can take up to 20mins (10G) or 40+mins(1G)."
            seq 1 4 | xargs -P 4 -I {} sh -c "tar xf ${JPEGIMAEGS_PART_TAR_ROOT}/JPEGImages.part{}.tar --strip-components=1 -C datasets/construction-safety/JPEGImages"
        fi
    fi
fi

# Link dataset folder for COCO
mkdir -p datasets/construction-safety-coco/annotations
ln -s ../construction-safety/JPEGImages datasets/construction-safety-coco/train2017
ln -s ../construction-safety/JPEGImages datasets/construction-safety-coco/val2017
ln -s ../construction-safety/JPEGImages datasets/construction-safety-coco/test2017

# Copy COCO Annotation
cp ${COCO_TRAINVAL_ANNOTATION} datasets/construction-safety-coco/annotations/
cp ${COCO_TEST_ANNOTATION} datasets/construction-safety-coco/annotations/

# Patch pycocotools for utilizing non-numeric filenames (image ids)
bash -c 'cd exps/jungin500/pycocotools-patch; bash patch.sh'

# Export variables for parent shell
# parent shell MUST source .trainrc after this script
cd /workspace
echo "export ARTIFACT_DIR=${ARTIFACT_DIR}" >> .trainrc
echo "export COMMIT_ID=${COMMIT_ID}" >> .trainrc
echo "export TRAINVAL_ANN_NAME=$(basename ${COCO_TRAINVAL_ANNOTATION})" >> .trainrc
echo "export TEST_ANN_NAME=$(basename ${COCO_TEST_ANNOTATION})" >> .trainrc
