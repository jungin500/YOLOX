# encoding: utf-8
import os

import torch
import torch.distributed as dist

from loguru import logger
from yolox.exp import Exp as MyExp

CLASSES = (
    "helmet_on",
    "helmet_off",
    "belt_on",
    "belt_off",
)

g_class_to_ind = dict(
    zip(CLASSES, range(len(CLASSES)))
)

import numpy as np

def collate_fn(args):
    batched_image = []
    batched_label = []
    batched_shape = []
    batched_coco_name = []

    for image, label, shapes, coco_name in args:
        batched_image.append(image)
        batched_label.append(label)
        batched_shape.append(shapes)
        batched_coco_name.append(coco_name[0])

    batched_image = np.array(batched_image)
    batched_label = np.array(batched_label)
    batched_shape = np.array(batched_shape).T  # Mandatory!!!
    batched_coco_name = np.array(batched_coco_name)

    batched_image = torch.from_numpy(batched_image)
    batched_label = torch.from_numpy(batched_label)
    batched_shape = torch.from_numpy(batched_shape)
    # Skip coco name

    return batched_image, batched_label, batched_shape, batched_coco_name

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # YOLOX-S base parameters
        self.depth = 0.33
        self.width = 0.50
        
        # Things to be clearify (values not changed at all)
        self.act = 'silu'
        self.multiscale_range = 5
        self.enable_mixup = True
        self.max_epoch = 300
        self.no_aug_epochs = 15
        self.momentum =0.9

        # Custom configuration below
        self.num_classes = 4
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.data_num_workers = 8
        
        # Debugging configuration
        self.print_interval = 20
        self.eval_interval = 10

        # COCO dataset configuration
        self.data_dir = os.path.join('datasets', 'construction-safety-coco')
        self.train_ann = "annotations_trainval.json"
        self.val_ann = "annotations_test.json"
        self.test_ann = "annotations_test.json"
        
        # Freeze backbone configuration
        self.freeze_backbone = 'none'

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_model(self):
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
        from yolox.utils import freeze_module
        from torch import nn

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()

        if self.freeze_backbone == 'cspdarknet':
            logger.warning("Freezing cspdarknet, check if pretrained weight is used. put freeze_backbone none to discard.")
            freeze_module(self.model.backbone.backbone)
        
        return self.model

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import (
            COCODataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import wait_for_the_master

        with wait_for_the_master():
            dataset = COCODataset(
                data_dir=self.data_dir,
                json_file=self.train_ann,
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob),
                cache=cache_img,
            )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        dataloader_kwargs["collate_fn"] = collate_fn

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import COCODataset, ValTransform

        valdataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann if not testdev else self.test_ann,
            name="val2017" if not testdev else "test2017",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
            "collate_fn": collate_fn,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader