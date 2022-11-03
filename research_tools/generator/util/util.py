import numpy as np
import torch


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


def xywh2xyminmax(bbox):
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

    
def cocoid2classid(coco_id: int):
    # COCO에서는 helmet_on(1) ~ belt_off(4)로 구성된다.
    return coco_id - 1

def classid2cocoid(class_id: int):
    # class_id는 0부터 시작한다.
    return class_id + 1