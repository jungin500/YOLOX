import numpy as np
import torch
import torch.utils.data
import torch.utils.data.distributed
import json
from tqdm.auto import tqdm

from yolox.utils import postprocess
from yolox.utils.dist import get_local_rank, get_world_size, wait_for_the_master

from .dataset_generator import DatasetGenerator
from .util import collate_fn, xywh2xyminmax, classid2cocoid, cocoid2classid, iou_np, ValDataPrefetcher


def find_iou_matching_np(bbox: np.ndarray, o_bboxes: np.ndarray, iou_thresh: float):
    iou_table = []
    for o_bbox in o_bboxes:
        iou_table.append(iou_np(bbox, o_bbox))
    iou_table = np.array(iou_table)

    iou_order_descending = np.argsort(iou_table)[::-1]
    iou_table = iou_table[iou_order_descending]
    o_bboxes = o_bboxes[iou_order_descending]
    max_iou, max_iou_o_bbox = iou_table[0], o_bboxes[0]

    if max_iou > iou_thresh:
        return True, max_iou_o_bbox, iou_order_descending[0], max_iou
    return False, max_iou_o_bbox, iou_order_descending[0], max_iou


class IOUGenerator(DatasetGenerator):

    def __init__(
        self,
        exp,
        model,
        conf,
        iou_thresh,
        device,
        is_distributed,
        batch_size,
        half_precision,
        oneshot_image_ids=None,
    ):
        super().__init__(
            exp=exp,
            model=model,  # manual assignment
            device=device,
            is_distributed=is_distributed,
            batch_size=batch_size,
            half_precision=half_precision,
            oneshot_image_ids=oneshot_image_ids)
        self.conf_thresh = conf
        self.iou_thresh = iou_thresh

    def init(self):
        from yolox.data import (ValTransform, COCODataset)
        import torch.distributed as dist
        from loguru import logger

        logger.info('Initializing dataloader ...')
        with wait_for_the_master(get_local_rank()):
            dataset = COCODataset(
                data_dir=self.exp.data_dir,
                json_file=self.exp.train_ann,
                name="train2017",
                img_size=self.exp.test_size,
                preproc=ValTransform(legacy=False),
            )

        if self.is_distributed:
            self.batch_size = self.batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                      rank=get_local_rank(),
                                                                      num_replicas=get_world_size(),
                                                                      shuffle=False,
                                                                      drop_last=False)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

        dataloader_kwargs = {
            "num_workers": self.exp.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
            "collate_fn": collate_fn,
        }
        dataloader_kwargs["batch_size"] = self.batch_size
        self.dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    def generate_dataset(self):
        results = []

        if self.is_distributed:
            desc_msg = "[Rank {}] Inferencing".format(get_local_rank())
        else:
            desc_msg = "Inferencing"

        prefetcher = ValDataPrefetcher(self.dataloader)
        pbar = tqdm(range(len(self.dataloader)), desc=desc_msg)
        while True:
            img, target, img_info, img_id = prefetcher.next()
            if type(img) == type(None):
                break  # End of prefetcher

            if self.half_precision:
                img = img.to(torch.float16)

            # Infer current scale
            with torch.no_grad():
                batched_outputs = self.model(img)
                batched_outputs = postprocess(batched_outputs,
                                              self.exp.num_classes,
                                              self.exp.test_conf,
                                              self.exp.nmsthre,
                                              class_agnostic=True)

            for batch_idx, output in enumerate(batched_outputs):
                if output is None:
                    continue

                ratio = min(self.exp.test_size[0] / img_info[0][batch_idx],
                            self.exp.test_size[1] / img_info[1][batch_idx])

                bboxes = output[:, 0:4]
                # preprocessing: resize
                bboxes /= ratio
                cls = output[:, 6]
                scores = output[:, 4] * output[:, 5]

                bboxes = bboxes.cpu().numpy()
                cls = cls.cpu().numpy().astype(int)
                scores = scores.cpu().numpy()
                current_img_id = img_id[batch_idx]

                matched_objects, gt_nonmatched_objects, infer_nonmatched_objects = self.iou_match(
                    image_name=current_img_id, bboxes=bboxes, cls=cls, scores=scores)

                results.append([current_img_id, matched_objects])

            pbar.update()

        # JSON Annotation 저장하기
        images_map = {item['id']: item for item in self.annotations['images']}
        result_annotations = []
        for image_id, bboxes in tqdm(results, desc="Organizing result bboxes"):
            for class_id in bboxes.keys():
                class_bboxes = bboxes[class_id]
                for bbox in class_bboxes:
                    bbox = [int(i) for i in bbox]  # np.int64 items does present
                    bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]  # xyminmax2xywh
                    result_annotations.append({
                        'area': bbox[2] * bbox[3],
                        'iscrowd': 0,
                        'bbox': bbox,
                        'category_id': int(classid2cocoid(class_id)),
                        'ignore': 0,
                        'segmentation': [],
                        'image_id': image_id,
                        'id': len(result_annotations) + 1  # 1부터 시작한다.
                    })

        return {
            "images": [images_map[image_id] for image_id, bboxes in results],
            "type": "instances",
            "annotations": result_annotations,
            "categories": self.annotations["categories"]
        }

    def iou_match(self, image_name, bboxes, cls, scores):
        if image_name not in self.annotation_map:
            # 어노테이션 없는 빈 이미지 (Background)
            o_items = []
        else:
            o_items = self.annotation_map[image_name]

        o_cls = np.array(list(map(lambda item: cocoid2classid(item['category_id']), o_items))).astype(int)
        o_bboxes = np.array(list(map(lambda item: item['bbox'], o_items)))
        o_bboxes = np.array(list(map(xywh2xyminmax, o_bboxes)))

        matched_objects = {}
        infer_nonmatched_objects = {}
        gt_nonmatched_objects = {}

        # 클래스별 IoU 매칭
        for target_cls in np.unique(cls):
            target_cls_idxname = int(target_cls)
            matched_objects[target_cls_idxname] = []
            infer_nonmatched_objects[target_cls_idxname] = []

            idxmap = np.where(cls == target_cls)
            assert len(bboxes[idxmap]) == len(cls[idxmap]) == len(scores[idxmap])

            o_bboxes_targetcls = o_bboxes[np.where(o_cls == target_cls)]

            for bbox, single_cls, score in zip(bboxes[idxmap], cls[idxmap], scores[idxmap]):
                if score < self.conf_thresh:
                    # print('Skipping bbox1 {} due to low confidence {}'.format(bbox, score))
                    continue

                if len(o_bboxes_targetcls) == 0:
                    # Add all bbox1es to nonmatched_objects (GT doesn't have any object!)
                    # print('bbox1 {} unmatched (no bbox2 candidates)'.format(bbox))
                    infer_nonmatched_objects[target_cls_idxname].append(bbox)
                    continue

                match_result, o_bbox, o_bbox_id, max_iou = find_iou_matching_np(bbox, o_bboxes_targetcls,
                                                                                self.iou_thresh)
                # Remove that bbox from o_bboxes
                o_bboxes_targetcls = np.delete(o_bboxes_targetcls, o_bbox_id, axis=0)

                if match_result:
                    matched_objects[target_cls_idxname].append(bbox)
                    # print('bbox1 {} matched bbox2 {} (IoU {})'.format(bbox, o_bbox, max_iou))
                else:
                    infer_nonmatched_objects[target_cls_idxname].append(bbox)
                    # print('bbox1 {} unmatched (max iou bbox2 {} had iou {}'.format(bbox, o_bbox, max_iou))

            # Check leftover o_bboxes
            if len(o_bboxes_targetcls) > 0:
                # print('leftover bbox2es: {}'.format(o_bboxes_targetcls))
                gt_nonmatched_objects[target_cls_idxname] = o_bboxes_targetcls
            else:
                gt_nonmatched_objects[target_cls_idxname] = []

        return matched_objects, infer_nonmatched_objects, gt_nonmatched_objects
