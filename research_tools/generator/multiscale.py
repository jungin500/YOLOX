import numpy as np
import torch
import json
from tqdm.auto import tqdm
from copy import deepcopy

from yolox.utils import postprocess
from yolox.utils.dist import get_local_rank, get_world_size, wait_for_the_master

from .dataset_generator import DatasetGenerator
from .util import collate_fn, xywh2xyminmax, classid2cocoid, cocoid2classid, iou_np


class MultiscaleGenerator(DatasetGenerator):
    def __init__(
        self,
        exp,
        model,
        scales,
        conf,
        iou_thresh,
        rematch_thresh,
        device,
        is_distributed,
        batch_size,
        half_precision
    ):
        super().__init__(
            exp = exp,
            model = None,  # manual assignment
            device = device,
            is_distributed = is_distributed,
            batch_size = batch_size,
            half_precision = half_precision
        )
        self.scales = scales
        self.conf_thresh = conf
        self.iou_thresh = iou_thresh
        self.rematch_thresh = rematch_thresh

        self.models = {}
        for scale in scales:
            self.models[scale] = deepcopy(model)

    def init(self):
        from yolox.data import (
            ValTransform,
            COCODataset
        )
        import torch.distributed as dist
        from loguru import logger
        
        dataset_map = {}
        sampler_map = {}
        self.dataloader_map = {}
        for scale in self.scales:
            logger.info('Initializing scale {} ...'.format(scale))
            with wait_for_the_master(get_local_rank()):
                dataset_map[scale] = COCODataset(
                    data_dir=self.exp.data_dir,
                    json_file=self.exp.val_ann,
                    name="train2017",
                    img_size=(scale, scale),
                    preproc=ValTransform(legacy=False),
                )

            if self.is_distributed:
                target_batch_size = self.batch_size // dist.get_world_size()
                sampler_map[scale] = torch.utils.data.distributed.DistributedSampler(
                    dataset_map[scale],
                    rank=get_local_rank(),
                    num_replicas=get_world_size(), 
                    shuffle=False,
                    drop_last=False
                )
            else:
                target_batch_size = self.batch_size
                sampler_map[scale] = torch.utils.data.SequentialSampler(dataset_map[scale])

            dataloader_kwargs = {
                "num_workers": self.exp.data_num_workers,
                "pin_memory": True,
                "sampler": sampler_map[scale],
                "collate_fn": collate_fn,
            }
            dataloader_kwargs["batch_size"] = target_batch_size
            self.dataloader_map[scale] = torch.utils.data.DataLoader(dataset_map[scale], **dataloader_kwargs)

    def generate_dataset(self):
        results = []
        
        if self.is_distributed:
            desc_msg = "[Rank {}] Inferencing".format(get_local_rank())
        else:
            desc_msg = "Inferencing"

        iterators = { scale: iter(self.dataloader_map[scale]) for scale in self.scales }
        for total_batch_idx in tqdm(range(len(self.dataloader_map[next(iter(self.scales))])), desc=desc_msg):
            bboxes_batched, cls_batched, scores_batched, image_names = self.infer_batch(iterators)

            for batch_idx in range(len(bboxes_batched)):
                matched_objects, gt_nonmatched_objects, infer_nonmatched_objects = self.multiscale_match(
                    image_name = image_names[batch_idx],
                    bboxes_scales = bboxes_batched[batch_idx],
                    cls_scales = cls_batched[batch_idx],
                    scores_scales = scores_batched[batch_idx],
                )

                results.append([image_names[batch_idx], matched_objects])
        
        # JSON Annotation 저장하기
        images_map = { item['id']: item for item in self.annotations['images'] }
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
            "images": [ images_map[image_id] for image_id, bboxes in results ],
            "type": "instances",
            "annotations": result_annotations,
            "categories": self.annotations["categories"]
        }

    def multiscale_match(self, image_name, bboxes_scales, cls_scales, scores_scales):
        o_items = self.annotation_map[image_name]

        o_cls = np.array(list(map(lambda item : cocoid2classid(item['category_id']), o_items))).astype(int)
        o_bboxes = np.array(list(map(lambda item : item['bbox'], o_items)))
        o_bboxes = np.array(list(map(xywh2xyminmax, o_bboxes)))

        gt_objects = {}
        for class_id in sorted(np.unique(o_cls)):
            gt_objects[class_id] = o_bboxes[np.where(o_cls == class_id)]

        infer_objects = {}
        for scale in self.scales:
            bboxes = bboxes_scales[scale]
            cls = cls_scales[scale]
            scores = scores_scales[scale]

            if np.all(bboxes == None):
                continue
            
            scoremap = np.where(scores > self.conf_thresh)
            bboxes = bboxes[scoremap]
            cls = cls[scoremap]
            scores = scores[scoremap]

            for class_id in sorted(np.unique(cls)):
                if class_id not in infer_objects:
                    infer_objects[class_id] = {}
                infer_objects[class_id][scale] = bboxes[np.where(cls == class_id)]
        
        match_table_all = {}
        gt_only_bbox_table_all = {}
        infer_only_bbox_table_all = {}

        for class_id in sorted(np.unique(np.concatenate([
            *[cls_scales[scale] for scale in self.scales if np.all(cls_scales[scale] != None)], o_cls
        ]).astype(int))):
            infer_only_extras = [] # To be used after picking infer_nonmatched

            if class_id not in infer_objects:
                match_table = {}
                gt_only_bbox_table = []
                if class_id in gt_objects:
                    for idx in range(len(gt_objects[class_id])):
                        match_table[idx] = []
                    gt_only_bbox_table = gt_objects[class_id]
                infer_only_bbox_table = []
            elif class_id not in gt_objects:
                match_table = {}
                gt_only_bbox_table = []
                infer_only_bbox_table = []
                if class_id in infer_objects:
                    for scale in sorted(infer_objects[class_id].keys()):
                        infer_only_bbox_table.extend(infer_objects[class_id][scale])
            else:
                gt_bboxes = gt_objects[class_id]
                infer_bboxes_scales = infer_objects[class_id]

                match_table = { idx: [] for idx in range(len(gt_bboxes)) }
                gt_only_bbox_table = np.copy(gt_bboxes)
                infer_only_bbox_table = []

                # First match GT with Infer
                # 매칭 결과는 match_table에 GT bbox index를 Key로 리스트로 저장
                # 매칭되지 않은 결과는 gt_only_bbox_table과 infer_only_bbox_tabLle에 저장
                for gt_idx, gt_bbox_single in enumerate(gt_bboxes):
                    over_iou_found = False

                    for scale in sorted(infer_bboxes_scales.keys()):
                        if len(infer_bboxes_scales[scale]) == 0:
                            continue

                        ious = []
                        for infer_bbox_single in infer_bboxes_scales[scale]:
                            ious.append(iou_np(gt_bbox_single, infer_bbox_single))
                        ious = np.array(ious)

                        # 비슷한 클래스에 bbox가 두개 이상 있는 경우도 존재할수 있다는것...
                        # assert np.sum(ious >= iou_thresh) in [0, 1], "More than 1 matching bbox detected!"

                        # 결국 아래에서 Max IoU인 bbox만 사용하기 때문에 무방하다.
                        maxiou_idx = np.argmax(ious)
                        if ious[maxiou_idx] >= self.iou_thresh:
                            # match_table[gt_idx].append({ scale: infer_bboxes_scales[scale] })
                            match_table[gt_idx].append(infer_bboxes_scales[scale])
                            infer_bboxes_scales[scale] = np.delete(infer_bboxes_scales[scale], maxiou_idx, axis=0)
                            over_iou_found = True
                        
                    if over_iou_found:
                        gt_newptr_idx = -1
                        for newptr_idx, gt_new_bbox_single in enumerate(gt_only_bbox_table):
                            if np.all(gt_new_bbox_single == gt_bbox_single):
                                gt_newptr_idx = newptr_idx
                                break
                        assert gt_newptr_idx != -1, "Array Inconsistency Detected!"

                        gt_only_bbox_table = np.delete(gt_only_bbox_table, gt_newptr_idx, axis=0)
                    
                # 모든 GT 매칭이 끝나고 남은 bbox를 추가한다.
                for scale in sorted(infer_bboxes_scales.keys()):
                    infer_only_bbox_table.extend(infer_bboxes_scales[scale])

            # Class-inaware per-class rematch infer_only_bbox
            # 이전 매칭 결과중 infer_only_bbox_table에 있는 박스들을 서로 매칭하여
            # 서로 N쌍 이상 매칭되는 쌍을 GT로 설정하고 해당 infer_only_bbox_table로부터 제거
            srcbbox_idx = 0
            while srcbbox_idx < len(infer_only_bbox_table):
                # Begin srcbbox loop
                srcbbox = infer_only_bbox_table[srcbbox_idx]
                
                iou_table = []
                for dstbbox in infer_only_bbox_table:  # Include self-bbox on purpose (Will be IoU=1.0)
                    iou_table.append(iou_np(srcbbox, dstbbox))
                iou_table = np.array(iou_table)

                is_srcbbox_removed = False
                iou_argsort = np.argsort(iou_table)
                if np.all(iou_table[iou_argsort][::-1][:self.rematch_thresh] >= self.iou_thresh):
                    # Remove all thresh_over_bbox from infer_only_bbox_table
                    thresh_over_flags = iou_table >= self.iou_thresh
                    thresh_over_bboxes = [bbox for idx, bbox in enumerate(infer_only_bbox_table) if thresh_over_flags[idx]]  # Reason why inclueded itself 
                    for thresh_over_bbox in thresh_over_bboxes:
                        # Find appropriate bbox and remove
                        # It will remove srcbbox as well, so no further removal required
                        for idx, dstbbox in enumerate(infer_only_bbox_table):
                            if np.all(dstbbox == thresh_over_bbox):
                                infer_only_bbox_table = np.delete(infer_only_bbox_table, idx, axis=0)
                                break
                    infer_only_extras.append(srcbbox)
                    is_srcbbox_removed = True

                if not is_srcbbox_removed:
                    srcbbox_idx += 1
                # End srcbbox loop

            match_table_target = [
                (np.array(gt_bboxes[gt_idx]).tolist(), len(match_table[gt_idx]))
                for gt_idx in match_table.keys()
                if len(match_table[gt_idx]) > 0
            ]
            match_table_target.extend([
                (np.array(item).tolist(), 1)
                for item in infer_only_extras
            ])
            if len(match_table_target) > 0:
                match_table_all[class_id] = match_table_target
            if len(gt_only_bbox_table) > 0:
                gt_only_bbox_table_all[class_id] = np.array(gt_only_bbox_table).tolist()
            if len(infer_only_bbox_table) > 0:
                infer_only_bbox_table_all[class_id] = np.array(infer_only_bbox_table).tolist()
        
        # 동일 위치에서 지배적인 클래스를 우선하는 Matching Strategy
        #
        # 전체 bbox를 돌면서 Inner match (match_table->match_table)과
        # Outer match (match_table->gt_only|infer_only)를 수행한다.
        flatten_matched_items = []
        for class_id in sorted(match_table_all.keys()):
            flatten_matched_items.extend([[*bbox, class_id, occurance] for (bbox, occurance) in match_table_all[class_id]])
        flatten_matched_items = np.array(flatten_matched_items)
            
        flatten_gt_only_items = []
        for class_id in sorted(gt_only_bbox_table_all.keys()):
            flatten_gt_only_items.extend([[*bbox, class_id, 1] for bbox in gt_only_bbox_table_all[class_id]])
        flatten_gt_only_items = np.array(flatten_gt_only_items)

        flatten_infer_only_items = []
        for class_id in sorted(infer_only_bbox_table_all.keys()):
            flatten_infer_only_items.extend([[*bbox, class_id, 1] for bbox in infer_only_bbox_table_all[class_id]])
        flatten_infer_only_items = np.array(flatten_infer_only_items)

        matched_item_idx = 0
        while matched_item_idx < len(flatten_matched_items):
            *bbox_origin, class_id, class_occurances = flatten_matched_items[matched_item_idx]
            
            iou_table = []  # Inner Match
            gt_iou_table = []  # Outer Match (GT->Matched)
            infer_iou_table = []  # Outer Match (Infer->Matched)
            
            # 지배적인 클래스를 찾아서 해당 클래스로 세팅하고,
            # flatten_items로부터 iou_over_items에 해당하는 bbox를 삭제한다.
            def sanitize_bboxes(flatten_matched_items, flatten_items, iou_over_items):
                # 지배적인 클래스 찾기
                all_classes = [class_id]
                for *_, target_class_id, target_class_occurances in iou_over_items:
                    for _ in range(int(target_class_occurances)):
                        all_classes.append(target_class_id)
                all_classes = np.array(all_classes)
                unique_class_ids, unique_class_counts = np.unique(all_classes, return_counts=True)
                dorminant_class_id = unique_class_ids[np.argmax(unique_class_counts)]

                # 겹치는 박스 모두 지우기
                for item in iou_over_items:
                    idx = -1
                    for target_idx, target_item in enumerate(flatten_items):
                        if np.all(item == target_item):
                            idx = target_idx
                            break
                    assert idx != -1
                    flatten_items = np.delete(flatten_items, idx, axis=0)

                # 현재 박스의 클래스를 지배적인 클래스로 변경하기
                flatten_matched_items[matched_item_idx] = np.array([*bbox_origin, dorminant_class_id, 1])
                return flatten_items
                
            # Inner Match
            for idx, (*bbox_target, class_id, class_occurances) in enumerate(flatten_matched_items):
                if np.all(bbox_origin == bbox_target):
                    continue
                iou_table.append((idx, iou_np(np.array(bbox_origin), np.array(bbox_target))))
            
            iou_over_items = [flatten_matched_items[idx] for idx, iou in iou_table if iou > self.iou_thresh]
            if len(iou_over_items) > 0:
                flatten_matched_items = sanitize_bboxes(flatten_matched_items, flatten_matched_items, iou_over_items)

            # Outer Match (GT->Matched)
            for idx, (*bbox_target, class_id, class_occurances) in enumerate(flatten_gt_only_items):
                gt_iou_table.append((idx, iou_np(np.array(bbox_origin), np.array(bbox_target))))
                
            iou_over_items = [flatten_gt_only_items[idx] for idx, iou in gt_iou_table if iou > self.iou_thresh]
            if len(iou_over_items) > 0:
                flatten_gt_only_items = sanitize_bboxes(flatten_matched_items, flatten_gt_only_items, iou_over_items)

            # Outer Match (Infer->Matched)
            for idx, (*bbox_target, class_id, class_occurances) in enumerate(flatten_infer_only_items):
                infer_iou_table.append((idx, iou_np(np.array(bbox_origin), np.array(bbox_target))))
                
            iou_over_items = [flatten_infer_only_items[idx] for idx, iou in infer_iou_table if iou > self.iou_thresh]
            if len(iou_over_items) > 0:
                flatten_infer_only_items = sanitize_bboxes(flatten_matched_items, flatten_infer_only_items, iou_over_items)
                
            matched_item_idx += 1

        # 다 끝난 flatten_ 박스들을 match_table 형식으로 되돌려놓는다.
        match_table_all = {}
        gt_only_bbox_table_all = {}
        infer_only_bbox_table_all = {}

        for *bbox, class_id, class_occurances in flatten_matched_items:
            class_id = int(class_id)
            if class_id not in match_table_all:
                match_table_all[class_id] = []
            match_table_all[class_id].append(bbox)
            
        for *bbox, class_id, class_occurances in flatten_gt_only_items:
            class_id = int(class_id)
            if class_id not in gt_only_bbox_table_all:
                gt_only_bbox_table_all[class_id] = []
            gt_only_bbox_table_all[class_id].append(bbox)
            
        for *bbox, class_id, class_occurances in flatten_infer_only_items:
            class_id = int(class_id)
            if class_id not in infer_only_bbox_table_all:
                infer_only_bbox_table_all[class_id] = []
            infer_only_bbox_table_all[class_id].append(bbox)

        return match_table_all, gt_only_bbox_table_all, infer_only_bbox_table_all

    def infer_batch(self, iterators):
        # Per-scale inference
        bboxes_batched = []
        cls_batched = []
        scores_batched = []
        image_names = []
        for scale_idx, scale in enumerate(self.scales):
            try:
                img, target, img_info, img_id = next(iterators[scale])
            except StopIteration:
                print("StopIteartion occured")
                continue

            if self.device == 'gpu':
                img = img.cuda()
            if self.half_precision:
                img = img.half()

            # Infer current scale
            with torch.no_grad():
                batched_outputs = self.models[scale](img)
                batched_outputs = postprocess(
                    batched_outputs, self.exp.num_classes, self.exp.test_conf,
                    self.exp.nmsthre, class_agnostic=True
                )

            for batch_idx, output in enumerate(batched_outputs):
                if scale_idx == 0:
                    bboxes_batched.append({})
                    cls_batched.append({})
                    scores_batched.append({})
                    image_names.append(img_id[batch_idx])

                if output is None:
                    bboxes_batched[batch_idx][scale] = None
                    cls_batched[batch_idx][scale] = None
                    scores_batched[batch_idx][scale] = None
                    continue

                ratio = min(scale / img_info[0][batch_idx], scale / img_info[1][batch_idx])

                bboxes = output[:, 0:4]
                # preprocessing: resize
                bboxes /= ratio
                cls = output[:, 6]
                scores = output[:, 4] * output[:, 5]

                bboxes_batched[batch_idx][scale] : np.ndarray = bboxes.cpu().numpy()
                cls_batched[batch_idx][scale] : np.ndarray = cls.cpu().numpy().astype(int)
                scores_batched[batch_idx][scale] : np.ndarray = scores.cpu().numpy()
        
        return bboxes_batched, cls_batched, scores_batched, image_names

