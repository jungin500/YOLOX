import torch
import json
from tqdm.auto import tqdm

from yolox.utils import postprocess
from yolox.utils.dist import get_local_rank, get_world_size, wait_for_the_master

from .dataset_generator import DatasetGenerator
from .util import collate_fn, classid2cocoid


class NaiiveGenerator(DatasetGenerator):
    def init(self):
        from yolox.data import (
            ValTransform,
            COCODataset
        )
        import torch.distributed as dist
        from loguru import logger
    
        logger.info('Initializing dataloader ...')
        with wait_for_the_master(get_local_rank()):
            dataset = COCODataset(
                data_dir=self.exp.data_dir,
                json_file=self.exp.val_ann,
                name="val2017",
                img_size=self.exp.test_size,
                preproc=ValTransform(legacy=False),
            )

        if self.is_distributed:
            self.batch_size = self.batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                rank=get_local_rank(),
                num_replicas=get_world_size(), 
                shuffle=False,
                drop_last=False
            )
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
        result_bboxes = []
        result_cls = []
        result_scores = []
        result_image_names = []

        if self.is_distributed:
            desc_msg = "[Rank {}] Inferencing".format(get_local_rank())
        else:
            desc_msg = "Inferencing"
            
        for total_batch_idx, (img, target, img_info, img_id) in tqdm(enumerate(self.dataloader), desc=desc_msg, total=len(self.dataloader)):
            if self.device == 'gpu':
                img = img.cuda()
            if self.half_precision:
                img = img.half()

            # Infer current scale
            with torch.no_grad():
                batched_outputs = self.model(img)
                batched_outputs = postprocess(
                    batched_outputs, self.exp.num_classes, self.exp.test_conf,
                    self.exp.nmsthre, class_agnostic=True
                )

            for batch_idx, output in enumerate(batched_outputs):
                if output is None:
                    continue

                ratio = min(self.exp.test_size[0] / img_info[0][batch_idx], self.exp.test_size[1] / img_info[1][batch_idx])

                bboxes = output[:, 0:4]
                # preprocessing: resize
                bboxes /= ratio
                cls = output[:, 6]
                scores = output[:, 4] * output[:, 5]

                result_bboxes.append(bboxes.cpu().numpy())
                result_cls.append(cls.cpu().numpy().astype(int))
                result_scores.append(scores.cpu().numpy())
                result_image_names.append(img_id[batch_idx])
    
        # JSON Annotation 저장하기
        images_map = { item['id']: item for item in self.annotations['images'] }
        result_annotations = []
        for bboxes, cls, scores, image_id in tqdm(zip(result_bboxes, result_cls, result_scores, result_image_names), desc="Organizing result bboxes", total=len(result_bboxes)):
            for bbox, cls, score in zip(bboxes, cls, scores):
                bbox = [int(i) for i in bbox]  # np.int64 items does present
                bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]  # xyminmax2xywh
                result_annotations.append({
                    'area': bbox[2] * bbox[3],
                    'iscrowd': 0,
                    'bbox': bbox,
                    'category_id': int(classid2cocoid(cls)),
                    'ignore': 0,
                    'segmentation': [],
                    'image_id': image_id,
                    'id': len(result_annotations) + 1  # 1부터 시작한다.
                })

        return {
            "images": [ images_map[image_id] for image_id in result_image_names ],
            "type": "instances",
            "annotations": result_annotations,
            "categories": self.annotations["categories"]
        }