import os
import json


class DatasetGenerator(object):
    def __init__(
        self,
        exp,
        model,
        device,
        is_distributed,
        batch_size,
        half_precision
    ):
        self.exp = exp
        self.model = model
        self.device = device
        self.is_distributed = is_distributed
        self.batch_size = batch_size
        self.half_precision = half_precision
        
        # Load COCO annotation
        with open(os.path.join(exp.data_dir, 'annotations', exp.val_ann), 'r') as f:
            self.annotations = json.load(f)

        # Create imageid to annotation mapping table
        self.annotation_map = {}
        for annotation in self.annotations['annotations']:
            image_id = annotation['image_id']
            if image_id not in self.annotation_map:
                self.annotation_map[image_id] = []
            self.annotation_map[image_id].append(annotation)

    def init(self):
        raise NotImplementedError()

    def generate_dataset(self):
        raise NotImplementedError()