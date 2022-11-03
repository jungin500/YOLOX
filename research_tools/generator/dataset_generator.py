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
        output_filename,
        half_precision
    ):
        self.exp = exp
        self.model = model
        self.device = device
        self.is_distributed = is_distributed
        self.batch_size = batch_size
        self.output_filename = output_filename
        self.half_precision = half_precision
        
        # Load COCO annotation
        with open(os.path.join(exp.data_dir, 'annotations', exp.val_ann), 'r') as f:
            self.annotations = json.load(f)

    def init(self):
        raise NotImplementedError()

    def generate_dataset(self):
        raise NotImplementedError()