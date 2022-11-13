import os
import json
from loguru import logger
import tempfile


class DatasetGenerator(object):
    def __init__(
        self,
        exp,
        model,
        device,
        is_distributed,
        batch_size,
        half_precision,
        oneshot_image_ids = None
    ):
        self.exp = exp
        self.model = model
        self.device = device
        self.is_distributed = is_distributed
        self.batch_size = batch_size
        self.half_precision = half_precision
        self.oneshot_image_ids = oneshot_image_ids
        
        self._load_coco_annotations(os.path.join(exp.data_dir, 'annotations', exp.train_ann))
        if self.oneshot_image_ids:
            # Manipulate JSON to only have requested single image
            logger.info('Loading JSON and filtering {} annotation(s) ...'.format(len(self.oneshot_image_ids)))
            with open(os.path.join(self.exp.data_dir, "annotations", self.exp.train_ann), 'r') as f:
                body = json.load(f)
            
            new_body = {
                "images": [ image for image in body["images"] if image["id"] in self.oneshot_image_ids ],
                "type": "instances",
                "annotations": [ annotation for annotation in body["annotations"] if annotation["image_id"] in self.oneshot_image_ids ],
                "categories": self.annotations["categories"]
            }
            
            assert len(new_body["images"]) == len(self.oneshot_image_ids), "Some of or any of images are found: {}".format(self.oneshot_image_ids)
            logger.info('Saving annotation: {}'.format(json.dumps(new_body, indent=4)))
            
            fd, tmp_json_filename = tempfile.mkstemp(suffix=".json")
            with open(fd, 'w') as f:
                json.dump(new_body, f)
            
            # Exploit exp!
            self.exp.train_ann = tmp_json_filename
            
    def _load_coco_annotations(self, annotation_path):
        # Load COCO annotation
        with open(annotation_path, 'r') as f:
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