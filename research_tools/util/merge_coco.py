def merge_coco(all_annotations):
    '''
    COCO Annotation을 합친다.
    '''
    # Merge annotation
    result_annotation = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": all_annotations[0]  # 그대로 가져감
    }

    for annotation in all_annotations:
        result_annotation["images"].extend(annotation["images"])
        result_annotation["annotations"].extend(annotation["annotations"])
    
    # Check for duplicate images (annotations will not be checked)
    image_id_list = list(map(lambda image: image["id"], annotation["images"]))
    assert len(set(image_id_list)) == len(image_id_list), "Fold image duplication detected"
    
    # Sort by key
    result_annotation["images"].sort(key=lambda item: item["id"])
    result_annotation["annotations"].sort(key=lambda item: item["image_id"])

    # Recreate indicies
    for idx, item in enumerate(result_annotation["annotations"]):
        item["id"] = idx + 1  # indicies starts from 1
        
    return result_annotation
