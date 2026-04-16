# preprocess_coco_mask.py

import os
import cv2
import numpy as np
from pycocotools.coco import COCO


def create_mask(annotation_path, image_id, output_size=(256, 256)):
    """
    Generate binary mask from COCO annotation
    """
    coco = COCO(annotation_path)
    ann_ids = coco.getAnnIds(imgIds=image_id)
    anns = coco.loadAnns(ann_ids)

    img_info = coco.loadImgs(image_id)[0]
    height, width = img_info['height'], img_info['width']

    mask = np.zeros((height, width), dtype=np.uint8)

    for ann in anns:
        mask = np.maximum(mask, coco.annToMask(ann))

    mask = cv2.resize(mask, output_size)
    return mask


if __name__ == "__main__":
    annotation_file = "coco2017/annotations/instances_train2017.json"
    mask = create_mask(annotation_file, image_id=1)
    print("Mask shape:", mask.shape)
