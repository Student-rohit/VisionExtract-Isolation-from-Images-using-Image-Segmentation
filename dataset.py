import os
import torch
import numpy as np
import cv2
from PIL import Image
from pycocotools.coco import COCO


class CocoSubjectDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]

        img_path = os.path.join(self.image_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")

        # Create mask
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

        for ann in anns:
            m = self.coco.annToMask(ann)
            mask = np.maximum(mask, m)

        mask = Image.fromarray(mask * 255)

        # Apply transforms
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Ensure mask is binary (0 or 1)
        mask = (mask > 0).float()

        return image, mask
        