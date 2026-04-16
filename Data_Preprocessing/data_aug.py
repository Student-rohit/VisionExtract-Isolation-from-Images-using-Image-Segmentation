# data_aug.py

import cv2
import numpy as np
import random


def horizontal_flip(image, mask=None):
    image = cv2.flip(image, 1)
    if mask is not None:
        mask = cv2.flip(mask, 1)
        return image, mask
    return image


def random_rotation(image, mask=None):
    angle = random.randint(-15, 15)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)

    image = cv2.warpAffine(image, M, (w, h))

    if mask is not None:
        mask = cv2.warpAffine(mask, M, (w, h))
        return image, mask

    return image


if __name__ == "__main__":
    print("Data augmentation module ready.")
