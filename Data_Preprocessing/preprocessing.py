# preprocessing.py

import cv2
import numpy as np

def load_image(image_path):
    """
    Load image from given path
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found at path:", image_path)
    return image


def resize_image(image, size=(256, 256)):
    """
    Resize image to fixed size
    """
    return cv2.resize(image, size)


def normalize_image(image):
    """
    Normalize image to range [0,1]
    """
    return image / 255.0


def preprocess_image(image_path):
    """
    Complete preprocessing pipeline
    """
    image = load_image(image_path)
    image = resize_image(image)
    image = normalize_image(image)
    return image


if __name__ == "__main__":
    img = preprocess_image("sample.jpg")
    print("Preprocessing Done. Shape:", img.shape)
