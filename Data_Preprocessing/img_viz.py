# img_viz.py

import cv2
import matplotlib.pyplot as plt


def show_image(image, title="Image"):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()


def show_mask(mask, title="Mask"):
    plt.imshow(mask, cmap='gray')
    plt.title(title)
    plt.axis("off")
    plt.show()


def show_overlay(image, mask):
    overlay = image.copy()
    overlay[mask > 0] = [0, 255, 0]  # Green mask

    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Overlay")
    plt.axis("off")
    plt.show()
