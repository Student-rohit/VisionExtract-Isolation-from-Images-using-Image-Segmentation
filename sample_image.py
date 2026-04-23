import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt

from dataset import CocoSubjectDataset

# ---- Transforms ----
image_transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

# ---- Dataset ----
dataset = CocoSubjectDataset(
    image_dir="data/data/coco2017/train2017",
    annotation_file="data/data/coco2017/annotations/instances_train2017.json",
    image_transform=image_transform
)

# ---- DataLoader ----
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# ---- Get one sample ----
image, mask = next(iter(dataloader))

# Remove batch dimension
image = image.squeeze(0)   # [3, H, W]
mask = mask.squeeze(0)     # [1, H, W]

# Convert to numpy for display
image_np = image.permute(1, 2, 0).numpy()
mask_np = mask.squeeze(0).numpy()

# ---- Plot ----
plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(image_np)
plt.title("Original Image")
plt.axis("off")

# Mask
plt.subplot(1, 2, 2)
plt.imshow(mask_np, cmap='gray')
plt.title("Generated Mask")
plt.axis("off")

plt.show()