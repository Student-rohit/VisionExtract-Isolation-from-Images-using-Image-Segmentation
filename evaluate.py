import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import os

from dataset import CocoSubjectDataset
from model_unet import UNet


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Transform
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
])


# Dataset
dataset = CocoSubjectDataset(
    image_dir="data/data/coco2017/train2017",
    annotation_file="data/data/coco2017/annotations/instances_train2017.json",
    transform=transform
)

subset_size = 3000
dataset = torch.utils.data.Subset(dataset, range(subset_size))

loader = DataLoader(dataset, batch_size=1, shuffle=False)


# Model
model = UNet().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

print("Model loaded successfully")


# Metrics
def iou_score(pred, target):
    pred = pred.bool()
    target = target.bool()
    intersection = (pred & target).float().sum()
    union = (pred | target).float().sum()
    return (intersection + 1e-6) / (union + 1e-6)


def dice_score(pred, target):
    pred = pred.bool()
    target = target.bool()
    intersection = (pred & target).float().sum()
    return (2. * intersection + 1e-6) / (pred.float().sum() + target.float().sum() + 1e-6)


def pixel_accuracy(pred, target):
    return (pred == target).float().mean()


# Output folder
os.makedirs("outputs", exist_ok=True)


# Evaluation loop
total_iou = 0
total_dice = 0
total_acc = 0
count = 0


with torch.no_grad():
    for idx, (images, masks) in enumerate(loader):

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        probs = torch.sigmoid(outputs)

        # 🔥 improved threshold
        preds = (probs > 0.5).float()

        total_iou += iou_score(preds, masks).item()
        total_dice += dice_score(preds, masks).item()
        total_acc += pixel_accuracy(preds, masks).item()
        count += 1

        # Convert to numpy
        image_np = images[0].cpu().permute(1, 2, 0).numpy()
        mask_np = preds[0].cpu().squeeze().numpy()

        # Subject isolation
        isolated = image_np.copy()
        isolated[mask_np == 0] = 0

        # Visualization
        plt.figure(figsize=(8, 3))

        plt.subplot(1, 3, 1)
        plt.title("Input")
        plt.imshow(image_np)
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Pred Mask")
        plt.imshow(mask_np, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Isolated")
        plt.imshow(isolated)
        plt.axis("off")

        plt.savefig(f"outputs/result_{idx}.png")
        plt.close()

        print(f"Saved result_{idx}.png")


# Final Metrics
print("\nValidation Results:")
print("IoU:", total_iou / count)
print("Dice:", total_dice / count)
print("Pixel Accuracy:", total_acc / count)