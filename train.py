import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T

from dataset import CocoSubjectDataset
from model_unet import UNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


transform = T.Compose([
    T.Resize((256, 256)),   
    T.ToTensor(),
])


dataset = CocoSubjectDataset(
    image_dir="data/data/coco2017/train2017",
    annotation_file="data/data/coco2017/annotations/instances_train2017.json",
    transform=transform
)

print("Total images in dataset:", len(dataset))


subset_size = 5000
dataset = torch.utils.data.Subset(dataset, range(subset_size))

print("Using subset size:", subset_size)


loader = DataLoader(dataset, batch_size=4, shuffle=True)


model = UNet().to(device)


criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training Loop
num_epochs = 8

for epoch in range(num_epochs):
    print(f"\nStarting Epoch {epoch+1}/{num_epochs}")
    model.train()
    epoch_loss = 0

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device).float()

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Batch [{batch_idx}/{len(loader)}] Loss: {loss.item():.4f}")

    print(f"\nEpoch Loss: {epoch_loss / len(loader):.4f}")

print("\nTraining Completed Successfully ")




torch.save(model.state_dict(), "model.pth")
print("Model saved as model.pth ")