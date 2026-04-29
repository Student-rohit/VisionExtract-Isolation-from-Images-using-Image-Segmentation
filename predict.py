import torch
import cv2
import numpy as np
import os
from model_unet import UNet

device = torch.device("cpu")

model = UNet()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

input_folder = "test_images"
output_folder = "outputs"

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    image_path = os.path.join(input_folder, file)
    image = cv2.imread(image_path)
    original = image.copy()

    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        mask = torch.sigmoid(output)
        mask = mask.squeeze().cpu().numpy()

    mask = (mask > 0.5).astype("uint8")
    mask = cv2.resize(mask, (original.shape[1], original.shape[0]))
    mask = np.expand_dims(mask, axis=2)

    result = original * mask

    output_path = os.path.join(output_folder, "output_" + file)
    cv2.imwrite(output_path, result)

print("All predictions completed successfully!")