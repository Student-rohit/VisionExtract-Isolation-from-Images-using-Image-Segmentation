import torch
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image

from model_unet import UNet

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load Model
# -----------------------------
model = UNet().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

print("Model Loaded")

# -----------------------------
# Transform
# -----------------------------
transform = T.Compose([
    T.Resize((256,256)),
    T.ToTensor()
])


# -----------------------------
# Subject Isolation Function
# -----------------------------
def isolate_subject(image_path, output_path):

    image = Image.open(image_path).convert("RGB")
    original = np.array(image)

    input_tensor = transform(image).unsqueeze(0).to(device)

    # Model prediction
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output)

    mask = probs[0].cpu().squeeze().numpy()

    # Resize mask to original image size
    mask = cv2.resize(mask, (original.shape[1], original.shape[0]))

    # Smooth edges
    mask = cv2.GaussianBlur(mask, (21,21), 0)

    # Normalize mask
    mask = mask / mask.max()

    # Convert to 3 channel
    mask_3 = np.dstack([mask, mask, mask])

    # Apply mask
    isolated = original * mask_3
    isolated = isolated.astype("uint8")

    # Save result
    Image.fromarray(isolated).save(output_path)

    return output_path