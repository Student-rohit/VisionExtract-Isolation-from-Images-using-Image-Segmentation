# VisionExtract-Isolation-from-Images-using-Image-Segmentation

## 📌 Overview
VisionExtract is a Deep Learning and Computer Vision project that performs image segmentation to isolate objects from images using the U-Net architecture. The project is trained on the COCO2017 dataset and provides functionalities for preprocessing, augmentation, training, inference, evaluation, and prediction.

This system helps in extracting meaningful regions from images for applications such as:
- Object Extraction
- Background Removal
- Medical Imaging
- Autonomous Systems
- Smart Surveillance
---

# 🚀 Features

✅ Image Segmentation using U-Net  
✅ COCO2017 Dataset Support  
✅ Data Preprocessing & Augmentation  
✅ Model Training and Evaluation  
✅ Prediction on Custom Images  
✅ Flask Web Application Integration  
✅ Segmented Output Visualization  
✅ Easy-to-use Project Structure  
---

# 🛠️ Technologies Used

- Python
- OpenCV
- NumPy
- PyTorch
- Flask
- Matplotlib
- COCO2017 Dataset

---

# 📂 Project Structure

```bash
VISIONEXTRACT/
│
├── data/
│   └── data/
│       └── coco2017/
│           ├── annotations/
│           ├── test2017/
│           ├── train2017/
│           └── val2017/
│
├── Data_Preprocessing/
│   ├── data_aug.py
│   ├── img_viz.py
│   ├── preprocessing.py
│   └── __pycache__/
│
├── img/                      # Project images/screenshots
├── outputs/                  # Segmented output images
├── static/                   # Flask static files
├── Templates/                # HTML templates
├── test_images/              # Test images for prediction
│
├── app.py                    # Flask application
├── dataset.py                # Dataset loader
├── evaluate.py               # Model evaluation
├── inference.py              # Inference pipeline
├── model_unet.py             # U-Net model architecture
├── model.pth                 # Trained model
├── predict.py                # Prediction script
├── requirements.txt          # Required dependencies
├── sample_image.py           # Sample image testing
├── train.py                  # Training script
│
└── README.md
```
---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
https://github.com/Student-rohit/VisionExtract-Isolation-from-Images-using-Image-Segmentation.git`

---

## 2️⃣ Navigate to Project Folder

```bash
cd VisionExtract-Isolation-from-Images-using-Image-Segmentation
```

---

## 3️⃣ Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

## 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt

---

# ▶️ How to Run

## Train the Model

```bash
python train.py
```

---

## Run Prediction

```bash
python predict.py
```

---

## Evaluate Model

```bash
python evaluate.py
```

---

## Run Flask Web Application

```bash
python app.py
```

Open browser:

```bash
http://127.0.0.1:5000/
```

---

# 🧠 Model Architecture

This project uses the **U-Net Deep Learning Architecture** for semantic image segmentation.

### Key Components:
- Encoder
- Bottleneck
- Decoder
- Skip Connections

---

# 📊 Dataset

## Dataset Used:
COCO2017 Dataset

### Dataset folders:
- train2017
- val2017
- test2017
- annotations

---

# 📸 Output

The model generates:
- Segmentation Masks
- Isolated Objects
- Processed Images

Outputs are saved inside:

```bash
outputs/
```

# 🌐 Applications

- Medical Image Analysis
- Satellite Imaging
- Background Removal
- AI Surveillance
---

# 🔮 Future Improvements

- Real-time video segmentation
- GPU optimization
- Mobile app integration
  
