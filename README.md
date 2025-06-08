
# 🚦 Accident Detection in CCTV Footage using YOLOv8

This project implements a **real-time accident detection system** using deep learning. It leverages the YOLOv8 object detection model to identify accidents in tunnel or CCTV video streams and images.

---

## 📝 Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Running Inference](#running-inference)
- [Performance Analysis](#performance-analysis)
- [Results](#results)
- [Future Work](#future-work)

---

## 📌 Introduction

The increasing number of road and tunnel accidents captured on CCTV requires an **automated, real-time accident detection system** to reduce response time and enhance public safety.

This project:
- Uses **YOLOv8** for fast object detection.
- Trains on a **custom accident dataset** in COCO format.
- Can process both **images and videos**.
- Provides **evaluation metrics** and **performance graphs** for analysis.

---

## 📂 Project Structure

```
├── detect_accident.py          # For running inference on images or videos
├── evaluate.py                 # Evaluate trained model on validation set
├── performance_analysis.py     # Tabular and graphical performance analysis
├── train_yolov8.py             # Training script for YOLOv8 on custom dataset
├── accident_results/           # Directory to store model weights, results, graphs
└── README.md                   # Project documentation (this file)
```

---

## ⚙️ Setup & Installation

### Requirements:
- Python ≥ 3.8
- PyTorch ≥ 1.8
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV
- Matplotlib
- Pandas
- tabulate

### Installation:

```bash
# Clone this repo
git clone https://github.com/your_username/accident-detection-yolov8.git
cd accident-detection-yolov8

# Install dependencies
pip install ultralytics opencv-python matplotlib pandas tabulate
```

---

## 🏋️‍♂️ Training the Model

Modify `train_yolov8.py` with your dataset path:

```python
data=r"D:\your_path\data.yaml"
```

Then run:

```bash
python train_yolov8.py
```

Outputs:
- Weights: `accident_results/accident_yolov8_model/weights/best.pt`
- Metrics: `metrics.csv`, `evaluation_graphs.png`

---

## 📊 Evaluating the Model

Run:

```bash
python evaluate.py
```

Outputs:
- **mAP@0.5**
- **mAP@0.5:0.95**
- **Precision**
- **Recall**

You can also run `performance_analysis.py` for **beautiful tabular + graphical analysis**:

```bash
python performance_analysis.py
```

---

## 🕵️ Running Inference

Run on an image or video:

```bash
python detect_accident.py
```

Modify the `image_or_video_path` in `detect_accident.py`:

```python
detect_accidents("path/to/your/image_or_video.mp4")
```

Results will be saved to:

```
runs/detect/accident_detection/
```

---

## 📈 Performance Analysis

- `performance_analysis.py` generates:
  - `evaluation_table.png`
  - Loss and metric trends per epoch.

---

## 🏆 Results

- Achieved **mAP@0.5 ≈ X.XXX**
- Real-time processing capability on 4GB GPU.
- Works well for **tunnel CCTV camera data**.

---

## 🚀 Future Work

- Integrate **SORT tracking** for continuous accident tracking.
- Deploy system on **edge devices** (Jetson Nano / Raspberry Pi).
- Optimize for **real-time alerts** to traffic control systems.

---

## 📚 Citation / Credits

Based on:
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
