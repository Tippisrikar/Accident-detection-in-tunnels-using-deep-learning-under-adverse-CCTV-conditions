# 🚦 Accident Detection in Tunnels Using Deep Learning Under Adverse CCTV Conditions

<p align="center">
  <strong>AI-Powered Accident Detection and Monitoring System for Tunnel CCTV Surveillance</strong>
  <br>
  An intelligent computer vision solution using YOLOv9, OpenCV, and Deep Learning to identify potential accidents in CCTV footage.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/YOLOv9-Object%20Detection-green" alt="YOLOv9">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-orange?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-red?logo=opencv" alt="OpenCV">
  <img src="https://img.shields.io/badge/Streamlit-Web%20Dashboard-FF4B4B?logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Status-Research%20Prototype-success" alt="Status">
</p>

---

## 📌 Project Overview

Road and tunnel accidents can create serious safety risks, particularly in enclosed environments where visibility may be affected by insufficient lighting, shadows, and challenging CCTV camera conditions.

**Accident Detection in Tunnels Using Deep Learning Under Adverse CCTV Conditions** is a computer vision-based project designed to detect potential accident events from tunnel surveillance footage using deep learning.

The system uses the **YOLOv9 object detection architecture** to identify accident-related objects or events in video frames. OpenCV handles video processing, while the SORT (Simple Online and Realtime Tracking) algorithm helps maintain object identities across consecutive frames.

The application also incorporates image enhancement techniques such as CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve local contrast in darker tunnel footage.

A Streamlit-based dashboard provides an interactive interface for uploading images and videos, visualizing detection results, and displaying accident alerts.

The primary objective is to explore how artificial intelligence and computer vision can support automated tunnel surveillance and improve the speed of accident identification.

## 🎯 Project Objectives

- Develop an automated accident detection system using deep learning.
- Identify potential accident-related events in tunnel CCTV footage.
- Explore object detection under challenging lighting conditions.
- Improve image contrast using CLAHE preprocessing.
- Track detected objects across consecutive video frames.
- Reduce false detections by considering detection persistence and confidence.
- Generate accident alert images for further inspection.
- Provide an interactive dashboard for analyzing surveillance footage.

## ✨ Key Features

### 1. AI-Based Accident Detection
Uses YOLOv9 to identify accident-related targets in images and video frames.

### 2. CCTV Video Processing
Processes tunnel surveillance videos frame by frame using OpenCV.

### 3. Adverse Lighting Enhancement
Uses CLAHE preprocessing in the command-line video detection pipeline to improve local image contrast in darker scenes.

### 4. Object Tracking
Integrates SORT tracking to associate detected bounding boxes across video frames and maintain object IDs.

### 5. Accident Persistence Verification
Uses repeated detections and average confidence thresholds to determine whether a detection should be considered confirmed.

### 6. Automatic Alert Generation
Saves cropped images of confirmed detections for further analysis.

### 7. Interactive Streamlit Dashboard
Provides an interface for uploading images and videos and viewing annotated detection results.

### 8. Model Training and Evaluation
Includes scripts for training the YOLOv9 model, evaluating detection metrics, and analyzing performance.

### 9. Visual Performance Analysis
Provides scripts for displaying evaluation metrics and summarizing training results.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| YOLOv9 | Deep learning-based object detection |
| PyTorch | Deep learning framework |
| Ultralytics | YOLO model training and inference |
| OpenCV | Image and video processing |
| NumPy | Numerical computations |
| SORT | Object tracking across video frames |
| Streamlit | Interactive web dashboard |
| Matplotlib | Graphical performance visualization |
| Pandas | Training result data analysis |
| Tabulate | Tabular evaluation output |
| SciPy | Mathematical and tracking operations |
| FilterPy | Kalman filtering used by SORT |

## 📋 System Requirements

### Hardware Requirements

- Desktop computer or laptop.
- Minimum 8 GB RAM recommended.
- NVIDIA GPU with CUDA support recommended for model training and faster inference.
- At least 4 GB of available GPU memory is a useful starting point for experimentation, although actual requirements depend on model size, batch size, and image resolution.
- Sufficient storage for datasets, model weights, and processed videos.
- Internet connection for downloading dependencies, pretrained weights, and the dataset.

### Software Requirements

- Python 3.11 recommended.
- PyTorch.
- Ultralytics YOLO.
- OpenCV.
- Streamlit.
- pip package manager.
- Visual Studio Code, PyCharm, or another Python IDE (optional).

**Note:** Training can run on a CPU, but it may be significantly slower. GPU performance depends on the installed PyTorch and CUDA configuration.

## 📦 Dataset Information

The project uses a custom accident detection dataset hosted on Roboflow.

**Dataset:** Accident in Tunnels

Dataset Link:

https://app.roboflow.com/mine2/accident-in-tunnels/1

Download the dataset in a YOLO-compatible format.

### Expected Dataset Structure

After downloading and extracting the dataset, its structure should resemble:

```text
Dataset/
│
├── train/
│   ├── images/
│   └── labels/
│
├── valid/
│   ├── images/
│   └── labels/
│
├── test/
│   ├── images/
│   └── labels/
│
└── data.yaml
```

### Dataset Configuration

The `data.yaml` file contains important dataset information, including:

- Training image directory.
- Validation image directory.
- Testing image directory.
- Number of detection classes.
- Class names.

Example:

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 1

names:
  0: accident
```

**Important:** The YAML example above is illustrative. Use the actual class count, class names, and paths from the Roboflow dataset you download. The current application logic should also be configured consistently with the dataset's accident class mapping.

**Note:** The dataset itself is not included in this GitHub repository and must be downloaded separately.

## 🚀 Installation and Setup

Follow these steps to configure and run the project.

### Step 1: Clone the Repository

Open your terminal or command prompt.

```bash
git clone https://github.com/Tippisrikar/Accident-detection-in-tunnels-using-deep-learning-under-adverse-CCTV-conditions.git
```

Navigate to the project folder:

```bash
cd Accident-detection-in-tunnels-using-deep-learning-under-adverse-CCTV-conditions
```

### Step 2: Create a Virtual Environment

Create a separate Python environment for the project.

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

For macOS/Linux:

```bash
source venv/bin/activate
```

### Step 3: Upgrade pip

```bash
python -m pip install --upgrade pip
```

### Step 4: Install Required Dependencies

Run the following command:

```bash
pip install ultralytics streamlit opencv-python numpy pandas matplotlib tabulate scikit-image filterpy scipy
```

These packages provide the required functionality for model training, inference, video processing, tracking, dashboard execution, and performance analysis.

**PyTorch Note:** Ultralytics installs its required PyTorch dependencies. For NVIDIA GPU acceleration, install a compatible CUDA-enabled PyTorch build using the official PyTorch installation selector before installing Ultralytics.

PyTorch installation: https://pytorch.org/get-started/locally/

### Step 5: Verify Installation

Execute:

```bash
python -c "import torch, cv2, ultralytics, streamlit; print('All core dependencies installed successfully!')"
```

To check GPU availability:

```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
```

If CUDA is available, the training script can use the detected NVIDIA GPU.

## 🏋️ Model Training

The project uses the YOLOv9c pretrained model as the starting point for custom accident detection training.

The training implementation is available in:

```text
train.py
```

### Step 1: Download the Dataset

Download the accident detection dataset from Roboflow and extract it into your preferred directory.

Locate the `data.yaml` file.

### Step 2: Configure the Dataset Path

Open `train.py`.

Locate the following configuration:

```python
data=r'E:\srikar\yolov9\Yolo_accident.v4i.yolov9\data.yaml'
```

Replace it with the actual location of your downloaded dataset.

For example:

```python
data=r'D:\AccidentDetection\Dataset\data.yaml'
```

Update the dataset path in both locations inside `train.py`.

### Step 3: Configure Training Parameters

The existing training script uses:

```python
model = YOLO('yolov9c.pt')

results = model.train(
    data=r'path/to/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='yolov9_accident_detection_run1',
    patience=30,
    workers=8
)
```

| Parameter | Description |
|---|---|
| Model | YOLOv9c pretrained model |
| Epochs | 100 training epochs |
| Image Size | 640 × 640 |
| Batch Size | 16 |
| Patience | 30 epochs |
| Workers | 8 data-loading workers |
| Device | Automatically selects CUDA when available, otherwise CPU |

Adjust the batch size and number of workers according to your hardware.

### Step 4: Start Training

Execute:

```bash
python train.py
```

The model will start training using the configured dataset.

### Step 5: Training Output

The trained model and training results are typically saved under:

```text
runs/
└── detect/
    └── yolov9_accident_detection_run1/
        ├── weights/
        │   ├── best.pt
        │   └── last.pt
        │
        ├── results.csv
        └── results.png
```

- `best.pt` – Best model checkpoint produced during training.
- `last.pt` – Model checkpoint from the final training epoch.
- `results.csv` – Training and validation metrics.
- `results.png` – Training performance plots.

The exact output location depends on the Ultralytics training configuration.

## 📊 Model Evaluation

The project includes an evaluation script:

```text
evaluate.py
```

It loads a trained YOLO model and evaluates its detection performance.

### Evaluation Command

```bash
python evaluate.py
```

The script reports the following metrics:

| Metric | Description |
|---|---|
| Precision | Proportion of predicted detections that are correct |
| Recall | Proportion of actual targets that are detected |
| mAP@0.5 | Mean Average Precision at IoU threshold 0.50 |
| mAP@0.5:0.95 | Mean Average Precision across IoU thresholds from 0.50 to 0.95 |

**Important:** Before running the evaluation script, update its model path to your generated `best.pt` checkpoint. The existing script currently references an older YOLOv8 model directory.

## 📈 Performance Analysis

The repository includes additional scripts for analyzing model performance.

### 1. Performance Analysis

File:

```text
performance_analysis.py
```

Run:

```bash
python performance_analysis.py
```

The script:

- Evaluates the trained YOLO model.
- Displays precision, recall, and mAP metrics.
- Generates a formatted evaluation table.
- Saves the table as an image.

Expected output:

```text
accident_results/
└── evaluation_table.png
```

Update the model path inside the script before running it.

### 2. Training Result Analysis

File:

```text
result_graph.py
```

This script reads the Ultralytics training results CSV file and prints a final training summary, including precision, recall, and training losses.

Update its `csv_path` variable to point to the `results.csv` generated by your training run.

**Note:** The script's reported average of precision and recall is a custom summary statistic, not standard classification accuracy.

## 🚨 Accident Detection Using Video

The project includes a video processing pipeline in:

```text
detect_video.py
```

This script uses the trained YOLO model to identify potential accidents in CCTV footage.

### Features of the Video Detection Pipeline

- Frame-by-frame video processing.
- YOLO object detection.
- CLAHE-based contrast enhancement.
- SORT object tracking when available.
- Simple IoU-based fallback tracking.
- Confidence threshold filtering.
- Detection persistence verification.
- Automatic saving of accident-related image crops.
- Annotated video generation.

### Step 1: Configure Model and Video

The script contains default paths for the trained model and input video.

You can override them directly through command-line arguments, avoiding changes to the default configuration.

### Step 2: Run Detection

Execute:

```bash
python detect_video.py --model "runs/detect/yolov9_accident_detection_run1/weights/best.pt" --source "path/to/tunnel_video.mp4"
```

You can also configure additional detection parameters:

```bash
python detect_video.py ^
--model "runs/detect/yolov9_accident_detection_run1/weights/best.pt" ^
--source "path/to/tunnel_video.mp4" ^
--conf 0.5 ^
--persist_frames 2 ^
--min_avg_conf 0.35 ^
--out_dir outputs ^
--alert_dir alerts
```

The multiline command above is for Windows Command Prompt. In PowerShell or Linux/macOS terminals, use the appropriate line-continuation character or run the command on one line.

### Step 3: View Detection Results

The system generates an annotated video containing detection bounding boxes and tracking IDs.

Expected output:

```text
outputs/
└── annotated_output_1.mp4

alerts/
└── alert_f..._id...jpg
```

Confirmed detections are highlighted and their cropped image regions are saved for inspection.

Press `q` in the OpenCV video window to stop processing early.

For systems without a display, use:

```bash
python detect_video.py --model "path/to/best.pt" --source "path/to/video.mp4" --headless
```

## 🌐 Streamlit Web Application

The project includes an interactive dashboard developed using Streamlit.

Main application:

```text
streamlit_app.py
```

### Dashboard Features

- Upload and analyze images.
- Upload CCTV video files.
- Display annotated detection results.
- Configure detection confidence thresholds.
- Configure minimum bounding-box area.
- Set the number of frames required to confirm a detection.
- Select tracking options.
- Save detected accident image crops.
- View detection information through a web interface.

### Step 1: Check the Model Path

The application looks for a trained model at:

```text
runs/detect/yolov9_accident_detection_run1/weights/best.pt
```

Ensure that your trained model exists at this location.

Alternatively, enter the correct model path in the dashboard sidebar.

### Step 2: Launch the Application

Run:

```bash
streamlit run streamlit_app.py
```

### Step 3: Access the Dashboard

Streamlit will display a local URL, usually:

```text
http://localhost:8501
```

Open that address in your web browser.

### Step 4: Choose an Input Source

The application provides three input modes:

1. **Image Upload:** Upload images for accident detection.
2. **Video Upload:** Upload CCTV videos and process their frames.
3. **Webcam:** Capture an individual image through the browser camera and analyze it.

**Note:** The current webcam option uses Streamlit's camera capture widget. It is not a continuous live CCTV or webcam streaming implementation.

## 📂 Project Structure

The repository contains the following main files:

```text
Accident-detection-in-tunnels-using-deep-learning-under-adverse-CCTV-conditions/
│
├── README.md
│
├── train.py
│   └── YOLOv9 model training
│
├── detect_video.py
│   └── Video-based accident detection pipeline
│
├── evaluate.py
│   └── Model evaluation
│
├── performance_analysis.py
│   └── Evaluation metrics and table generation
│
├── result_graph.py
│   └── Training results analysis
│
├── sort.py
│   └── SORT object tracking implementation
│
├── streamlit_app.py
│   └── Main interactive accident detection dashboard
│
└── streamlit__app_1.py
    └── Alternative video dashboard with browser-based alarm playback
```

Additional files and directories are generated during model training and inference.

**Note:** The dataset and trained model weights are maintained separately and are not included in the current repository.

## 🔄 System Workflow

The overall accident detection process follows this workflow:

```text
              START
                |
                v
        Input CCTV Footage
                |
                v
        Extract Video Frames
                |
                v
       Image Preprocessing
                |
                v
        CLAHE Enhancement
                |
                v
       YOLOv9 Object Detection
                |
                v
         Extract Bounding Boxes
                |
                v
          SORT Tracking
                |
                v
      Detection Confidence Check
                |
                v
      Persistence Verification
                |
           +----+----+
           |         |
           v         v
     Detection     No Confirmed
     Confirmed     Detection
           |         |
           v         v
      Generate     Continue
      Alert Crop   Processing
           |
           v
      Annotated Video
           |
           v
       Save Results
           |
           v
           END
```

## 🔍 Troubleshooting

| Problem | Possible Solution |
|---|---|
| YOLO model not found | Verify the path to `best.pt`. |
| Dataset not found | Check the `data.yaml` path in `train.py`. |
| CUDA not available | Install a PyTorch build compatible with your GPU and CUDA environment. |
| Out of memory during training | Reduce batch size or image resolution. |
| Video cannot be opened | Verify the video format and file path. |
| Streamlit model loading error | Confirm that the trained model exists at the configured location. |
| Missing Python modules | Activate the virtual environment and reinstall dependencies. |
| SORT import error | Check installation of `filterpy`, `scipy`, `scikit-image`, and `matplotlib`. |
| No accident detections | Check the trained model, class mapping, confidence threshold, and input video quality. |
| Incorrect evaluation results | Confirm that the evaluation script uses the correct trained model and dataset configuration. |

## ⚠️ Important Implementation Notes

- The repository does not contain pretrained accident-specific weights. Train the model or provide an appropriate trained checkpoint before running detection.
- The training script currently contains a machine-specific Windows dataset path that must be updated.
- The video detection script also has machine-specific default model and video paths. Use its `--model` and `--source` arguments to configure your own files.
- The evaluation and performance analysis scripts reference an older `accident_results/accident_yolov8_model` location. Update those paths to match the model you train.
- The video detection pipeline treats detected boxes as accident-related targets without explicitly filtering class IDs. Ensure the model and class configuration match the intended accident detection task.
- The alternate Streamlit application `streamlit__app_1.py` contains hard-coded model and alarm file paths that must be updated before use.
- CLAHE preprocessing is enabled by default in `detect_video.py`; the main Streamlit application does not apply that same preprocessing.
- The current project is an experimental computer vision prototype. Its detections and alerts should not be treated as guaranteed accident identification or as a replacement for certified safety monitoring and human supervision.

## 🔐 Privacy and Safety

- Process only CCTV footage that you are authorized to use.
- Protect uploaded video files and generated accident images.
- Avoid exposing surveillance footage containing identifiable individuals.
- Use the application as a research and decision-support prototype.
- Human verification should be performed before any emergency response is initiated based on a model prediction.

## 🚀 Future Enhancements

Potential improvements for the project include:

- Real-time integration with tunnel CCTV camera streams.
- Automatic emergency notifications to monitoring authorities.
- Improved detection accuracy under low-light and adverse weather conditions.
- Advanced image enhancement and noise reduction.
- Integration with additional deep learning architectures.
- Improved object tracking and multi-camera tracking.
- Real-time dashboard updates.
- Accident severity classification.
- Automatic incident reporting and timestamp logging.
- Deployment on edge devices for low-latency processing.
- Model optimization for faster inference.
- Evaluation on larger and more diverse tunnel CCTV datasets.

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome.

To contribute:

1. Fork this repository.
2. Create a new feature branch.
3. Implement your modifications.
4. Commit your changes with a meaningful message.
5. Push your branch.
6. Submit a pull request.

Bug reports and feature suggestions are appreciated.

## 📄 License

No repository-level LICENSE file is currently included.

The bundled `sort.py` implementation contains a GNU GPL v3-or-later license notice. Review its licensing terms and ensure appropriate licensing information is included before redistributing the complete project.

## 👨‍💻 Author

**Tippi Srikar**

- GitHub: [@Tippisrikar](https://github.com/Tippisrikar)
- Project Repository: [Accident Detection in Tunnels Using Deep Learning](https://github.com/Tippisrikar/Accident-detection-in-tunnels-using-deep-learning-under-adverse-CCTV-conditions)

---

<p align="center">
  <strong>Built with Python ❤️, Deep Learning, and Computer Vision.</strong>
  <br>
  Towards Smarter and Safer Tunnel Surveillance! 🚦
</p>
