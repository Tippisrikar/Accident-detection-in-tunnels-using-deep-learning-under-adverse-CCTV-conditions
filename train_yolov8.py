from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
import os

def train_model():
    # Load model - yolov8s is good for 4GB VRAM
    model = YOLO("yolov8s.pt")  # Using 's' (small) variant for your GPU
    
    # Training parameters optimized for RTX 3050 4GB
    results = model.train(
        data=r"D:\srikar\Projects\cctv project\Yolo_project\shivayya.v1i.yolov8\data.yaml",
        epochs=50,  # Increased from 30 for better convergence
        imgsz=640,  # Can reduce to 480 or 320 if you get CUDA out of memory
        batch=4,    # Reduced from 16 to fit 4GB VRAM
        device=0,
        workers=4,  # Fewer parallel data loading processes
        name="accident_yolov8_model",
        project="accident_results",
        save=True,
        verbose=True,
        augment=True,  # Enable data augmentation
        lr0=0.01,     # Learning rate
        patience=20,   # Early stopping patience
        hsv_h=0.015,   # Color augmentation
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,    # Vertical flip probability
        fliplr=0.5,    # Horizontal flip probability
        degrees=10.0,  # Rotation augmentation
        translate=0.1, # Translation augmentation
    )

    # Step 3: Extract training metrics
    metrics = results.results_dict  # This gets all the epoch-wise metrics
    csv_path = os.path.join("accident_results", "accident_yolov8_model", "metrics.csv")
    df = pd.DataFrame(metrics)
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Metrics saved to: {csv_path}")

    # Step 4: Plot training metrics
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(df["metrics/mAP_0.5"], label="mAP@0.5", marker='o')
    plt.title("mAP@0.5 per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("mAP@0.5")
    plt.grid()

    plt.subplot(1, 3, 2)
    plt.plot(df["metrics/precision"], label="Precision", marker='s', color='orange')
    plt.title("Precision per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.grid()

    plt.subplot(1, 3, 3)
    plt.plot(df["metrics/recall"], label="Recall", marker='^', color='green')
    plt.title("Recall per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.grid()

    plt.tight_layout()
    graph_path = os.path.join("accident_results", "accident_yolov8_model", "evaluation_graphs.png")
    plt.savefig(graph_path)
    plt.show()

if __name__ == '__main__':
    train_model()