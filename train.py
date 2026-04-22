from ultralytics import YOLO
import torch

def train_accident_detector():
    """
    Main function to train the YOLO model for accident detection.
    """
    # Check if a GPU is available and set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- 1. CHOOSE YOUR MODEL ---
    # Load a pre-trained model. 
    # Use 'yolov9c.pt' for YOLOv9 (recommended for performance).
    # Use 'yolov8n.pt' for YOLOv8 nano (faster but less accurate).
    model = YOLO('yolov9c.pt')
    model.to(device) # Move the model to the selected device (GPU/CPU)

    # --- 2. CONFIGURE TRAINING PARAMETERS ---
    # The 'data' parameter should point to your 'data.yaml' file from Roboflow.
    # Adjust epochs, imgsz, and batch size as needed.
    results = model.train(
        data=r'E:\srikar\yolov9\Yolo_accident.v4i.yolov9\data.yaml',  # <-- IMPORTANT: SET THIS PATH
        epochs=100,
        imgsz=640,
        batch=16,  # Adjust based on your GPU memory (e.g., 8, 16, 32)
        name='yolov9_accident_detection_run1', # Name for the output folder
        patience=30, # Stop training if no improvement after 30 epochs
        workers=8    # Number of threads for data loading
    )

    print("Training complete!")
    print("Results saved in the 'runs/detect/' directory.")

    # --- 3. (OPTIONAL) VALIDATE ON THE TEST SET ---
    # After training, it's good practice to evaluate the final model on the test set.
    # We load the 'best.pt' weights that were saved during training.
    print("Validating the model on the test set...")
    best_model = YOLO('runs/detect/yolov9_accident_detection_run1/weights/best.pt')
    
    # Run validation
    metrics = best_model.val(
        data=r'E:\srikar\yolov9\Yolo_accident.v4i.yolov9\data.yaml',
        split='test'  # Specify that we are using the test set
    )

    print("Validation metrics on the test set:")
    print(f"  mAP@50-95: {metrics.box.map:.4f}")
    print(f"  mAP@50: {metrics.box.map50:.4f}")
    print(f"  Precision: {metrics.box.p[0]:.4f}") # Precision for the first class
    print(f"  Recall: {metrics.box.r[0]:.4f}") # Recall for the first class

if __name__ == '__main__':
    train_accident_detector()
