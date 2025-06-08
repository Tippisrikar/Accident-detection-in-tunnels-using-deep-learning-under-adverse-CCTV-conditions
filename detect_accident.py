from ultralytics import YOLO
import cv2
import os

def detect_accidents(image_or_video_path, save_output=True):
    # Load the trained model
    model = YOLO(r"D:\srikar\Projects\cctv project\Yolo_project\accident_results\accident_yolov8_model\weights\best.pt")

    # Create output directory
    output_dir = "runs/detect/accident_detection"
    os.makedirs(output_dir, exist_ok=True)

    # Check if input is an image or video
    if image_or_video_path.lower().endswith((".jpg", ".jpeg", ".png")):
        results = model.predict(source=image_or_video_path, save=save_output, conf=0.5)
        print(f"[INFO] Detection complete. Results saved to {output_dir}")
    elif image_or_video_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        results = model.predict(source=image_or_video_path, save=save_output, conf=0.5, stream=False)
        print(f"[INFO] Video detection complete. Results saved to {output_dir}")
    else:
        print("[ERROR] Unsupported file format. Please provide an image or video.")

if __name__ == '__main__':
    detect_accidents("path/to/your/image_or_video")
