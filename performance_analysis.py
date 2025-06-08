from ultralytics import YOLO
from tabulate import tabulate
import matplotlib.pyplot as plt
import os

def evaluate_model():
    # Load trained model
    model = YOLO("accident_results/accident_yolov8_model/weights/best.pt")

    # Evaluate model on validation set
    results = model.val()

    # Collect evaluation metrics
    metrics = [
        ["mAP@0.5", f"{results.box.map50:.4f}"],
        ["mAP@0.5:0.95", f"{results.box.map:.4f}"],
        ["Mean Precision", f"{results.box.mp:.4f}"],
        ["Mean Recall", f"{results.box.mr:.4f}"]
    ]

    # Print in console
    print("\n[INFO] Evaluation Metrics:\n")
    print(tabulate(metrics, headers=["Metric", "Value"], tablefmt="fancy_grid"))

    # Save as image
    save_path = "accident_results/evaluation_table.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=metrics, colLabels=["Metric", "Value"], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    plt.title("YOLOv8 Evaluation Metrics", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n[INFO] Table image saved to: {save_path}")

if __name__ == "__main__":
    evaluate_model()
