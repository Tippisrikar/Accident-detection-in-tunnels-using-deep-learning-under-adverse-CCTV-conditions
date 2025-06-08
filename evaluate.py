from ultralytics import YOLO

def evaluate_model():
    # Load trained model
    model = YOLO("accident_results/accident_yolov8_model/weights/best.pt")

    # Evaluate model on validation set
    results = model.val()

    # Print metrics correctly (removed parentheses since these are now properties)
    print(f"mAP@0.5: {results.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {results.box.map:.4f}")
    print(f"Mean Precision: {results.box.mp:.4f}") 
    print(f"Mean Recall: {results.box.mr:.4f}")

    print("[INFO] Evaluation complete.")

if __name__ == "__main__":
    evaluate_model()

'''import pandas as pd
import matplotlib.pyplot as plt
import os
from tabulate import tabulate

def plot_metrics(csv_path, output_dir="accident_results"):
    df = pd.read_csv(csv_path)

    # Prepare final metrics
    final_metrics = {
        "Epoch": int(df["epoch"].iloc[-1]) + 1,
        "Train Box Loss": df["train/box_loss"].iloc[-1],
        "Train Class Loss": df["train/cls_loss"].iloc[-1],
        "Val Box Loss": df["val/box_loss"].iloc[-1],
        "Val Class Loss": df["val/cls_loss"].iloc[-1],
        "Precision": df["metrics/precision(B)"].iloc[-1],
        "Recall": df["metrics/recall(B)"].iloc[-1],
        "mAP@0.5": df["metrics/mAP50(B)"].iloc[-1],
        "mAP@0.5:0.95": df["metrics/mAP50-95(B)"].iloc[-1]
    }

    # Save table image
    os.makedirs(output_dir, exist_ok=True)
    table_data = [[k, f"{v:.4f}" if isinstance(v, float) else v] for k, v in final_metrics.items()]
    table_img_path = os.path.join(output_dir, "training_metrics_table.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=table_data, colLabels=["Metric", "Value"], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    plt.title("Final Training Metrics", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(table_img_path)
    print(f"[INFO] Table saved at: {table_img_path}")

    # Plot graphs
    graph_path = os.path.join(output_dir, "training_metrics_graphs.png")
    plt.figure(figsize=(15, 6))

    # Plot mAP, Precision, Recall
    plt.subplot(1, 3, 1)
    plt.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP@0.5", marker='o')
    plt.plot(df["epoch"], df["metrics/precision(B)"], label="Precision", marker='s')
    plt.plot(df["epoch"], df["metrics/recall(B)"], label="Recall", marker='^')
    plt.title("Evaluation Metrics per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.grid()
    plt.legend()

    # Plot losses
    plt.subplot(1, 3, 2)
    plt.plot(df["epoch"], df["train/box_loss"], label="Train Box Loss", color="red", marker='x')
    plt.plot(df["epoch"], df["val/box_loss"], label="Val Box Loss", color="orange", marker='x')
    plt.plot(df["epoch"], df["train/cls_loss"], label="Train Class Loss", color="purple", marker='x')
    plt.plot(df["epoch"], df["val/cls_loss"], label="Val Class Loss", color="green", marker='x')
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()

    # Plot approximate accuracy (Precision × Recall)
    plt.subplot(1, 3, 3)
    accuracy = df["metrics/precision(B)"] * df["metrics/recall(B)"]
    plt.plot(df["epoch"], accuracy, label="Approx Accuracy", color="blue", marker='D')
    plt.title("Approx Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (P × R)")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig(graph_path)
    print(f"[INFO] Graphs saved at: {graph_path}")
    plt.show()

if __name__ == "__main__":
    csv_path = "accident_results/accident_yolov8_model/results.csv"  # Adjust if needed
    plot_metrics(csv_path)'''
