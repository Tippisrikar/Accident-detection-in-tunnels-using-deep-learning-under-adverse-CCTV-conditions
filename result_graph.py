'''import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv(r"D:\srikar\Projects\yolov9\runs\detect\yolov9_accident_detection_run1\results.csv")

# Clean column names (remove leading/trailing spaces)
df.columns = df.columns.str.strip()

print("Columns in CSV:", df.columns.tolist())  # debug check

# Plot losses
plt.figure(figsize=(10,6))
plt.plot(df["epoch"], df["train/box_loss"], label="Box Loss")
plt.plot(df["epoch"], df["train/cls_loss"], label="Cls Loss")
plt.plot(df["epoch"], df["train/dfl_loss"], label="Dfl Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Losses")
plt.legend()
plt.grid()
plt.show()

# Plot metrics
plt.figure(figsize=(10,6))
plt.plot(df["epoch"], df["metrics/precision(B)"], label="Precision")
plt.plot(df["epoch"], df["metrics/recall(B)"], label="Recall")
plt.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP50")
plt.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP50-95")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Validation Metrics")
plt.legend()
plt.grid()
plt.show()'''


'''import pandas as pd
import matplotlib.pyplot as plt

# 📌 Path to your results.csv file
csv_path = r"D:\srikar\Projects\Projects\yolov9\runs\detect\yolov9_accident_detection_run1\results.csv"

df = pd.read_csv(csv_path)

# Calculate accuracy
df["train_accuracy"] = (df["metrics/precision(B)"] + df["metrics/recall(B)"]) / 2
df["val_accuracy"] = (df["metrics/mAP50(B)"] + df["metrics/mAP50-95(B)"]) / 2

# Use a safe built-in style
plt.style.use("ggplot")

# Create a 3×2 grid
fig, axs = plt.subplots(3, 2, figsize=(15, 14))
fig.suptitle("YOLOv9 Training Summary (with Accuracy Curves)", fontsize=16, fontweight='bold')

# -------------------------------
# 1️⃣ TRAIN LOSSES
# -------------------------------
axs[0, 0].plot(df["epoch"], df["train/box_loss"], label="Box Loss")
axs[0, 0].plot(df["epoch"], df["train/cls_loss"], label="Cls Loss")
axs[0, 0].plot(df["epoch"], df["train/dfl_loss"], label="DFL Loss")
axs[0, 0].set_title("Training Loss")
axs[0, 0].legend()

# -------------------------------
# 2️⃣ VALIDATION LOSSES
# -------------------------------
axs[0, 1].plot(df["epoch"], df["val/box_loss"], label="Box Loss")
axs[0, 1].plot(df["epoch"], df["val/cls_loss"], label="Cls Loss")
axs[0, 1].plot(df["epoch"], df["val/dfl_loss"], label="DFL Loss")
axs[0, 1].set_title("Validation Loss")
axs[0, 1].legend()

# -------------------------------
# 3️⃣ TRAINING ACCURACY
# -------------------------------
axs[1, 0].plot(df["epoch"], df["train_accuracy"], label="Train Accuracy", color="blue")
axs[1, 0].set_title("Training Accuracy")
axs[1, 0].set_ylim(0, 1)
axs[1, 0].legend()

# -------------------------------
# 4️⃣ VALIDATION ACCURACY
# -------------------------------
axs[1, 1].plot(df["epoch"], df["val_accuracy"], label="Validation Accuracy", color="green")
axs[1, 1].set_title("Validation Accuracy")
axs[1, 1].set_ylim(0, 1)
axs[1, 1].legend()

# -------------------------------
# 5️⃣ METRICS (Precision, Recall, mAPs)
# -------------------------------
axs[2, 0].plot(df["epoch"], df["metrics/precision(B)"], label="Precision")
axs[2, 0].plot(df["epoch"], df["metrics/recall(B)"], label="Recall")
axs[2, 0].plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP50")
axs[2, 0].plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP50-95")
axs[2, 0].set_title("Performance Metrics")
axs[2, 0].legend()

# -------------------------------
# 6️⃣ LEARNING RATE CURVES
# -------------------------------
axs[2, 1].plot(df["epoch"], df["lr/pg0"], label="LR pg0")
axs[2, 1].plot(df["epoch"], df["lr/pg1"], label="LR pg1")
axs[2, 1].plot(df["epoch"], df["lr/pg2"], label="LR pg2")
axs[2, 1].set_title("Learning Rates")
axs[2, 1].legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save output image
save_path = r"D:\srikar\Projects\Projects\yolov9\training_summary_with_accuracy.png"
plt.savefig(save_path, dpi=300)
plt.show()

print(f"✔ Training summary with accuracy saved to:\n{save_path}")'''

import pandas as pd

# Path to your results.csv
csv_path = r"D:\srikar\Projects\Projects\yolov9\runs\detect\yolov9_accident_detection_run1\results.csv"

df = pd.read_csv(csv_path)

# Get last epoch row
last = df.iloc[-1]

# Overall training accuracy
train_accuracy = (last["metrics/precision(B)"] + last["metrics/recall(B)"]) / 2

# Overall training loss
train_loss = last["train/box_loss"] + last["train/cls_loss"] + last["train/dfl_loss"]

print("\n===== FINAL TRAINING METRICS =====")
print(f"✔ Overall Training Accuracy : {train_accuracy:.4f}")
print(f"✔ Overall Training Loss     : {train_loss:.4f}")

# Optional: show values individually
print("\n--- Breakdown ---")
print(f"Precision : {last['metrics/precision(B)']:.4f}")
print(f"Recall    : {last['metrics/recall(B)']:.4f}")
print(f"Box Loss  : {last['train/box_loss']:.4f}")
print(f"Cls Loss  : {last['train/cls_loss']:.4f}")
print(f"DFL Loss  : {last['train/dfl_loss']:.4f}")

