# detect_video.py
# Improved, robust accident detection runner for YOLOv9/ultralytics models.
# - Includes defaults inside the script (so you can run without passing args)
# - Handles different ultralytics result formats safely
# - Uses SORT if available, otherwise a simple IoU-based tracker fallback
# - Adds CLAHE preprocessing to help tunnel lighting

import os
import cv2
import argparse
import numpy as np
from ultralytics import YOLO
import time
import math
import typing as t

# ---------- Configurable defaults (edit these paths if you want) ----------
DEFAULT_MODEL = r"D:\srikar\Projects\Projects\yolov9\runs\detect\yolov9_accident_detection_run1\weights\best.pt"
DEFAULT_VIDEO = r"D:\srikar\Projects\Projects\cctv project\videos\video1.mp4"
DEFAULT_OUT_DIR = "outputs"
DEFAULT_ALERT_DIR = "alerts"
# -------------------------------------------------------------------------

# Try to import SORT; if missing, provide a lightweight fallback tracker
try:
    from sort import Sort  # SORT tracker (https://github.com/abewley/sort)
    SORT_AVAILABLE = True
except Exception:
    SORT_AVAILABLE = False

    class SimpleTracker:
        """Minimal IoU-based tracker fallback.
        Not as sophisticated as SORT but useful when SORT isn't installed.
        Assigns persistent integer IDs and tries to match boxes by IoU.
        """
        def __init__(self, max_age=15, iou_threshold=0.3):
            self.max_age = max_age
            self.iou_threshold = iou_threshold
            self.tracks = {}  # id -> {'bbox': [x1,y1,x2,y2], 'age':0, 'missed':0}
            self._next_id = 1

        def iou(self, a, b):
            # a & b: [x1,y1,x2,y2]
            xa1, ya1, xa2, ya2 = a
            xb1, yb1, xb2, yb2 = b
            xi1 = max(xa1, xb1)
            yi1 = max(ya1, yb1)
            xi2 = min(xa2, xb2)
            yi2 = min(ya2, yb2)
            wi = max(0, xi2 - xi1)
            hi = max(0, yi2 - yi1)
            inter = wi * hi
            a_area = max(0, (xa2 - xa1)) * max(0, (ya2 - ya1))
            b_area = max(0, (xb2 - xb1)) * max(0, (yb2 - yb1))
            union = a_area + b_area - inter
            return inter / union if union > 0 else 0.0

        def update(self, detections: np.ndarray):
            """
            detections: Nx5 array [x1,y1,x2,y2,score]
            returns: Mx5 array [x1,y1,x2,y2,track_id]
            """
            dets = detections.tolist() if detections is not None and len(detections) else []
            assigned = set()
            out_tracks = []

            # Try match detections to existing tracks by IoU
            for tid, tr in list(self.tracks.items()):
                tr['missed'] += 1

            for det in dets:
                db = det[:4]
                best_id = None
                best_iou = 0.0
                for tid, tr in self.tracks.items():
                    i = self.iou(db, tr['bbox'])
                    if i > best_iou:
                        best_iou = i
                        best_id = tid
                if best_iou >= self.iou_threshold and best_id is not None:
                    # update track
                    self.tracks[best_id]['bbox'] = db
                    self.tracks[best_id]['missed'] = 0
                    self.tracks[best_id]['age'] += 1
                    out_tracks.append([int(db[0]), int(db[1]), int(db[2]), int(db[3]), int(best_id)])
                else:
                    # new track
                    new_id = self._next_id
                    self._next_id += 1
                    self.tracks[new_id] = {'bbox': db, 'age': 1, 'missed': 0}
                    out_tracks.append([int(db[0]), int(db[1]), int(db[2]), int(db[3]), int(new_id)])

            # remove old tracks
            to_delete = [tid for tid, tr in self.tracks.items() if tr['missed'] > self.max_age]
            for tid in to_delete:
                del self.tracks[tid]

            return np.array(out_tracks, dtype=int)


def create_tracker(max_age=15, min_hits=3, iou_threshold=0.3):
    if SORT_AVAILABLE:
        try:
            return Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
        except Exception:
            return SimpleTracker(max_age=max_age, iou_threshold=iou_threshold)
    else:
        return SimpleTracker(max_age=max_age, iou_threshold=iou_threshold)


# ------------------ Utility functions ------------------
def apply_clahe_bgr(frame: np.ndarray) -> np.ndarray:
    """Apply CLAHE to each channel (helps dark tunnel frames)."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def xyxy_to_list(xyxy_arr: np.ndarray):
    """Convert an Nx4 array to Python list of tuples (x1,y1,x2,y2)."""
    return [tuple(map(float, box)) for box in xyxy_arr]


def ensure_int_box(box):
    x1, y1, x2, y2 = box
    return int(max(0, round(x1))), int(max(0, round(y1))), int(round(x2)), int(round(y2))


# ------------------ Main detection pipeline ------------------
def run_detection(
    model_path: str = DEFAULT_MODEL,
    source: str = DEFAULT_VIDEO,
    out_dir: str = DEFAULT_OUT_DIR,
    alert_dir: str = DEFAULT_ALERT_DIR,
    conf_thresh: float = 0.5,
    persist_frames: int = 2,
    min_avg_conf: float = 0.35,
    min_area_ratio: float = 0.001,
    max_age: int = 20,
    min_hits: int = 1,
    iou_threshold: float = 0.15,
    use_clahe: bool = True,
    headless: bool = False
):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(alert_dir, exist_ok=True)

    print(f"🔎 Loading model: {model_path}")
    model = YOLO(model_path)

    print("🧭 Creating tracker...")
    tracker = create_tracker(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open source: {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 1 else 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.join(out_dir, "annotated_output_1.mp4")
    out_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_id = 0
    accident_memory: dict = {}  # track_id -> list[(frame_id, conf)]
    last_time = time.time()
    print(f"▶ Starting processing: {source} -> {out_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        proc_frame = frame.copy()
        if use_clahe:
            try:
                proc_frame = apply_clahe_bgr(proc_frame)
            except Exception:
                proc_frame = frame.copy()

        # Run model inference
        try:
            # ultralytics supports calling model on numpy BGR directly
            results = model(proc_frame, conf=conf_thresh, verbose=False)[0]
        except Exception as e:
            print("⚠️ Model inference exception:", e)
            break

        # Extract boxes reliably (works across ultralytics versions)
        detections = []  # format for tracker: [x1, y1, x2, y2, score]
        try:
            # prefer boxes.xyxy, boxes.conf
            if hasattr(results, "boxes") and results.boxes is not None and len(results.boxes) > 0:
                # boxes.xyxy might be a tensor or list
                try:
                    xyxy = results.boxes.xyxy.cpu().numpy()
                    confs = results.boxes.conf.cpu().numpy()
                except Exception:
                    # fallback shapes
                    xyxy = np.array([b[:4] for b in results.boxes.cpu().numpy()])
                    confs = np.array([b[4] if len(b) >= 5 else 0.0 for b in results.boxes.cpu().numpy()])

                for (x1, y1, x2, y2), conf in zip(xyxy, confs):
                    # Apply confidence filter (already applied by model(..., conf=conf_thresh), but keep safe)
                    if conf < conf_thresh:
                        continue
                    # Clip boxes
                    x1 = max(0, min(x1, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    x2 = max(0, min(x2, width - 1))
                    y2 = max(0, min(y2, height - 1))
                    # Some frameworks return degenerate boxes - skip
                    if x2 <= x1 or y2 <= y1:
                        continue
                    detections.append([float(x1), float(y1), float(x2), float(y2), float(conf)])
        except Exception as e:
            print("⚠ Error parsing detections:", e)
            detections = []

        dets_np = np.array(detections) if len(detections) else np.empty((0, 5))

        # Tracker update -> tracked_objects: Nx5 [x1,y1,x2,y2,id]
        try:
            tracked_objects = tracker.update(dets_np)
        except Exception as e:
            # In case SORT expects float32 etc.
            try:
                tracked_objects = tracker.update(dets_np.astype(np.float32))
            except Exception as e2:
                print("⚠ Tracker update error:", e2)
                tracked_objects = np.empty((0, 5))

        # For each tracked object, decide accident persistence
        for to in tracked_objects:
            # Some trackers (SimpleTracker) return ints already
            try:
                tx1, ty1, tx2, ty2, tid = to
            except Exception:
                # If different shape, skip
                continue
            x1, y1, x2, y2 = map(int, (tx1, ty1, tx2, ty2))
            track_id = int(tid)

            # compute area filter (to ignore very small detections)
            area = max(0, (x2 - x1)) * max(0, (y2 - y1))
            frame_area = max(1, width * height)
            if area < (min_area_ratio * frame_area):
                # small detection - draw faint rectangle and continue
                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 0), 1)
                cv2.putText(frame, f"tiny {track_id}", (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)
                continue

            # find the confidence associated with this bounding box (best match from detections)
            conf_val = 0.0
            if len(detections) > 0:
                # find det with largest IoU
                best_iou = 0.0
                best_conf = 0.0
                for det in detections:
                    iou = 0.0
                    # compute IoU
                    ix1 = max(x1, int(det[0])); iy1 = max(y1, int(det[1]))
                    ix2 = min(x2, int(det[2])); iy2 = min(y2, int(det[3]))
                    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
                    inter = iw * ih
                    a_area = (x2 - x1) * (y2 - y1)
                    b_area = max(1, (int(det[2]) - int(det[0])) * (int(det[3]) - int(det[1])))
                    union = a_area + b_area - inter
                    if union > 0:
                        iou = inter / union
                    if iou > best_iou:
                        best_iou = iou
                        best_conf = det[4]
                conf_val = float(best_conf) if best_iou > 0 else 0.0

            # Track memory
            if track_id not in accident_memory:
                accident_memory[track_id] = []
            accident_memory[track_id].append((frame_id, conf_val))
            # Keep last N frames in memory (cap)
            accident_memory[track_id] = accident_memory[track_id][-30:]

            # Decide confirmed accident based on persistence & avg confidence
            confirmed = False
            if len(accident_memory[track_id]) >= persist_frames:
                avg_conf = np.mean([c for _, c in accident_memory[track_id]])
                if avg_conf >= min_avg_conf:
                    confirmed = True

            # Draw boxes and labels
            if confirmed:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, f"ACCIDENT ID {track_id} (avg:{avg_conf:.2f})",
                            (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                # Save alert crop (if not saved recently)
                ts = int(time.time())
                alert_path = os.path.join(alert_dir, f"alert_f{frame_id}_id{track_id}_{ts}.jpg")
                try:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        cv2.imwrite(alert_path, crop)
                except Exception:
                    pass
            else:
                # suspicious / tracked but not confirmed
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                last_conf = accident_memory[track_id][-1][1] if len(accident_memory[track_id]) else 0.0
                cv2.putText(frame, f"{track_id} {last_conf:.2f}",
                            (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Write annotated frame
        out_writer.write(frame)

        # Optionally show
        if not headless:
            cv2.imshow("Accident Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("🛑 Stopped by user (q).")
                break

        # status print every 200 frames
        if frame_id % 200 == 0:
            cur_time = time.time()
            elapsed = cur_time - last_time
            last_time = cur_time
            print(f"Processed {frame_id} frames. (approx fps {200/elapsed:.1f})")

    cap.release()
    out_writer.release()
    if not headless:
        cv2.destroyAllWindows()
    print(f"✅ Finished. Annotated video saved to: {out_path}")
    print(f"✅ Alerts saved to: {os.path.abspath(alert_dir)}")


# ------------------ CLI (but defaults embedded) ------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect accidents in a video using YOLO (YOLOv9/ultralytics).")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to YOLO .pt model (default in script)")
    parser.add_argument("--source", default=DEFAULT_VIDEO, help="Path to input video file (default in script)")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="Where annotated video will be saved")
    parser.add_argument("--alert_dir", default=DEFAULT_ALERT_DIR, help="Where alert crops will be saved")
    parser.add_argument("--conf", type=float, default=0.10, help="Confidence threshold (lower = more sensitive)")
    parser.add_argument("--persist_frames", type=int, default=2, help="Frames required to confirm accident")
    parser.add_argument("--min_avg_conf", type=float, default=0.35, help="Minimum average confidence to confirm accident")
    parser.add_argument("--min_area_ratio", type=float, default=0.001, help="Minimum bbox area ratio relative to frame")
    parser.add_argument("--max_age", type=int, default=20, help="Tracker max age")
    parser.add_argument("--min_hits", type=int, default=1, help="Tracker min hits")
    parser.add_argument("--iou_threshold", type=float, default=0.15, help="Tracker IOU threshold")
    parser.add_argument("--no_clahe", dest="no_clahe", action="store_true", help="Disable CLAHE preprocessing")
    parser.add_argument("--headless", action="store_true", help="Run without display")
    args = parser.parse_args()

    run_detection(
        model_path=args.model,
        source=args.source,
        out_dir=args.out_dir,
        alert_dir=args.alert_dir,
        conf_thresh=args.conf,
        persist_frames=args.persist_frames,
        min_avg_conf=args.min_avg_conf,
        min_area_ratio=args.min_area_ratio,
        max_age=args.max_age,
        min_hits=args.min_hits,
        iou_threshold=args.iou_threshold,
        use_clahe=not args.no_clahe,
        headless=args.headless
    )
