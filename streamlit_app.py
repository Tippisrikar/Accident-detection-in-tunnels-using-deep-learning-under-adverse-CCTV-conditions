'''# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from pathlib import Path
from ultralytics import YOLO

# Try to import SORT; fallback to a simple centroid tracker if not present
try:
    from sort import Sort
    SORT_AVAILABLE = True
except Exception:
    SORT_AVAILABLE = False

st.set_page_config(page_title="Accident Detection (YOLO)", layout="wide")

# -----------------------
# Helper trackers
# -----------------------
class SimpleCentroidTracker:
    """Very simple tracker: assigns IDs based on nearest centroid (not robust, but works if SORT absent)."""
    def __init__(self, max_lost=15):
        self.next_id = 1
        self.objects = {}  # id -> (centroid, lost_frames)
        self.max_lost = max_lost

    def update(self, detections):
        # detections: Nx5 array [x1,y1,x2,y2,score]
        centers = []
        for d in detections:
            x1, y1, x2, y2, score = d
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            centers.append((cx, cy))
        new_objects = {}
        assigned = set()

        for oid, (centroid, lost) in list(self.objects.items()):
            # find nearest detection
            best_idx = None
            best_dist = None
            for i, c in enumerate(centers):
                if i in assigned:
                    continue
                dist = (centroid[0] - c[0]) ** 2 + (centroid[1] - c[1]) ** 2
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_idx = i
            if best_idx is not None and best_dist < 1e6:  # some threshold
                # update object
                new_objects[oid] = (centers[best_idx], 0)
                assigned.add(best_idx)
            else:
                # mark lost
                if lost + 1 <= self.max_lost:
                    new_objects[oid] = (centroid, lost + 1)
                # else drop
        # create new objects for unassigned detections
        for i, c in enumerate(centers):
            if i not in assigned:
                new_objects[self.next_id] = (c, 0)
                self.next_id += 1
        self.objects = new_objects

        # return list of tracks in form [[x1,y1,x2,y2, id], ...]
        tracks = []
        # We don't have mapping to boxes here; so approximate by using the detection boxes order again
        for i, d in enumerate(detections):
            # find the id whose centroid is closest to detection center
            cx = (d[0] + d[2]) / 2
            cy = (d[1] + d[3]) / 2
            best_id, best_dist = None, None
            for oid, (centroid, lost) in self.objects.items():
                dist = (centroid[0] - cx) ** 2 + (centroid[1] - cy) ** 2
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_id = oid
            if best_id is not None:
                tracks.append([int(d[0]), int(d[1]), int(d[2]), int(d[3]), int(best_id)])
        return np.array(tracks)


# -----------------------
# Utility functions
# -----------------------
def safe_get_boxes(results):
    """
    Returns list of detections in format [[x1,y1,x2,y2,conf,cls], ...]
    Defensive: handles ultralytics results variations.
    """
    detections = []
    if results is None:
        return detections
    # results may be a Results object (single). Try common attributes
    try:
        boxes = results.boxes  # object
        # xyxy & conf & cls are usual
        if hasattr(boxes, "xyxy") and hasattr(boxes, "conf"):
            xyxy = boxes.xyxy.cpu().numpy()  # Nx4
            confs = boxes.conf.cpu().numpy()
            try:
                classes = boxes.cls.cpu().numpy()
            except Exception:
                classes = [0] * len(confs)
            for i in range(len(confs)):
                x1, y1, x2, y2 = xyxy[i]
                conf = float(confs[i])
                cls = int(classes[i]) if len(classes) > i else 0
                detections.append([float(x1), float(y1), float(x2), float(y2), conf, cls])
            return detections
        # fallback: boxes.xywh or boxes.data
        if hasattr(boxes, "data"):
            arr = boxes.data.cpu().numpy()  # sometimes shape Nx6 where last columns are conf/class
            # try to parse typical layout (x1,y1,x2,y2,conf,cls)
            for row in arr:
                if len(row) >= 6:
                    detections.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), int(row[5])])
            return detections
    except Exception:
        pass
    # if nothing matched, return empty
    return detections

def draw_boxes(frame, tracks, labels=None):
    """Draw detections/tracks on frame. tracks: list of [x1,y1,x2,y2,id]"""
    for tr in tracks:
        x1, y1, x2, y2, tid = map(int, tr)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"ID:{tid}"
        if labels and tid in labels:
            text += f" {labels[tid]}"
        cv2.putText(frame, text, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return frame

def crop_and_save(frame, box, out_dir, prefix="alert"):
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    fname = os.path.join(out_dir, f"{prefix}_{int(time.time())}_{np.random.randint(1e6)}.jpg")
    cv2.imwrite(fname, crop)
    return fname

# -----------------------
# UI - Sidebar
# -----------------------
st.title("🚨 Accident Detection App (Streamlit + YOLO)")

st.sidebar.header("Model & thresholds")
model_path = st.sidebar.text_input("Model path (.pt)", value=str(Path.cwd() / "runs" / "detect" / "yolov9_accident_detection_run1" / "weights" / "best.pt"))
conf_thresh = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.4, 0.05)
min_area_ratio = st.sidebar.slider("Min bbox area ratio (frame area)", 0.0005, 0.05, 0.002, 0.0005)
persist_frames = st.sidebar.slider("Persistence frames to confirm accident", 1, 30, 6)
min_avg_conf = st.sidebar.slider("Min avg confidence for confirmation", 0.0, 1.0, 0.5, 0.05)
max_process_fps = st.sidebar.slider("Max processing FPS (for video/webcam)", 1, 30, 10)

st.sidebar.markdown("---")
st.sidebar.write("Tracking:")
use_sort = False
if SORT_AVAILABLE:
    use_sort = st.sidebar.checkbox("Use SORT tracker (if installed)", value=True)
else:
    st.sidebar.write("SORT not available — using simple centroid tracker.")
    use_sort = st.sidebar.checkbox("Use simple centroid tracker", value=True)

st.sidebar.markdown("---")
out_dir = st.sidebar.text_input("Alerts output directory", value="alerts")
os.makedirs(out_dir, exist_ok=True)

# -----------------------
# Load model (deferred until user clicks)
# -----------------------
@st.cache_resource(show_spinner=False)
def load_model(path):
    st.info(f"Loading YOLO model from: {path}")
    model = YOLO(path)
    return model

# -----------------------
# Main app: input selection
# -----------------------
input_mode = st.radio("Input source", ["Image Upload", "Video Upload", "Webcam (live)"])

col1, col2 = st.columns([1,1])
with col1:
    st.write("Model status")
    model = None
    try:
        if Path(model_path).exists():
            model = load_model(model_path)
            st.success("Model loaded.")
        else:
            st.warning("Model path does not exist. Provide a correct path in the sidebar.")
    except Exception as e:
        st.error(f"Error loading model: {e}")

with col2:
    st.write("Info")
    st.write(f"Confidence ≥ **{conf_thresh}**  •  Persist **{persist_frames} frames**  •  Min avg conf **{min_avg_conf}**")

# -----------------------
# Accident memory and tracker initialization (per-run)
# -----------------------
if 'accident_memory' not in st.session_state:
    st.session_state['accident_memory'] = {}  # track_id -> list of (frame_id, conf)
if 'tracker' not in st.session_state:
    if SORT_AVAILABLE and use_sort:
        st.session_state['tracker'] = Sort(max_age=15, min_hits=1, iou_threshold=0.3)
    else:
        st.session_state['tracker'] = SimpleCentroidTracker(max_lost=15)
if 'frame_id' not in st.session_state:
    st.session_state['frame_id'] = 0

# -----------------------
# Processing functions
# -----------------------
def process_frame(frame, model, tracker, frame_id):
    """Run model on a single frame and return annotated frame and list of confirmed alerts"""
    alerts = []
    # run inference (Ultralytics can accept BGR numpy frame)
    results = model(frame, conf=conf_thresh, verbose=False)[0]
    dets = safe_get_boxes(results)  # [[x1,y1,x2,y2,conf,cls], ...]
    detections_for_tracker = []
    # build [x1,y1,x2,y2,score] expected by SORT or our simple tracker
    for d in dets:
        x1,y1,x2,y2,conf,cls = d
        if conf < conf_thresh:
            continue
        detections_for_tracker.append([x1,y1,x2,y2,conf])
    if len(detections_for_tracker) == 0:
        tracked_objects = np.empty((0,5))
    else:
        tracked_objects = tracker.update(np.array(detections_for_tracker))

    # tracked_objects is array of [x1,y1,x2,y2,id] for both SORT and simple tracker
    # Annotate and track persistency
    h, w = frame.shape[:2]
    frame_area = h * w
    for tr in tracked_objects:
        x1, y1, x2, y2, tid = map(int, tr)
        # area filter
        area = (x2 - x1) * (y2 - y1)
        if area < min_area_ratio * frame_area:
            continue
        # find confidence for this object from detections_for_tracker by IoU approx
        conf_val = 0.0
        for d in detections_for_tracker:
            dx1, dy1, dx2, dy2, dconf = d
            # IoU-like check
            ix1 = max(dx1, x1); iy1 = max(dy1, y1); ix2 = min(dx2, x2); iy2 = min(dy2, y2)
            iw = max(0, ix2-ix1); ih = max(0, iy2-iy1)
            inter = iw*ih
            area_d = (dx2-dx1)*(dy2-dy1) + 1e-9
            if inter/area_d > 0.2:
                conf_val = max(conf_val, dconf)
        # update memory
        mem = st.session_state['accident_memory'].setdefault(tid, [])
        mem.append((frame_id, float(conf_val)))
        # keep last 30 frames
        st.session_state['accident_memory'][tid] = mem[-30:]
        label = None
        if len(st.session_state['accident_memory'][tid]) >= persist_frames:
            avg_conf = float(np.mean([c for _, c in st.session_state['accident_memory'][tid]]))
            if avg_conf >= min_avg_conf:
                label = f"ACCIDENT CONFIRMED (avg {avg_conf:.2f})"
                # save cropped alert
                saved = crop_and_save(frame, (x1,y1,x2,y2), out_dir, prefix=f"alert_id{tid}")
                if saved:
                    alerts.append((tid, avg_conf, saved))
            else:
                label = f"Suspicious (avg {avg_conf:.2f})"
        # draw
        display_label = label if label else f"ID:{tid} conf:{conf_val:.2f}"
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, display_label, (x1, max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return frame, alerts

# -----------------------
# Input-specific flows
# -----------------------
if input_mode == "Image Upload":
    uploaded = st.file_uploader("Upload image(s)", type=["jpg","jpeg","png"], accept_multiple_files=True)
    if uploaded:
        cols = st.columns(2)
        for file in uploaded:
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                st.error("Could not read image")
                continue
            if model is None:
                st.error("Model not loaded.")
                break
            st.session_state['frame_id'] += 1
            annotated, alerts = process_frame(img.copy(), model, st.session_state['tracker'], st.session_state['frame_id'])
            cols[0].image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Annotated", use_column_width=True)
            if alerts:
                cols[1].write("🔴 Alerts:")
                for tid, avg_conf, path in alerts:
                    cols[1].write(f"- ID {tid} avg_conf {avg_conf:.2f}")
                    cols[1].image(path, width=240)
            else:
                cols[1].write("No confirmed accidents detected.")

elif input_mode == "Video Upload":
    uploaded = st.file_uploader("Upload video", type=["mp4","mov","avi","mkv"])
    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        vid_path = tfile.name
        st.video(vid_path, start_time=0)
        if st.button("Run detection on video"):
            cap = cv2.VideoCapture(vid_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS)>0 else 20
            display_placeholder = st.empty()
            frame_count = 0
            last_time = 0
            out_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(out_temp.name, fourcc, fps, (width, height))
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                # throttle processing to max_process_fps
                now = time.time()
                if (now - last_time) < (1.0 / max_process_fps):
                    # but still write original frame to output
                    out_writer.write(frame)
                    continue
                last_time = now
                st.session_state['frame_id'] += 1
                annotated, alerts = process_frame(frame.copy(), model, st.session_state['tracker'], st.session_state['frame_id'])
                out_writer.write(annotated)
                # show in UI
                display_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)
            cap.release()
            out_writer.release()
            st.success("Processing finished.")
            st.video(out_temp.name)
            st.write(f"Alert crops saved in: {out_dir}")

else:  # Webcam
    st.write("Webcam input (use browser camera).")
    cam_img = st.camera_input("Click to take a frame (for continuous webcam you'd use capture loops; Streamlit camera_input captures single frames)")
    if cam_img:
        # camera_input returns an UploadedFile-like
        file_bytes = np.asarray(bytearray(cam_img.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.session_state['frame_id'] += 1
        annotated, alerts = process_frame(frame.copy(), model, st.session_state['tracker'], st.session_state['frame_id'])
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)
        if alerts:
            st.write("Alerts:")
            for tid, avg_conf, path in alerts:
                st.write(f"- ID {tid} avg_conf {avg_conf:.2f}")
                st.image(path, width=240)
        else:
            st.write("No confirmed accidents detected.")

# -----------------------
# Footer / Notes
# -----------------------
st.markdown("---")
st.write("Notes & troubleshooting:")
st.write("""
- Make sure the model path is correct and the Ultralytics package and torch are installed in the same Python executable that runs Streamlit.  
- If SORT is not installed, the app uses a simple centroid tracker; install SORT for better tracking.  
- Alerts (cropped images) are saved into the configured alerts directory.
""")'''

import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import numpy as np

# -------------------------------
# Load Trained YOLO Model
# -------------------------------
model = YOLO(r"D:\srikar\Projects\Projects\yolov9\runs\detect\yolov9_accident_detection_run1\weights\best.pt")  # change to your model file name if needed

# -------------------------------
# Streamlit UI Setup
# -------------------------------
st.set_page_config(page_title="Accident Detection Dashboard", layout="wide")
st.title("🚦 Real-Time Accident Detection Dashboard")
st.write("Upload CCTV footage to detect accidents automatically using YOLOv9")

uploaded_video = st.file_uploader("Upload a tunnel CCTV video", type=["mp4", "avi", "mov", "mkv"])

alert_placeholder = st.empty()   # to show alert message dynamically
frame_window = st.image([])      # for live frame updates

# -------------------------------
# Accident detection processing
# -------------------------------
if uploaded_video is not None:
    # save uploaded video temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_video.read())

    cap = cv2.VideoCapture(temp_file.name)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.info("Video has ended.")
            break

        # Run YOLO model inference
        results = model(frame, stream=False)
        annotated_frame = results[0].plot()
        
        # accident detection logic
        accident_detected = False
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = box.conf[0]
            if cls == 0 and conf > 0.50:   # class 0 = accident class
                accident_detected = True

        # Show alert if accident detected
        if accident_detected:
            alert_placeholder.error("⚠ ACCIDENT DETECTED — Immediate Action Required!")
        else:
            alert_placeholder.info("No Accident Detected")

        # display annotated video frame
        frame_window.image(annotated_frame, channels="BGR")

    cap.release()
