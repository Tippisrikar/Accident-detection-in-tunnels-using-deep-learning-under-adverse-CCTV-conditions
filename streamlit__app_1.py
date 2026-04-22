import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import base64
import numpy as np
import time
import wave
import contextlib
import os

# ---------------------------------------------------------
#  PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Accident Detection System", layout="wide")

# ---------------------------------------------------------
#  CUSTOM CSS
# ---------------------------------------------------------
page_bg_css = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #000000 0%, #1c1c1c 40%, #3d3d3d 100%);
}
h1, h2, h3, h4, h5, h6, label, span, p {
    color: #f5f5f5 !important;
    font-family: "Segoe UI", sans-serif;
}
.upload-box {
    padding: 20px;
    border-radius: 12px;
    background-color: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.18);
}
.alert-box {
    padding: 20px;
    font-size: 25px;
    font-weight: bold;
    border-radius: 12px;
    background-color: #E50914;
    color: white;
    text-align: center;
    animation: blinker 1s infinite;
}
@keyframes blinker { 50% { opacity: 0; } }
.sidebar-text {
    color: white;
}
.info-box {
    padding: 12px;
    border-radius: 10px;
    background-color: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    color: #ddd;
}
</style>
"""
st.markdown(page_bg_css, unsafe_allow_html=True)

# ---------------------------------------------------------
#  SIDEBAR SETTINGS
# ---------------------------------------------------------
st.sidebar.markdown("<h2 class='sidebar-text'>⚙ Settings</h2>", unsafe_allow_html=True)

conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.10, 1.0, 0.50, 0.05
)

st.sidebar.markdown("---")
st.sidebar.markdown("<h2 class='sidebar-text'>🔊 Alarm Settings</h2>", unsafe_allow_html=True)

alarm_file = st.sidebar.text_input(
    "Alarm Sound File Path (WAV recommended):",
    r"C:\Users\tippi\Downloads\mixkit-street-public-alarm-997.wav"
)

st.sidebar.markdown(
    "<div class='info-box'>Tip: use a WAV file for accurate duration detection. "
    "If not WAV, a default duration (5s) will be used.</div>",
    unsafe_allow_html=True
)

# ---------------------------------------------------------
#  MODEL LOAD (adjust path to your model)
# ---------------------------------------------------------
MODEL_PATH = r"D:\srikar\Projects\Projects\yolov9\runs\detect\yolov9_accident_detection_run1\weights\best.pt"

@st.cache_resource
def load_model(path):
    return YOLO(path)

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Error loading YOLO model: {e}")
    st.stop()

# ---------------------------------------------------------
#  UTIL: get audio duration (seconds)
# ---------------------------------------------------------
def get_audio_duration(path, fallback=5.0):
    try:
        if not os.path.exists(path):
            return fallback
        # Handle WAV via built-in wave module
        if path.lower().endswith(".wav"):
            with contextlib.closing(wave.open(path, 'r')) as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / float(rate)
                return float(duration)
        # For non-wav (mp3 etc.) we don't require extra deps here -> fallback
        return float(fallback)
    except Exception:
        return float(fallback)

# ---------------------------------------------------------
#  ALARM PLAY (embed as base64 audio tag)
# ---------------------------------------------------------
def play_alarm_html(path):
    try:
        with open(path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            # Try to detect mime type by extension
            ext = os.path.splitext(path)[1].lower()
            mime = "audio/wav" if ext == ".wav" else "audio/mpeg"
            audio_html = f"""
                <audio autoplay>
                    <source src="data:{mime};base64,{b64}" type="{mime}">
                    Your browser does not support the audio element.
                </audio>
            """
            st.markdown(audio_html, unsafe_allow_html=True)
            return True
    except Exception as e:
        st.warning(f"⚠ Unable to play alarm: {e}")
        return False

# Initialize session state for last_alarm_time
if "last_alarm_time" not in st.session_state:
    st.session_state["last_alarm_time"] = 0.0

# ---------------------------------------------------------
#  PAGE TITLE & UPLOAD
# ---------------------------------------------------------
st.markdown("<h1>🚦 Real-Time Accident Detection System</h1>", unsafe_allow_html=True)
st.markdown("Upload a CCTV tunnel video to automatically detect accidents using YOLOv9.")

uploaded_video = st.file_uploader(
    "Upload Tunnel CCTV Video",
    type=["mp4", "avi", "mov", "mkv"],
    help="Upload your video for accident detection."
)

frame_window = st.empty()
alert_placeholder = st.empty()
meta_col = st.empty()

# ---------------------------------------------------------
#  PROCESS VIDEO
# ---------------------------------------------------------
if uploaded_video:
    # Save uploaded to a temp file that OpenCV can open
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())
    tfile.flush()
    cap = cv2.VideoCapture(tfile.name)

    st.markdown("<h3>🎥 Processing Video...</h3>", unsafe_allow_html=True)

    # compute audio duration once
    alarm_duration = get_audio_duration(alarm_file, fallback=5.0)

    # main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            st.info("✔ Video completed.")
            break

        # Run model inference (ultralytics returns results object)
        results = model(frame, stream=False, conf=conf_threshold)
        annotated_frame = results[0].plot()

        # decide if any detection corresponds to 'accident' class
        # (This assumes your accident/tracked class index is 0 — adjust if different)
        accident_detected = False
        highest_conf = 0.0
        for box in results[0].boxes:
            try:
                cls_idx = int(box.cls[0])
                conf_val = float(box.conf[0])
            except Exception:
                continue
            if cls_idx == 0 and conf_val >= conf_threshold:
                accident_detected = True
                highest_conf = max(highest_conf, conf_val)

        # Only play alarm if either:
        #  - no previous alarm played (last_alarm_time == 0), or
        #  - time since last alarm >= alarm_duration
        now = time.time()
        time_since_last = now - st.session_state["last_alarm_time"]

        if accident_detected:
            alert_placeholder.markdown(
                "<div class='alert-box'>⚠ ACCIDENT DETECTED — EMERGENCY ALERT</div>",
                unsafe_allow_html=True
            )
            # re-trigger alarm only when previous has finished
            if time_since_last >= alarm_duration:
                played = play_alarm_html(alarm_file)
                if played:
                    st.session_state["last_alarm_time"] = time.time()
        else:
            alert_placeholder.empty()

        # Show metadata (confidence / time since last alarm)
        meta_col.markdown(
            f"<div class='info-box'>Highest accident conf: {highest_conf:.2f} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"Time since last alarm: {time_since_last:.1f}s</div>",
            unsafe_allow_html=True
        )

        # Display annotated frame
        frame_window.image(annotated_frame, channels="BGR", use_column_width=True)

    cap.release()
    try:
        os.unlink(tfile.name)
    except Exception:
        pass
else:
    st.info("Upload a video to start accident detection. Use the sidebar to change confidence and alarm file.")