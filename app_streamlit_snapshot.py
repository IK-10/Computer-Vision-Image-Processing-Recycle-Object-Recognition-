# app_streamlit_snapshot.py
import streamlit as st
from pathlib import Path
import tempfile, time, os
import cv2, numpy as np
import torch
from ultralytics import YOLO
import requests
from PIL import Image, ImageOps

st.set_page_config(page_title="Recycle Detector (snapshot)", layout="centered")

# ---------------- CONFIG ----------------
# If you want the app to fetch weights automatically, put a public URL here (GitHub release, GDrive raw, HF link)
MODEL_URL = ""  # e.g. "https://.../best.pt"  (leave empty if you will add best.pt to the repo)
MODEL_LOCAL = Path("best.pt")   # where the model will be stored inside the app folder

GROUP_NAMES = ['paper','glass','plastic_metal','non_recyclable']
COLOR_MAP_RGB = {
    0: (0, 0, 255),    # Blue
    1: (150, 75, 0),   # Brown
    2: (255, 165, 0),  # Orange
    3: (0, 0, 0)       # Black
}
# ----------------------------------------

st.title("♻️ Recycle Detector (Snapshot mode)")
st.markdown("Take a snapshot with your webcam. The model will predict which bin (Blue/Brown/Orange/Black) the item belongs to.")

# ---------- helper: download model ----------
def download_file(url: str, dest: Path, chunk_size=1024*1024):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)

if MODEL_LOCAL.exists():
    st.info(f"Found local model: {MODEL_LOCAL.name}")
else:
    if MODEL_URL:
        st.info("Downloading model... this may take a while.")
        try:
            download_file(MODEL_URL, MODEL_LOCAL)
            st.success("Model downloaded.")
        except Exception as e:
            st.error(f"Failed to download model: {e}")
    else:
        st.warning("Model not found in repo and no MODEL_URL provided. Upload `best.pt` to repo or set MODEL_URL.")

# ---------- load model (cached) ----------
@st.cache_resource
def load_model(path: str):
    assert Path(path).exists(), "Model file not found. Upload or set MODEL_URL."
    m = YOLO(path)
    # Note: on Streamlit Cloud this will be CPU-only
    return m

if MODEL_LOCAL.exists():
    model = load_model(str(MODEL_LOCAL))
else:
    st.stop()  # nothing to run

# ---------- UI params ----------
conf = st.sidebar.slider("Confidence threshold", 0.05, 0.9, 0.25, 0.01)
imgsz = st.sidebar.selectbox("Image size (inference)", [320, 416, 640], index=0)
st.sidebar.markdown("Colors: Blue=Paper, Brown=Glass, Orange=Plastic/Metal, Black=Non-recyclable")

# ---------- camera snapshot ----------
st.markdown("## Capture")
img_file = st.camera_input("Take a snapshot")

if img_file is not None:
    # convert to OpenCV image
    img = Image.open(img_file)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img_np = np.array(img)[:, :, ::-1]  # RGB->BGR for OpenCV/Ultralytics

    st.image(np.array(img), caption="Your snapshot (preview)", use_column_width=True)

    with st.spinner("Running detection..."):
        # Ultralytics accepts numpy BGR images
        results = model.predict(source=img_np, imgsz=imgsz, conf=conf, verbose=False)
        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.array([])
        scores = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.array([])
        classes = r.boxes.cls.cpu().numpy().astype(int) if len(r.boxes) else np.array([])

    # draw on image (matplotlib-friendly RGB)
    draw = img_np.copy()
    for (x1,y1,x2,y2), s, c in zip(boxes, scores, classes):
        if s < conf: 
            continue
        color = COLOR_MAP_RGB.get(int(c), (255,0,0))
        # convert RGB color to BGR for cv2
        color_bgr = (int(color[2]), int(color[1]), int(color[0]))
        cv2.rectangle(draw, (int(x1),int(y1)), (int(x2),int(y2)), color_bgr, 2)
        cv2.putText(draw, f"{GROUP_NAMES[int(c)]} {s:.2f}", (int(x1), int(y1)-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2, cv2.LINE_AA)

    # recommended bin: top scoring detection
    recommended = None
    if len(scores) > 0:
        idx = int(np.argmax(scores))
        if scores[idx] >= conf:
            recommended = GROUP_NAMES[int(classes[idx])]

    # show processed image
    out_rgb = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    st.image(out_rgb, caption="Detections", use_column_width=True)

    if recommended:
        color = COLOR_MAP_RGB[GROUP_NAMES.index(recommended)]
        st.success(f"Recommended bin: {recommended.upper()}")
    else:
        st.info("No confident detection. Try another snapshot or change confidence threshold.")
