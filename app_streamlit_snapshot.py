# app_streamlit_snapshot_safe.py
# Streamlit snapshot app with cv2 fallback (Pillow) to avoid libGL errors on cloud
# Usage: put this file in your repo and deploy. If best.pt isn't in the repo, set MODEL_URL to a public link.

import streamlit as st
from pathlib import Path
import tempfile
import time
import requests
import io
import numpy as np
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(page_title="Recycle Detector (safe)", layout="centered")

# ---------------- CONFIG ----------------
# If you want the app to fetch weights automatically, set a public URL here (GitHub release raw, HF, GDrive direct link)
MODEL_URL = ""  # e.g. "https://.../best.pt"  (leave empty if best.pt is in repo)
MODEL_LOCAL = Path("best.pt")   # location to store model in app filesystem

GROUP_NAMES = ['paper','glass','plastic_metal','non_recyclable']
COLOR_MAP_RGB = {
    0: (0, 0, 255),    # Blue
    1: (150, 75, 0),   # Brown
    2: (255, 165, 0),  # Orange
    3: (0, 0, 0)       # Black
}
# ----------------------------------------

st.title("♻️ Recycle Detector (Snapshot mode — safe)")

# ---------------- safe cv2 import ----------------
USE_CV2 = False
try:
    import cv2
    _ = cv2.__version__  # quick check
    USE_CV2 = True
except Exception as e:
    # cv2 import failed (likely libGL missing). We'll use Pillow fallback.
    st.info("cv2 not available in this environment; using Pillow-based fallback for image processing.")
    USE_CV2 = False

# ---------------- helper: download model ----------------
def download_file(url: str, dest: Path, chunk_size=1024*1024):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)

# ---------------- helper: image I/O ----------------
def pil_from_bytes(b: bytes):
    return Image.open(io.BytesIO(b)).convert("RGB")

def numpy_from_pil(img_pil: Image.Image):
    return np.array(img_pil)  # RGB numpy array

def prepare_for_model(img_rgb_np: np.ndarray):
    """
    Ultralytics YOLO accepts numpy arrays. Common usage expects BGR; to be safe,
    convert RGB->BGR if cv2 is not available (we do manual reverse).
    """
    if USE_CV2:
        # cv2 expects BGR; convert RGB->BGR
        return cv2.cvtColor(img_rgb_np, cv2.COLOR_RGB2BGR)
    else:
        # simple channel flip RGB->BGR
        return img_rgb_np[..., ::-1]

def draw_boxes_pil(img_rgb_np: np.ndarray, boxes, classes, scores, class_names, conf_thres=0.25, color_map=None):
    """Draw boxes on an RGB numpy image and return PIL Image"""
    img = Image.fromarray(img_rgb_np.astype('uint8'), 'RGB')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    if color_map is None:
        color_map = COLOR_MAP_RGB
    for (x1,y1,x2,y2), c, s in zip(boxes, classes, scores):
        if s < conf_thres: 
            continue
        color = tuple(color_map.get(int(c), (255,0,0)))
        # draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{class_names[int(c)]} {s:.2f}"
        text_w, text_h = draw.textsize(label, font=font)
        draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill=color)
        draw.text((x1 + 2, y1 - text_h - 2), label, fill=(255,255,255), font=font)
    return img

# ---------------- load model (cached) ----------------
@st.cache_resource
def load_model(path: str):
    from ultralytics import YOLO
    m = YOLO(path)
    return m

# Ensure model exists (download if MODEL_URL provided)
if not MODEL_LOCAL.exists():
    if MODEL_URL:
        with st.spinner("Downloading model (this may take a while)..."):
            try:
                download_file(MODEL_URL, MODEL_LOCAL)
                st.success("Model downloaded.")
            except Exception as e:
                st.error(f"Failed to download model: {e}")
                st.stop()
    else:
        st.warning("Model file not found (best.pt). Please either upload best.pt to the repo or set MODEL_URL in the app.")
        st.stop()

# Load model
with st.spinner("Loading model..."):
    try:
        model = load_model(str(MODEL_LOCAL))
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

# Sidebar controls
conf = st.sidebar.slider("Confidence threshold", 0.05, 0.9, 0.25, 0.01)
imgsz = st.sidebar.selectbox("Image size (inference)", [320, 416, 640], index=0)
st.sidebar.markdown("Colors: Blue=Paper, Brown=Glass, Orange=Plastic/Metal, Black=Non-recyclable")

# Input: camera snapshot OR file upload
st.markdown("## Capture / Upload")
col1, col2 = st.columns([2,1])

with col1:
    img_file = st.camera_input("Take a snapshot (browser camera)")
    if img_file is None:
        img_file = st.file_uploader("Or upload an image", type=["jpg","jpeg","png"])
with col2:
    st.write("Tips:")
    st.write("- Hold the object close to the camera")
    st.write("- Use plain background and good lighting")
    st.write("- For best results, show objects similar to training data")

if img_file is not None:
    # read bytes -> PIL -> numpy RGB
    if isinstance(img_file, bytes):
        raw = img_file
    else:
        raw = img_file.getvalue()
    pil_img = pil_from_bytes(raw)
    img_rgb = numpy_from_pil(pil_img)  # RGB numpy

    st.image(img_rgb, caption="Input image (preview)", use_column_width=True)

    # Prepare input for model
    model_input = prepare_for_model(img_rgb)  # BGR numpy if needed

    # Run detection
    with st.spinner("Running detection..."):
        try:
            # model.predict accepts numpy array
            results = model.predict(source=model_input, imgsz=imgsz, conf=conf, verbose=False)
            r = results[0]
            try:
                boxes = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0,4))
                scores = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros((0,))
                classes = r.boxes.cls.cpu().numpy().astype(int) if len(r.boxes) else np.zeros((0,), dtype=int)
            except Exception:
                boxes = np.zeros((0,4)); scores = np.zeros((0,)); classes = np.zeros((0,), dtype=int)
        except Exception as e:
            st.error(f"Detection error: {e}")
            boxes, scores, classes = np.zeros((0,4)), np.zeros((0,)), np.zeros((0,), dtype=int)

    # Convert boxes to simple Python floats for PIL drawing and clip to image dims
    H, W = img_rgb.shape[0], img_rgb.shape[1]
    boxes_clipped = []
    for b in boxes:
        x1, y1, x2, y2 = map(float, b)
        x1 = max(0, min(W-1, x1)); x2 = max(0, min(W-1, x2))
        y1 = max(0, min(H-1, y1)); y2 = max(0, min(H-1, y2))
        boxes_clipped.append((x1,y1,x2,y2))

    # Draw boxes using PIL fallback (RGB)
    img_out = draw_boxes_pil(img_rgb, boxes_clipped, classes, scores, model.names if hasattr(model,'names') else GROUP_NAMES, conf_thres=conf)
    st.image(img_out, caption="Detections", use_column_width=True)

    # Recommendation: highest-scoring detection
    recommended = None
    if len(scores) > 0:
        idx = int(np.argmax(scores))
        if scores[idx] >= conf:
            recommended = (int(classes[idx]), float(scores[idx]))
    if recommended:
        cid, sc = recommended
        st.success(f"Recommended bin: **{(model.names[cid] if hasattr(model,'names') else GROUP_NAMES[cid]).upper()}** (score={sc:.2f})")
    else:
        st.info("No confident detection. Try another snapshot or lower the confidence threshold.")

else:
    st.info("Take a snapshot or upload an image to run detection.")
