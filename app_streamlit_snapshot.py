# app_streamlit_snapshot.py
# Streamlit app with YOLOv8n (safe mode for Streamlit Cloud)

import streamlit as st
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests, io

st.set_page_config(page_title="Recycle Detector (safe)", layout="centered")
st.title("♻️ Recycle Detector (Snapshot mode — safe)")

# ---------------- CONFIG ----------------
MODEL_LOCAL = Path("best.pt")
MODEL_URL = ""  # optional public link to your weights (e.g. HuggingFace/GDrive raw URL)
CLASS_NAMES = ['Paper', 'Glass', 'Plastic/Metal', 'Non-recyclable']
COLOR_MAP = {
    0: (0, 0, 255),    # Blue
    1: (150, 75, 0),   # Brown
    2: (255, 165, 0),  # Orange
    3: (0, 0, 0)       # Black
}
# ----------------------------------------

# ---- cv2 optional import ----
USE_CV2 = False
try:
    import cv2
    _ = cv2.__version__
    USE_CV2 = True
except Exception as e:
    st.info("cv2 not available — using Pillow fallback for image processing.")

# ---- safe image I/O ----
def pil_from_bytes(b: bytes):
    return Image.open(io.BytesIO(b)).convert("RGB")

def numpy_from_pil(img_pil: Image.Image):
    return np.array(img_pil)

def prepare_for_model(img_rgb_np: np.ndarray):
    if USE_CV2:
        return cv2.cvtColor(img_rgb_np, cv2.COLOR_RGB2BGR)
    else:
        return img_rgb_np[..., ::-1]  # RGB→BGR

def draw_boxes(img_rgb_np, boxes, classes, scores, class_names, conf_thres=0.25):
    img = Image.fromarray(img_rgb_np.astype('uint8'), 'RGB')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    for (x1, y1, x2, y2), c, s in zip(boxes, classes, scores):
        if s < conf_thres:
            continue
        color = COLOR_MAP.get(int(c), (255, 0, 0))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{class_names[int(c)]} {s:.2f}"
        tw, th = draw.textsize(label, font=font)
        draw.rectangle([x1, y1 - th - 4, x1 + tw + 4, y1], fill=color)
        draw.text((x1 + 2, y1 - th - 2), label, fill=(255, 255, 255), font=font)
    return img

# ---- download weights if needed ----
def download_file(url: str, dest: Path, chunk=1024*1024):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for c in r.iter_content(chunk_size=chunk):
                if c:
                    f.write(c)

if not MODEL_LOCAL.exists():
    if MODEL_URL:
        with st.spinner("Downloading model..."):
            try:
                download_file(MODEL_URL, MODEL_LOCAL)
                st.success("Model downloaded.")
            except Exception as e:
                st.error(f"Failed to download model: {e}")
                st.stop()
    else:
        st.error("Model file best.pt not found. Please upload it or set MODEL_URL.")
        st.stop()

# ---- load YOLO model safely ----
model = None
MODEL_LOADED = False
try:
    with st.spinner("Loading model..."):
        from ultralytics import YOLO
        model = YOLO(str(MODEL_LOCAL))
    MODEL_LOADED = True
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.warning("Model failed to load in this cloud environment (likely missing system libs). "
               "You can still run this app locally with `streamlit run app_streamlit_snapshot.py`.")
    MODEL_LOADED = False

# ---- Sidebar ----
conf = st.sidebar.slider("Confidence threshold", 0.05, 0.9, 0.25, 0.01)
imgsz = st.sidebar.selectbox("Image size", [320, 416, 640], index=0)
st.sidebar.markdown("Colors: Blue=Paper, Brown=Glass, Orange=Plastic/Metal, Black=Non-recyclable")

# ---- Main ----
if MODEL_LOADED:
    img_file = st.camera_input("Take a snapshot") or st.file_uploader("Or upload an image", type=["jpg","jpeg","png"])

    if img_file is not None:
        raw = img_file.getvalue()
        pil_img = pil_from_bytes(raw)
        img_rgb = numpy_from_pil(pil_img)

        st.image(img_rgb, caption="Input", use_column_width=True)

        model_input = prepare_for_model(img_rgb)

        with st.spinner("Running detection..."):
            try:
                results = model.predict(source=model_input, imgsz=imgsz, conf=conf, verbose=False)
                r = results[0]
                boxes = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0,4))
                scores = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros((0,))
                classes = r.boxes.cls.cpu().numpy().astype(int) if len(r.boxes) else np.zeros((0,), dtype=int)
            except Exception as e:
                st.error(f"Detection error: {e}")
                boxes, scores, classes = np.zeros((0,4)), np.zeros((0,)), np.zeros((0,), dtype=int)

        H, W = img_rgb.shape[:2]
        boxes_clipped = [(max(0,x1), max(0,y1), min(W-1,x2), min(H-1,y2)) for x1,y1,x2,y2 in boxes]

        img_out = draw_boxes(img_rgb, boxes_clipped, classes, scores, CLASS_NAMES, conf_thres=conf)
        st.image(img_out, caption="Detections", use_column_width=True)

        if len(scores) > 0:
            idx = int(np.argmax(scores))
            if scores[idx] >= conf:
                cid, sc = int(classes[idx]), float(scores[idx])
                st.success(f"Recommended bin: **{CLASS_NAMES[cid]}** (score={sc:.2f})")
            else:
                st.info("No confident detection.")
        else:
            st.info("No objects detected.")
    else:
        st.info("Take a snapshot or upload an image to run detection.")
else:
    st.stop()
