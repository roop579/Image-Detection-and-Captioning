# app.py - All-in-one VisuaLingo
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import pyttsx3
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

st.set_page_config(page_title="VisuaLingo", layout="centered")
st.title("VisuaLingo — Vision + Language")

#  Load models globally 
@st.cache_resource
def load_models():
    # YOLO detection
    yolo_model = YOLO("yolov8n.pt")
    # BLIP caption
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model.eval()
    return yolo_model, processor, blip_model

yolo_model, processor, blip_model = load_models()

# Helper functions 
def detect_objects_and_annotate(image_bgr):
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = yolo_model(img_rgb)
    r = results[0]
    labels = []
    annotated = image_bgr.copy()
    for box in r.boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        cls_idx = int(box.cls[0].cpu().numpy())
        conf = float(box.conf[0].cpu().numpy())
        name = yolo_model.names[cls_idx]
        labels.append((name, conf, xyxy))
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"{name} {conf:.2f}", (x1, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return labels, annotated

def generate_caption_from_pil(image_pil):
    inputs = processor(images=image_pil, return_tensors="pt")
    out = blip_model.generate(**inputs, max_length=64)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def process_image(image_bgr):
    labels, annotated = detect_objects_and_annotate(image_bgr)
    pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    caption = generate_caption_from_pil(pil_img)
    detected_names = [lbl[0] for lbl in labels]
    return {
        "detected": detected_names,
        "label_conf_pairs": labels,
        "caption": caption,
        "annotated_image": annotated
    }

# Streamlit UI
uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Input Image", use_column_width=True)

    with st.spinner("Analyzing image..."):
        result = process_image(image_bgr)

    st.subheader("Detected Objects")
    if result["detected"]:
        st.write(", ".join(result["detected"]))
    else:
        st.write("No objects detected")

    st.subheader("Generated Caption")
    st.success(result["caption"])

    st.subheader("Annotated Image")
    ann = cv2.cvtColor(result["annotated_image"], cv2.COLOR_BGR2RGB)
    st.image(ann, use_column_width=True)

    if st.button("Speak Caption"):
        engine = pyttsx3.init()
        engine.say(result["caption"])
        engine.runAndWait()
