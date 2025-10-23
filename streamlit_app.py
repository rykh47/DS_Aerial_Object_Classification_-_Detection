import io
import json
from typing import List

import albumentations as A
import cv2
import numpy as np
import streamlit as st
import torch

from src.models.custom_cnn import build_model as build_custom
from src.models.transfer import build_model as build_transfer


st.set_page_config(page_title="Aerial Bird/Drone Classification", layout="centered")
st.title("Aerial Object Classification")
st.caption("Binary classification: Bird vs Drone. Optional detection via YOLOv8.")


@st.cache_resource(show_spinner=False)
def load_classifier(weights_path: str, class_json_path: str, arch: str):
    from src.utils.common import get_device, load_checkpoint

    device = get_device()
    with open(class_json_path, "r", encoding="utf-8") as f:
        class_names: List[str] = json.load(f)["class_names"]
    num_classes = len(class_names)
    if arch == "custom_cnn":
        model = build_custom(num_classes)
    else:
        model = build_transfer(arch, num_classes=num_classes, pretrained=False)
    ckpt = load_checkpoint(weights_path, map_location=str(device))
    model.load_state_dict(ckpt.model_state)
    model.to(device)
    model.eval()
    return model, class_names, device


def preprocess_image(image_bgr: np.ndarray, img_size: int = 224) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    t = A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    out = t(image=image_rgb)["image"]
    tensor = torch.from_numpy(np.transpose(out, (2, 0, 1)).astype(np.float32)).unsqueeze(0)
    return tensor


with st.sidebar:
    st.header("Model")
    weights = st.text_input("Classifier weights path", value="runs/cls_resnet50/best.pt")
    class_json = st.text_input("Class names json", value="runs/cls_resnet50/class_names.json")
    arch = st.selectbox("Architecture", ["resnet50", "mobilenet_v3_small", "efficientnet_b0", "custom_cnn"], index=0)
    img_size = st.number_input("Image size", min_value=128, max_value=640, value=224, step=32)
    load_btn = st.button("Load Model")

if load_btn:
    try:
        model, class_names, device = load_classifier(weights, class_json, arch)
        st.success(f"Loaded. Classes: {class_names}")
        st.session_state["model_loaded"] = True
        st.session_state["model"] = model
        st.session_state["class_names"] = class_names
        st.session_state["device"] = device
    except Exception as e:
        st.error(str(e))


uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded is not None and st.session_state.get("model_loaded", False):
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Input", use_column_width=True)

    with st.spinner("Predicting..."):
        tensor = preprocess_image(image, img_size)
        model: torch.nn.Module = st.session_state["model"]
        device = st.session_state["device"]
        class_names: List[str] = st.session_state["class_names"]
        with torch.no_grad():
            logits = model(tensor.to(device))
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))
            pred_name = class_names[pred_idx]
            confidence = float(probs[pred_idx])

    st.subheader("Prediction")
    st.markdown(f"**{pred_name}** — confidence: {confidence:.3f}")

st.divider()
st.header("Optional: YOLOv8 Detection")
with st.expander("Run YOLOv8 on the same image"):
    yolo_weights = st.text_input("YOLO weights", value="runs/yolo/weights/best.pt")
    if uploaded is not None and st.button("Detect with YOLOv8"):
        try:
            from ultralytics import YOLO

            yolo = YOLO(yolo_weights)
            results = yolo.predict(source=cv2.cvtColor(image, cv2.COLOR_BGR2RGB), verbose=False)[0]
            plotted = results.plot()
            st.image(plotted, caption="YOLOv8 Detections", use_column_width=True)
        except Exception as e:
            st.error(str(e))


