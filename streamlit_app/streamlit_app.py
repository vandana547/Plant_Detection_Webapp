import streamlit as st
from PIL import Image, ImageDraw
import cv2
import numpy as np
import tempfile
import os
from inference_sdk import InferenceHTTPClient

# --- Helper Functions ---

def load_image(image_file):
    img = Image.open(image_file)
    return img

def save_temp_image(image: Image.Image, quality: int) -> str:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    image.save(temp_file.name, format="JPEG", quality=quality)
    return temp_file.name

def capture_from_webcam(quality: int):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Webcam not available.")
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        st.error("Failed to capture image from webcam.")
        return None
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    return pil_img

def draw_bboxes(image: Image.Image, predictions, threshold: float):
    draw = ImageDraw.Draw(image)
    for pred in predictions:
        if pred.get('confidence', 0) < threshold:
            continue
        x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
        left = x - w / 2
        top = y - h / 2
        right = x + w / 2
        bottom = y + h / 2
        draw.rectangle([left, top, right, bottom], outline="lime", width=3)
        label = f"{pred.get('class', 'Unknown')} ({pred.get('confidence', 0):.2f})"
        draw.text((left, top - 10), label, fill="lime")
    return image

def detect_plants(image_path, client, threshold):
    try:
        result = client.infer(image_path, model_id="plant-type-detection-sex71/1")
        if result and 'predictions' in result:
            predictions = [p for p in result['predictions'] if p.get('confidence', 0) >= threshold]
            if predictions:
                result_text = "### Detected Plants\n"
                for pred in predictions:
                    result_text += f"- **{pred.get('class', 'Unknown')}** (Confidence: {pred.get('confidence', 0):.2f})\n"
            else:
                result_text = "No plants detected above threshold."
        else:
            result_text = "No results from API."
        return result, result_text
    except Exception as e:
        return None, f"API Error: {str(e)}"

def get_detected_plants_text(predictions):
    """Return formatted detected plant names and confidences for display above image."""
    if not predictions:
        return ""
    lines = ["<b>Detected Plants</b>"]
    for pred in predictions:
        name = pred.get('class', 'Unknown')
        conf = pred.get('confidence', 0)
        lines.append(f"{name} <span style='color:#27ae60;'>(Confidence: {conf:.2f})</span>")
    return "<br>".join(lines)

# --- Streamlit UI ---

st.set_page_config(page_title="🌱 Plant Type Detector", layout="wide")
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #1b2b1b 0%, #223322 100%);
    }
    .main {
        background-color: transparent !important;
    }
    .stApp {
        padding: 0rem 0rem 0rem 0rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: #223322;
        color: #fff;
        border-radius: 1.5rem 1.5rem 0 0;
        padding: 0.5rem 1.5rem;
        margin-right: 0.2rem;
        font-weight: 600;
        border: none;
        transition: background 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background: #27ae60 !important;
        color: #fff !important;
        box-shadow: 0 2px 8px #0002;
    }
    .stButton>button, .stDownloadButton>button {
        background: linear-gradient(90deg, #27ae60 60%, #219150 100%);
        color: #fff;
        border-radius: 1.5rem;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1.5rem;
        box-shadow: 0 2px 8px #0002;
        transition: background 0.2s, transform 0.1s;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background: linear-gradient(90deg, #219150 60%, #27ae60 100%);
        transform: translateY(-2px) scale(1.03);
    }
    .stSlider>div>div>div {background: #27ae60;}
    .stSlider>div[data-baseweb="slider"] {
        margin-bottom: 0.5rem;
    }
    .stSidebar {
        background: #182818;
        border-radius: 1.5rem;
        margin: 0.5rem 0.5rem 0.5rem 0.5rem;
        box-shadow: 0 2px 8px #0002;
    }
    .stSidebar .css-1d391kg {
        padding-top: 1rem;
    }
    .st-cq {
        border-radius: 1.5rem !important;
        box-shadow: 0 2px 8px #0002 !important;
    }
    .stImage>img, .element-container img {
        border-radius: 1.2rem;
        box-shadow: 0 2px 8px #0003;
        margin-bottom: 0.5rem;
    }
    .stMarkdown, .stCaption, .stTextInput, .stSlider, .stCheckbox {
        font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
    }
    .stTextInput>div>input {
        border-radius: 1rem;
        background: #223322;
        color: #fff;
        border: 1px solid #27ae60;
    }
    .stCheckbox>label>div:first-child {
        border-radius: 0.5rem;
        border: 2px solid #27ae60;
    }
    .stInfo {
        border-radius: 1rem;
        background: #223322;
        color: #fff;
    }
    .stAlert {
        border-radius: 1rem;
    }
    .stDownloadButton>button {
        margin-top: 0.5rem;
    }
    /* Hide Streamlit top menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True
)

st.markdown(
    """
    <div style='text-align:center; margin-top:1.5rem; margin-bottom:0.5rem;'>
        <span style='font-size:2.5rem; font-weight:800; font-family:Segoe UI,Roboto,Arial,sans-serif; color:#27ae60; letter-spacing:0.03em;'>
            🌱 Plant Type Detector
        </span>
        <div style='font-size:1.1rem; color:#b2dfb2; margin-top:0.2rem; font-weight:400;'>
            Detect plant types in your images using AI. Powered by Roboflow.
        </div>
    </div>
    """, unsafe_allow_html=True
)

# Sidebar - Settings
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("Roboflow API Key", value="Djt4RUTIw141MVv7y83Z", type="password")
quality = st.sidebar.slider("Image Quality (%)", min_value=10, max_value=100, value=70, step=5)
threshold = st.sidebar.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
show_bboxes = st.sidebar.checkbox("Show Bounding Boxes", value=True)
st.sidebar.markdown("---")
st.sidebar.info("You can upload an image or use your webcam.")

# Inference Client
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=api_key
)

# Session state for history
if "history" not in st.session_state:
    st.session_state["history"] = []

# Main UI
tab1, tab2, tab3 = st.tabs(["📷 Upload Image", "🎥 Webcam Capture", "🕑 Detection History"])

def add_to_history(img, predictions, result_text):
    # Save image and results in session state
    st.session_state["history"].append({
        "img": img.copy(),
        "predictions": predictions,
        "result_text": result_text
    })

with tab1:
    uploaded_file = st.file_uploader("Upload an image (jpg, png)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = load_image(uploaded_file)
        col1, col2 = st.columns([1, 1], gap="small")
        with col1:
            st.markdown(
                "<div style='background:#223322;border-radius:1rem;padding:0.5rem 0.5rem 0.2rem 0.5rem;box-shadow:0 2px 8px #0002;'>"
                "<b style='color:#b2dfb2;'>Input Image</b>",
                unsafe_allow_html=True
            )
            # Add vertical space to drag image down
            st.markdown("<div style='height:2.5rem;'></div>", unsafe_allow_html=True)
            st.image(img, use_container_width=True, clamp=True)
        temp_path = save_temp_image(img, quality)
        if st.button("Detect Plants", key="detect_upload"):
            with st.spinner("Detecting..."):
                result, result_text = detect_plants(temp_path, client, threshold)
                if "API Error" in result_text or "No plants detected" in result_text or "No results from API" in result_text:
                    st.info(result_text)
                if result and show_bboxes and result.get('predictions'):
                    img_bbox = img.copy()
                    img_bbox = draw_bboxes(img_bbox, result['predictions'], threshold)
                    with col2:
                        st.markdown(
                            "<div style='background:#223322;border-radius:1rem;padding:0.5rem 0.5rem 0.2rem 0.5rem;box-shadow:0 2px 8px #0002;'>"
                            f"{get_detected_plants_text(result['predictions'])}",
                            unsafe_allow_html=True
                        )
                        # Add vertical space to drag detection image down
                        st.markdown("<div style='height:2.5rem;'></div>", unsafe_allow_html=True)
                        st.image(img_bbox, use_container_width=True, clamp=True)
                        buf = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                        img_bbox.save(buf.name, format="JPEG")
                        # Robustly remove temp file (avoid PermissionError)
                        import time
                        import gc
                        # Add a unique key for each download_button using a hash of the filename
                        with open(buf.name, "rb") as f:
                            st.download_button(
                                "Download Result Image",
                                f,
                                file_name="detections.jpg",
                                key=f"download_{os.path.basename(buf.name)}"
                            )
                        gc.collect()
                        for _ in range(5):
                            try:
                                os.remove(buf.name)
                                break
                            except PermissionError:
                                time.sleep(0.2)
                add_to_history(img, result.get('predictions', []) if result else [], result_text)
            os.remove(temp_path)

with tab2:
    if st.button("Capture from Webcam"):
        img = capture_from_webcam(quality)
        if img is not None:
            col1, col2 = st.columns([1, 1], gap="small")
            with col1:
                st.markdown(
                    "<div style='background:#223322;border-radius:1rem;padding:0.5rem 0.5rem 0.2rem 0.5rem;box-shadow:0 2px 8px #0002;'>"
                    "<b style='color:#b2dfb2;'>Webcam Capture</b>",
                    unsafe_allow_html=True
                )
                st.markdown("<div style='height:2.5rem;'></div>", unsafe_allow_html=True)
                st.image(img, use_container_width=True, clamp=True)
            temp_path = save_temp_image(img, quality)
            if st.button("Detect Plants", key="detect_webcam"):
                with st.spinner("Detecting..."):
                    result, result_text = detect_plants(temp_path, client, threshold)
                    st.info(result_text)
                    if result and show_bboxes and result.get('predictions'):
                        img_bbox = img.copy()
                        img_bbox = draw_bboxes(img_bbox, result['predictions'], threshold)
                        with col2:
                            st.markdown(
                                "<div style='background:#223322;border-radius:1rem;padding:0.5rem 0.5rem 0.2rem 0.5rem;box-shadow:0 2px 8px #0002;'>"
                                f"{get_detected_plants_text(result['predictions'])}",
                                unsafe_allow_html=True
                            )
                            st.markdown("<div style='height:2.5rem;'></div>", unsafe_allow_html=True)
                            st.image(img_bbox, use_container_width=True, clamp=True)
                            buf = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                            img_bbox.save(buf.name, format="JPEG")
                            import time
                            import gc
                            with open(buf.name, "rb") as f:
                                st.download_button(
                                    "Download Result Image",
                                    f,
                                    file_name="detections.jpg",
                                    key=f"download_{os.path.basename(buf.name)}"
                                )
                            gc.collect()
                            for _ in range(5):
                                try:
                                    os.remove(buf.name)
                                    break
                                except PermissionError:
                                    time.sleep(0.2)
                    add_to_history(img, result.get('predictions', []) if result else [], result_text)
                os.remove(temp_path)

with tab3:
    st.subheader("Detection History")
    if not st.session_state["history"]:
        st.info("No detections yet.")
    else:
        for idx, entry in enumerate(reversed(st.session_state["history"]), 1):
            with st.container():
                col1, col2 = st.columns([1, 1], gap="small")
                with col1:
                    st.markdown(
                        "<div style='background:#223322;border-radius:1rem;padding:0.5rem 0.5rem 0.2rem 0.5rem;box-shadow:0 2px 8px #0002;'>"
                        f"<b style='color:#b2dfb2;'>Detection #{idx} - Input</b>",
                        unsafe_allow_html=True
                    )
                    st.markdown("<div style='height:2.5rem;'></div>", unsafe_allow_html=True)
                    st.image(entry["img"], use_container_width=True, clamp=True)
                if show_bboxes and entry["predictions"]:
                    img_bbox = entry["img"].copy()
                    img_bbox = draw_bboxes(img_bbox, entry["predictions"], threshold)
                    with col2:
                        st.markdown(
                            "<div style='background:#223322;border-radius:1rem;padding:0.5rem 0.5rem 0.2rem 0.5rem;box-shadow:0 2px 8px #0002;'>"
                            f"{get_detected_plants_text(entry['predictions'])}",
                            unsafe_allow_html=True
                        )
                        st.markdown("<div style='height:2.5rem;'></div>", unsafe_allow_html=True)
                        st.image(img_bbox, use_container_width=True, clamp=True)
                        buf = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                        img_bbox.save(buf.name, format="JPEG")
                        import time
                        import gc
                        with open(buf.name, "rb") as f:
                            st.download_button(
                                "Download Result Image",
                                f,
                                file_name="detections.jpg",
                                key=f"download_{os.path.basename(buf.name)}"
                            )
                        gc.collect()
                        for _ in range(5):
                            try:
                                os.remove(buf.name)
                                break
                            except PermissionError:
                                time.sleep(0.2)
                st.markdown("---")

# Footer
st.markdown("---")
st.markdown(
    "<center><small>Made with ❤️ using Streamlit & Roboflow API BY Vandana(</small></center>",
    unsafe_allow_html=True
)
