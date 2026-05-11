import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="CyberVision AI",
    page_icon="🤖",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(135deg, #050816, #0a0f2c, #111827);
        color: white;
    }

    .main-title {
        font-size: 55px;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg,#00F5FF,#7B61FF,#00FFA3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        color: #B8C1EC;
        font-size: 18px;
        margin-bottom: 35px;
    }

    .glass {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        backdrop-filter: blur(10px);
        padding: 25px;
        border-radius: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }

    .prediction-box {
        background: linear-gradient(135deg,#00F5FF22,#7B61FF22);
        padding: 20px;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.08);
        text-align: center;
        margin-top: 20px;
    }

    .metric {
        font-size: 28px;
        font-weight: bold;
        color: #00F5FF;
    }

    .stButton>button {
        background: linear-gradient(90deg,#00F5FF,#7B61FF);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 25px;
        font-weight: bold;
        transition: 0.3s;
    }

    .stButton>button:hover {
        transform: scale(1.03);
        box-shadow: 0 0 20px rgba(0,245,255,0.5);
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# TITLE
# ==========================================

st.markdown('<div class="main-title">CYBERVISION AI</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">Futuristic Age & Gender Detection System</div>',
    unsafe_allow_html=True
)

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_model.keras")
    return model

model = load_model()

# ==========================================
# FACE DETECTOR
# ==========================================

face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# ==========================================
# PREPROCESS FUNCTION
# ==========================================

def preprocess_face(face):

    face = cv2.resize(face, (224, 224))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)

    return face

# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_age_gender(face):

    processed = preprocess_face(face)

    # YOUR MODEL RETURNS TWO OUTPUTS
    age_pred, gender_pred = model.predict(processed, verbose=0)

    age = int(age_pred[0][0])

    gender_prob = gender_pred[0][0]

    # YOUR MODEL LOGIC:
    # 1 = Female
    # 0 = Male

    gender = "Female" if gender_prob > 0.5 else "Male"

    confidence = float(max(gender_prob, 1 - gender_prob))

    return age, gender, confidence

# ==========================================
# PROCESS FRAME
# ==========================================

def process_frame(frame):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    results = []

    for (x, y, w, h) in faces:

        face = rgb[y:y+h, x:x+w]

        age, gender, conf = predict_age_gender(face)

        results.append((age, gender, conf))

        color = (0, 255, 255)

        cv2.rectangle(rgb, (x, y), (x+w, y+h), color, 3)

        label = f"{gender} | {age} yrs"

        cv2.putText(
            rgb,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    return rgb, results

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.markdown("## ⚡ Detection Modes")

option = st.sidebar.radio(
    "Choose Input Mode",
    [
        "🎥 Live Webcam Detection",
        "📸 Capture From Webcam",
        "📂 Upload Image"
    ]
)

# ==========================================
# LIVE WEBCAM
# ==========================================

if option == "🎥 Live Webcam Detection":

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.subheader("🎥 Real-Time Face Detection")

    st.info("Live AI-powered gender and age prediction")

    class VideoProcessor(VideoProcessorBase):

        def recv(self, frame):

            img = frame.to_ndarray(format="bgr24")

            processed, _ = process_frame(img)

            return av.VideoFrame.from_ndarray(processed, format="rgb24")

    webrtc_streamer(
        key="live-detection",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# CAPTURE FROM WEBCAM
# ==========================================

elif option == "📸 Capture From Webcam":

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.subheader("📸 Capture & Analyze")

    captured_image = st.camera_input("Take a photo")

    if captured_image:

        image = Image.open(captured_image)

        image_np = np.array(image)

        processed_image, results = process_frame(image_np)

        st.image(processed_image, use_container_width=True)

        if results:

            for age, gender, conf in results:

                st.markdown(
                    f"""
                    <div class="prediction-box">
                        <div class="metric">{gender}</div>
                        <h2>{age} Years Old</h2>
                        <p>Confidence: {conf:.2f}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:
            st.warning("No face detected")

    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# UPLOAD IMAGE
# ==========================================

elif option == "📂 Upload Image":

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.subheader("📂 Upload Image")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file)

        image_np = np.array(image)

        processed_image, results = process_frame(image_np)

        st.image(processed_image, use_container_width=True)

        if results:

            for age, gender, conf in results:

                st.markdown(
                    f"""
                    <div class="prediction-box">
                        <div class="metric">{gender}</div>
                        <h2>{age} Years Old</h2>
                        <p>Confidence: {conf:.2f}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:
            st.warning("No face detected")

    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# FOOTER
# ==========================================

st.markdown(
    """
    <hr>
    <center>
        <p style='color:#B8C1EC;'>
        Powered by TensorFlow • Streamlit • OpenCV
        </p>
    </center>
    """,
    unsafe_allow_html=True
)