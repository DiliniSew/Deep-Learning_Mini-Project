import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

st.title("Age & Gender Detection")

# Load model
model = load_model("best_model.keras")

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Preprocess function
def preprocess(face):
    # Convert BGR → RGB (IMPORTANT)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # Resize to model input
    face = cv2.resize(face, (224, 224))

    # Normalize (-1 to 1) → try this first
    face = (face / 127.5) - 1

    # Reshape
    face = np.reshape(face, (1, 224, 224, 3))

    return face

# Prediction function
def predict(face):
    age_pred, gender_pred = model.predict(face)

    age = age_pred[0][0]

    gender_prob = gender_pred[0][0]

    # Default assumption: 1 = Female, 0 = Male
    gender = "Female" if gender_prob > 0.5 else "Male"

    return int(age), gender, gender_prob


# 📸 Camera input
img_file = st.camera_input("Take a picture")

if img_file is not None:
    # Convert to OpenCV format
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    st.image(frame, channels="BGR", caption="Captured Image")

    # Face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)

    if len(faces) == 0:
        st.warning("No face detected")
    else:
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]

            processed = preprocess(face)
            age, gender, gender_prob = predict(processed)

            label = f"{gender}, {age}"

            # Draw box + label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, label,
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0,255,0), 2)

        # Show result
        st.image(frame, channels="BGR", caption="Result")

        # Optional debug info (very useful)
        st.write(f"Gender confidence: {gender_prob:.2f}")