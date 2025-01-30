import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st

# Load the Haar Cascade face detection model
haarcascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haarcascade_path)

# Load your trained mask detection model
model = load_model("face_mask_model_new.h5")

# Function to detect faces using Haar Cascade
def detect_faces_haarcascade(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# Streamlit app
st.title("Face Mask Detection from Video Upload")

# File uploader for video
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_video is not None:
    # Save the uploaded video to a temporary location
    with open("uploaded_video.mp4", "wb") as f:
        f.write(uploaded_video.read())

    # Read the uploaded video
    video_capture = cv2.VideoCapture("uploaded_video.mp4")

    stframe = st.empty()

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        # Detect faces
        faces = detect_faces_haarcascade(frame)

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            if face.size == 0:
                continue
            resized_face = cv2.resize(face, (256, 256))
            normalized_face = resized_face / 255.0
            reshaped_face = np.expand_dims(normalized_face, axis=0)

            # Predict mask/no mask
            prediction = model.predict(reshaped_face)[0][0]
            label = "Mask" if prediction > 0.5 else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Display frame
        stframe.image(frame, channels="BGR", use_container_width=True)

    video_capture.release()
    st.success("Video processing complete!")
