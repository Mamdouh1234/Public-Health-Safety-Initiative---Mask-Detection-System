import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st

# Load the pre-trained face detection model
prototxt_path = "deploy.prototxt"
caffemodel_path = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

# Load your trained mask detection model
model = load_model("face_mask_model_new.h5")

# Function to detect faces using DNN
def detect_faces_dnn(frame, conf_threshold=0.5):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    net.setInput(blob)
    detections = net.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            faces.append((x1, y1, x2, y2))
    return faces

# Streamlit app
st.title("Real-Time Face Mask Detection")

# Webcam feed
video_capture = cv2.VideoCapture(0)

stframe = st.empty()

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    # Detect faces
    faces = detect_faces_dnn(frame)

    for (x1, y1, x2, y2) in faces:
        face = frame[y1:y2, x1:x2]
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
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Display frame
    stframe.image(frame, channels="BGR", use_column_width=True)

video_capture.release()
