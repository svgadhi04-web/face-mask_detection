import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image

# Load trained model
model_path = os.path.join(os.path.dirname(__file__), "face_mask_model.keras")
model = tf.keras.models.load_model(model_path)

# Load Face Detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

st.title("😷 Face Mask Detection")
st.write("Upload a human face image.")

uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    # No face detected
    if len(faces) == 0:
        st.error("❌ No Human Face Detected")

    else:

        x, y, w, h = faces[0]

        face = img[y:y+h, x:x+w]

        face = cv2.resize(face, (128,128))

        face = face / 255.0

        face = np.expand_dims(face, axis=0)

        prediction = model.predict(face, verbose=0)[0][0]

        if prediction > 0.5:
            st.success("🙂 Without Mask")
        else:
            st.success("😷 With Mask")