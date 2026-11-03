import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
import cv2
import os

# ---- SECTION 1: LOAD SAMPLE A/B DATA (Simulation) ----
@st.cache_data
def load_emnist_ab():
    from sklearn.datasets import fetch_openml
    # Fetch EMNIST (Letters) from OpenML — slow first time
    data = fetch_openml('EMNIST_Letters', version=1)
    X = data['data']
    y = data['target'].astype(int)

    # EMNIST labels: A=1, B=2, ..., Z=26 (not 0-indexed!)
    mask = (y == 1) | (y == 2)
    X_ab = X[mask]
    y_ab = y[mask]
    return X_ab / 255.0, y_ab

# ---- SECTION 2: TRAIN THE SVM ----
@st.cache_resource
def train_svm(X_train, y_train):
    clf = svm.SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)
    return clf

# ---- SECTION 3: APP UI ----
st.title("🖊️ Is That an A or B? - Handwritten Letter SVM Classifier")

st.markdown("""
Draw a letter **A** or **B** below (uppercase only).  
The AI will try to recognize it using an SVM trained on real EMNIST letters.
""")

# Load and train
with st.spinner("Loading EMNIST and training SVM..."):
    X_ab, y_ab = load_emnist_ab()
    clf = train_svm(X_ab, y_ab)

# ---- SECTION 4: CANVAS FOR DRAWING ----
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=15,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# ---- SECTION 5: PROCESS DRAWING ----
if canvas_result.image_data is not None:
    img = canvas_result.image_data

    if st.button("Predict My Letter"):
        # Convert to grayscale & resize to 28x28
        gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
        resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        flattened = resized.flatten().reshape(1, -1)
        flattened = flattened / 255.0  # Normalize like EMNIST

        # Predict
        pred = clf.predict(flattened)[0]
        prob = clf.predict_proba(flattened)[0]

        letter = "A" if pred == 1 else "B"
        confidence = prob[pred - 1] * 100  # Adjust index (1-based to 0-based)

        st.markdown(f"### 🧠 Prediction: **{letter}**")
        st.markdown(f"**Confidence:** {confidence:.2f}%")

        st.image(resized, width=100, caption="Your Drawing (28x28)")



