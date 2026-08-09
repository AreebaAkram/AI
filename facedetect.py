import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Face Detection")
file = st.file_uploader("Upload an image", type = ["png", "jpeg", "jpg", "jpeg"])
if file is not None:
    image = Image.open(file).convert("RGB")
    img_np = np.array(image)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_image = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor = 1.1, minNeighbors = 5, minSize = (30,30))
    
    for(x,y,w,h) in faces:
        cv2.rectangle(img_cv, (x,y), (x+w , y+h), (0,255,0),2)
    result_image = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    st.image(result_image, caption ="Detected Faces")
    