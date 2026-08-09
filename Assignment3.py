import streamlit as st
from PIL import Image
from transformers import pipeline

st.title("Implementation of VQA, Text generator, & Summarizer")
pipe = pipeline("visual-question-answering", model="dandelin/vilt-b32-finetuned-vqa")
image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "jfif"])

if image:
    img = Image.open(image)
    st.image(img, use_container_width=True)
else:
    st.info("error loading img")
question = st.text_input("Ask any question relevant to image")
if st.button ("Get answer"):
    result = pipe(img , question , top_k = 5)
    answer = result[0] [ 'answer']
    st.write(f"Result: {answer}")

    