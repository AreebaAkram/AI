import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import torch
from sklearn.linear_model import LinearRegression
from transformers import pipeline, ViTImageProcessor, ViTForImageClassification
import io

def load_image():
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg", "jfif"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image",width= 400)
        return image
    return None

def resize_image(image, width, height):
    return image.resize((width, height), Image.Resampling.LANCZOS)

def convert_to_grayscale(image):
    return image.convert("L")

def apply_edge_detection(image):
    image_np = np.array(image)
    if len(image_np.shape) == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(image_np, 100, 200)
    return Image.fromarray(edges)

def adjust_contrast(image, factor):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def shift_image(image, x_shift, y_shift):
    image_np = np.array(image)
    rows, cols = image_np.shape[:2]
    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    shifted = cv2.warpAffine(image_np, M, (cols, rows), borderValue=(0, 0, 0))
    return Image.fromarray(shifted)

def apply_image_processing(image):
    st.sidebar.header("Image Processing Options")
    resize = st.sidebar.checkbox("Resize Image")
    if resize:
        width = st.sidebar.slider("Width", 100, 1000, 512)
        height = st.sidebar.slider("Height", 100, 1000, 512)
        image = resize_image(image, width, height)

    grayscale = st.sidebar.checkbox("Convert to Grayscale")
    if grayscale:
        image = convert_to_grayscale(image)

    edge_detection = st.sidebar.checkbox("Apply Edge Detection")
    if edge_detection:
        image = apply_edge_detection(image)

    contrast = st.sidebar.checkbox("Adjust Contrast")
    if contrast:
        contrast_factor = st.sidebar.slider("Contrast Factor", 0.1, 2.0, 1.0, 0.1)
        image = adjust_contrast(image, contrast_factor)

    shift = st.sidebar.checkbox("Shift Image")
    if shift:
        x_shift = st.sidebar.slider("Horizontal Shift (pixels)", -100, 100, 0)
        y_shift = st.sidebar.slider("Vertical Shift (pixels)", -100, 100, 0)
        image = shift_image(image, x_shift, y_shift)

    st.image(image, caption="Processed Image", width = 400)
    return image
    
def extract_brightness_features(image):
    image_np = np.array(image)
    if len(image_np.shape) == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    mean_brightness = np.mean(image_np)
    std_brightness = np.std(image_np)
    return np.array([[mean_brightness, std_brightness]])

def predict_brightness(image):
    try:
        X_train = np.array([[50, 20], [100, 30], [150, 40], [200, 50]])
        y_train = np.array([20, 40, 60, 80])
        model = LinearRegression()
        model.fit(X_train, y_train)
        features = extract_brightness_features(image)
        score = model.predict(features)[0]
        st.success("Brightness Score:"  + str(score))
        return score
    except Exception as e:
        st.error("Error in brightness prediction: " + str(e))
        return None

@st.cache_resource
def load_yolo_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.eval()
    return model

def predict_object_count(image):
    try:
        model = load_yolo_model()
        if image.mode != "RGB":
            image = image.convert("RGB")
        results = model(image)
        detections = results.xyxy[0].numpy()
        if len(detections) == 0:
            count = 0
            object_types = []
        else:
            class_ids = detections[:, 5]
            unique_classes = np.unique(class_ids)
            count = len(unique_classes)
            object_types = [results.names[int(cls)] for cls in unique_classes]

        st.success(f"Object Count: {count}")
        if object_types:
            st.write(f"Detected: {', '.join(object_types)}")
        return count, object_types
    except Exception as e:
        st.error(f"Error in object count prediction: {str(e)}")
        return None, None

@st.cache_resource
def load_classification_model():
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    return processor, model

def classify_image(image):
    try:
        processor, model = load_classification_model()
        if image.mode != "RGB":
            image = image.convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        top_k = torch.topk(probabilities, k=3, dim=-1)
        top_indices = top_k.indices[0].tolist()
        top_probs = top_k.values[0].tolist()

        human_keywords = ['person', 'man', 'woman', 'child', 'people', 'boy', 'girl', 'human', 'crowd', 'male', 'female']
        animal_keywords = ['dog', 'cat', 'horse', 'cow', 'pig', 'sheep', 'goat', 'deer', 'bear', 'lion', 'tiger', 
                          'elephant', 'wolf', 'fox', 'rabbit', 'squirrel', 'mouse', 'monkey', 'chimpanzee', 'gorilla', 
                          'kangaroo', 'zebra', 'giraffe', 'rhino', 'hippopotamus', 'cheetah', 'leopard', 'panda', 
                          'koala', 'raccoon', 'badger', 'weasel', 'otter', 'hyena', 'mongoose', 'lynx', 'bobcat', 
                          'labrador', 'retriever', 'poodle', 'chihuahua', 'bulldog', 'terrier', 'spaniel', 'shepherd']
        bird_keywords = ['bird', 'eagle', 'owl', 'parrot', 'penguin', 'hawk', 'falcon', 'sparrow', 'crow', 'pigeon', 
                         'seagull', 'duck', 'goose', 'swan', 'turkey', 'rooster', 'hen', 'peacock', 'flamingo', 'vulture']

        category = "other"
        top_labels = []
        for idx, prob in zip(top_indices, top_probs):
            label = model.config.id2label[idx].lower()
            top_labels.append(f"{label} ({prob:.2f})")
            if any(keyword in label for keyword in human_keywords):
                category = "human"
                break
            elif any(keyword in label for keyword in animal_keywords):
                category = "animal"
                break
            elif any(keyword in label for keyword in bird_keywords):
                category = "bird"
                break

        st.success(f"Classified as: {category}")
        st.write(f"Top-3 ImageNet labels: {', '.join(top_labels)}")
        return category
    except Exception as e:
        st.error(f"Error classifying image: {str(e)}")
        return None

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentiment(feedback):
    try:
        sentiment_analyzer = load_sentiment_model()
        result = sentiment_analyzer(feedback)[0]
        label, score = result['label'], result['score']
        st.success(f"Sentiment: {label} (Confidence: {score:.2f})")
        return label, score
    except Exception as e:
        st.error(f"Error analyzing sentiment: {str(e)}")
        return None, None

def main():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Quicksand', sans-serif;
            background: linear-gradient(to right, #fdfbfb, #ebedee);
        }

        .stApp {
            background: linear-gradient(to right, #f3e7e9, #e3eeff);
            padding: 20px;
            border-radius: 12px;
        }

        h1, h2, h3 {
            color: #393e46;
            text-align: center;
        }

        .stButton>button {
            background: linear-gradient(to right, #36d1dc, #5b86e5);
            color: white;
            border: none;
            border-radius: 20px;
            padding: 10px 20px;
            font-size: 16px;
            margin: 10px 0;
            transition: 0.3s ease;
        }

        .stButton>button:hover {
            transform: scale(1.05);
            background: linear-gradient(to right, #5b86e5, #36d1dc);
        }

        .stDownloadButton>button {
            background-color: #00b894;
            color: white;
            border-radius: 20px;
            padding: 10px;
        }

        .stDownloadButton>button:hover {
            background-color: #00cec9;
        }

        .stSidebar {
            background: linear-gradient(to bottom, #dff9fb, #c7ecee);
            border-radius: 8px;
        }

        .stTextArea textarea {
            border-radius: 10px;
            padding: 10px;
        }

        img {
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            border-radius: 10px;
        }

        .stSuccess {
            color: #2ecc71;
            font-weight: bold;
        }

        .stError {
            color: #e74c3c;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("\U0001F3A8 Image Processing and Analysis ")

    image = load_image()
    if image is not None:
        st.session_state["image"] = image

    if "image" in st.session_state:
        image = st.session_state["image"]
        processed_image = apply_image_processing(image)
        st.session_state["processed_image"] = processed_image

        if st.button("Predict Brightness"):
            predict_brightness(processed_image)

        if st.button("Predict Object Count"):
            predict_object_count(st.session_state["image"])

        if st.button("Classify Image"):
            classify_image(st.session_state["image"])

        feedback = st.text_area("Feedback", "This image looks great!")
        if st.button("Analyze Feedback"):
            analyze_sentiment(feedback)

        buffered = io.BytesIO()
        processed_image.save(buffered, format="PNG")
        st.download_button(
            label="Download Processed Image",
            data=buffered.getvalue(),
            file_name="processed_image.png",
            mime="image/png"
        )

if __name__ == "__main__":
    main()
