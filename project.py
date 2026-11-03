import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.linear_model import LinearRegression
import io

# Module 1: Image Loading
def load_image():
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        return image
    return None

# Module 2: Enhanced Image Processing
def adjust_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def adjust_contrast(image, factor):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def apply_edge_detection(image):
    # Convert PIL image to OpenCV format
    image_np = np.array(image)
    if len(image_np.shape) == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(image_np, 100, 200)
    return Image.fromarray(edges)

def convert_to_grayscale(image):
    return image.convert("L")

def apply_image_edits(image):
    st.sidebar.header("Image Editing Options")
    brightness_factor = st.sidebar.slider("Brightness", 0.1, 2.0, 1.0, 0.1)
    contrast_factor = st.sidebar.slider("Contrast", 0.1, 2.0, 1.0, 0.1)
    grayscale = st.sidebar.checkbox("Convert to Grayscale")
    edge_detection = st.sidebar.checkbox("Apply Edge Detection")
    
    edited_image = image
    if brightness_factor != 1.0:
        edited_image = adjust_brightness(edited_image, brightness_factor)
    if contrast_factor != 1.0:
        edited_image = adjust_contrast(edited_image, contrast_factor)
    if grayscale:
        edited_image = convert_to_grayscale(edited_image)
    if edge_detection:
        edited_image = apply_edge_detection(edited_image)
    
    st.image(edited_image, caption="Edited Image", use_column_width=True)
    return edited_image

# Module 3: Linear Regression for Brightness Prediction
def extract_image_features(image):
    image_np = np.array(image)
    if len(image_np.shape) == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    mean_brightness = np.mean(image_np)
    std_brightness = np.std(image_np)
    return np.array([[mean_brightness, std_brightness]])

def predict_brightness_score(image):
    # Simulated training data: [mean_brightness, std_brightness], brightness_score
    X_train = np.array([[50, 20], [100, 30], [150, 40], [200, 50]])
    y_train = np.array([20, 40, 60, 80])  # Arbitrary brightness scores
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    features = extract_image_features(image)
    score = model.predict(features)[0]
    return score

# Module 4: Neural Network for Image Classification
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes for CIFAR-10
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

@st.cache_resource
def load_trained_model():
    model = SimpleCNN()
    # Load a small subset of CIFAR-10 for demo purposes
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    model.eval()
    return model, transform

def classify_image(image):
    model, transform = load_trained_model()
    cifar10_classes = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    # Preprocess image
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return cifar10_classes[predicted.item()]

# Module 5: Main Application
def main():
    st.title("Enhanced Photo Editor and Classifier")
    st.write("Upload an image, apply edits, predict brightness, and classify it using a neural network.")
    
    image = load_image()
    if image is not None:
        edited_image = apply_image_edits(image)
        
        if st.button("Predict Brightness"):
            score = predict_brightness_score(edited_image)
            st.success(f"Predicted Brightness Score: {score:.2f}")
        
        if st.button("Classify Image"):
            with st.spinner("Classifying..."):
                label = classify_image(edited_image)
                st.success(f"Predicted Class: {label}")
        
        # Option to download edited image
        buffered = io.BytesIO()
        edited_image.save(buffered, format="PNG")
        st.download_button(
            label="Download Edited Image",
            data=buffered.getvalue(),
            file_name="edited_image.png",
            mime="image/png"
        )

if __name__ == "__main__":
    main()