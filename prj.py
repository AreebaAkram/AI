import streamlit as st
import numpy as np
from PIL import Image
from transformers import pipeline
from textblob import TextBlob
from sklearn.neural_network import MLPRegressor
import pandas as pd
from io import StringIO
from datetime import datetime
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Product Review Analyzer", layout="wide")

# Custom CSS for minimalist, attractive design
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@300;400;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #f0f4ff 0%, #e6f0fa 100%);
        padding: 15px;
        border-radius: 10px;
        font-family: 'Open Sans', sans-serif;
    }
    .stButton>button {
        background: linear-gradient(45deg, #48cae4, #90e0ef);
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-family: 'Open Sans', sans-serif;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #00b4d8, #48cae4);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stButton>button::after {
        content: attr(data-tooltip);
        position: absolute;
        bottom: 120%;
        left: 50%;
        transform: translateX(-50%);
        background-color: #343a40;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 12px;
        white-space: nowrap;
        opacity: 0;
        transition: opacity 0.3s ease;
        pointer-events: none;
    }
    .stButton>button:hover::after {
        opacity: 1;
    }
    .card {
        background: #ffffff;
        padding: 12px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 10px;
        animation: fadeIn 0.5s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .title {
        color: #023e8a;
        font-size: 3em;
        font-family: 'Open Sans', sans-serif;
        font-weight: 700;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        color: #0077b6;
        font-size: 1.6em;
        font-family: 'Open Sans', sans-serif;
        font-weight: 600;
        margin: 10px 0;
    }
    .stTextInput>div>input, .stTextArea>div>textarea, .stNumberInput>div>input {
        background: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        color: #212529;
        font-family: 'Open Sans', sans-serif;
    }
    .stExpander {
        background: transparent;
        border: none;
    }
    .stExpander>div>details {
        background: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
    .spinner {
        border: 4px solid rgba(0, 119, 182, 0.2);
        border-top: 4px solid #0077b6;
        border-radius: 50%;
        width: 24px;
        height: 24px;
        animation: spin 1s linear infinite;
        margin: auto;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">🛒 Product Review Analyzer</div>', unsafe_allow_html=True)

# Initialize Hugging Face models
@st.cache_resource
def load_vqa_model():
    try:
        with st.spinner("Loading VQA model..."):
            st.markdown('<div class="spinner"></div>', unsafe_allow_html=True)
            vqa_pipe = pipeline("visual-question-answering", model="dandelin/vilt-b32-finetuned-vqa", framework="pt")
        st.success("VQA model loaded!")
        return vqa_pipe
    except Exception as e:
        st.error(f"Error loading VQA model: {e}")
        return None

@st.cache_resource
def load_caption_model():
    try:
        with st.spinner("Loading captioning model..."):
            st.markdown('<div class="spinner"></div>', unsafe_allow_html=True)
            caption_pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", framework="pt")
        st.success("Captioning model loaded!")
        return caption_pipe
    except Exception as e:
        st.error(f"Error loading captioning model: {e}")
        return None

vqa_pipeline = load_vqa_model()
caption_pipeline = load_caption_model()
if vqa_pipeline is None or caption_pipeline is None:
    st.stop()

# Price prediction with neural network
def predict_price(sentiment, review_length):
    try:
        features = np.array([[0.8, 50], [0.6, 30], [0.9, 70], [0.4, 20], [0.7, 40]])
        prices = np.array([199.99, 149.99, 249.99, 99.99, 179.99])
        model = MLPRegressor(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
        model.fit(features, prices)
        prediction = model.predict([[sentiment, review_length]])
        return round(max(50.0, prediction[0]), 2)
    except Exception as e:
        st.error(f"Error in price prediction: {e}")
        return None

# Rating prediction with neural network
def predict_rating(sentiment, review_length):
    try:
        features = np.array([[0.8, 50], [0.6, 30], [0.9, 70], [0.4, 20], [0.7, 40]])
        ratings = np.array([4.5, 3.8, 4.8, 3.2, 4.0])
        model = MLPRegressor(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
        model.fit(features, ratings)
        prediction = model.predict([[sentiment, review_length]])
        return round(max(1.0, min(5.0, prediction[0])), 2)
    except Exception as e:
        st.error(f"Error in rating prediction: {e}")
        return None

# Sale chart based on sentiment
def plot_sale_chart(sentiment):
    try:
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
        base_sales = [100, 120, 110, 130, 140]
        sentiment_factor = 1 + sentiment  # Positive sentiment boosts sales
        adjusted_sales = [sale * sentiment_factor for sale in base_sales]
        
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(months, adjusted_sales, marker='o', color='#0077b6', linewidth=2)
        ax.set_title('Hypothetical Sales Trend')
        ax.set_ylabel('Sales (Units)')
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting sale chart: {e}")

# Main layout (single column)
st.markdown('<div class="card">', unsafe_allow_html=True)

# Image upload
uploaded_image = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    max_width = 500
    ratio = max_width / image.width
    new_height = int(image.height * ratio)
    resized_image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)
    st.image(resized_image, caption="Uploaded Product", use_column_width=True)

    # Image captioning
    with st.expander("Product Description", expanded=True):
        if st.button("Generate Description", key="caption_button", help="Generate product description"):
            with st.spinner("Generating..."):
                st.markdown('<div class="spinner"></div>', unsafe_allow_html=True)
                caption = caption_pipeline(resized_image)[0]["generated_text"]
                st.markdown(f"**Description**: {caption}")

    # VQA section
    with st.expander("Ask About the Product"):
        question = st.text_input("", "What is this product?", key="vqa_input", label_visibility="collapsed")
        if st.button("Answer Question", key="vqa_button", help="Analyze with VQA"):
            with st.spinner("Analyzing..."):
                st.markdown('<div class="spinner"></div>', unsafe_allow_html=True)
                vqa_result = vqa_pipeline(resized_image, question, top_k=1)
                vqa_answer = vqa_result[0]["answer"]
                st.markdown(f"**Answer**: {vqa_answer}")

st.markdown('</div>', unsafe_allow_html=True)

# Review analysis
st.markdown('<div class="card">', unsafe_allow_html=True)
with st.expander("Analyze Reviews", expanded=True):
    reviews = st.text_area("", "Great product, love it! But delivery was slow.", key="review_input", label_visibility="collapsed")
    if st.button("Analyze Reviews", key="review_button", help="Check sentiment and correct text"):
        blob = TextBlob(reviews)
        corrected_reviews = blob.correct()
        sentiment = blob.sentiment.polarity
        review_length = len(reviews.split())
        st.markdown(f"**Corrected Reviews**: {corrected_reviews}")
        st.markdown(f"**Sentiment Score**: {sentiment:.2f} (Positive > 0, Negative < 0)")

        # Sentiment visualization
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.bar(['Sentiment'], [sentiment], color='#48cae4')
        ax.set_ylim(-1, 1)
        ax.set_ylabel('Score')
        st.pyplot(fig)

# Price prediction
with st.expander("Predict Price"):
    if st.button("Predict Price", key="price_button", help="Estimate product price"):
        if 'sentiment' in locals() and 'review_length' in locals():
            predicted_price = predict_price(sentiment, review_length)
            if predicted_price:
                st.markdown(f"**Predicted Price**: ${predicted_price:,.2f}")
        else:
            st.error("Analyze reviews first.")

# Rating prediction
with st.expander("Predict Rating"):
    if st.button("Predict Rating", key="rating_button", help="Estimate product rating"):
        if 'sentiment' in locals() and 'review_length' in locals():
            predicted_rating = predict_rating(sentiment, review_length)
            if predicted_rating:
                st.markdown(f"**Predicted Rating**: {predicted_rating}/5")
        else:
            st.error("Analyze reviews first.")

# Sale chart
with st.expander("Sale Trend"):
    if st.button("Show Sale Trend", key="sale_button", help="View sales based on sentiment"):
        if 'sentiment' in locals():
            plot_sale_chart(sentiment)
        else:
            st.error("Analyze reviews first.")

st.markdown('</div>', unsafe_allow_html=True)

# Downloadable summary
st.markdown('<div class="card">', unsafe_allow_html=True)
if uploaded_image and st.button("Generate Summary", key="summary_button", help="Download analysis as CSV"):
    summary_data = {
        "Feature": ["Timestamp", "Description", "VQA Answer", "Sentiment", "Predicted Price", "Predicted Rating"],
        "Result": [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            caption if 'caption' in locals() else "Not generated",
            vqa_answer if 'vqa_answer' in locals() else "Not generated",
            f"{sentiment:.2f}" if 'sentiment' in locals() else "Not analyzed",
            f"${predicted_price:,.2f}" if 'predicted_price' in locals() else "Not calculated",
            f"{predicted_rating}/5" if 'predicted_rating' in locals() else "Not calculated"
        ]
    }
    df = pd.DataFrame(summary_data)
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Summary CSV",
        data=csv,
        file_name=f"product_review_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        key="download_button",
        help="Save results as CSV"
    )
st.markdown('</div>', unsafe_allow_html=True)