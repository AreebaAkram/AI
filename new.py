import streamlit as st
import numpy as np
from PIL import Image
from transformers import pipeline
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
import pandas as pd
from io import StringIO
from datetime import datetime
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Product Review Analyzer", layout="wide")

# Custom CSS for minimalist, attractive design
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;500;600;700&display=swap');
    
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
    .progress-bar {
        background: #dee2e6;
        border-radius: 5px;
        height: 8px;
        overflow: hidden;
    }
    .progress-bar-fill {
        background: linear-gradient(90deg, #48cae4, #90e0ef);
        height: 100%;
        transition: width 0.5s ease;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vqa_answer' not in st.session_state:
    st.session_state.vqa_answer = None
if 'sentiment' not in st.session_state:
    st.session_state.sentiment = None
if 'predicted_price' not in st.session_state:
    st.session_state.predicted_price = None
if 'predicted_rating' not in st.session_state:
    st.session_state.predicted_rating = None

# Title
st.markdown('<div class="title">🛒 Product Review Analyzer</div>', unsafe_allow_html=True)

# Initialize Hugging Face VQA model
@st.cache_resource
def load_vqa_model():
    try:
        progress = st.progress(0)
        for i in range(100):
            progress.progress(i + 1)
            if i == 50:
                vqa_pipe = pipeline("visual-question-answering", model="dandelin/vilt-b32-finetuned-vqa", framework="pt")
        progress.empty()
        st.success("VQA model loaded!")
        return vqa_pipe
    except Exception as e:
        st.error(f"Error loading VQA model: {e}")
        return None

vqa_pipeline = load_vqa_model()
if vqa_pipeline is None:
    st.stop()

# Price prediction with linear regression
def predict_price(quality_score, demand_score):
    try:
        features = np.array([[8, 7], [6, 5], [9, 8], [4, 3], [7, 6]])  # [quality_score, demand_score]
        prices = np.array([199.99, 149.99, 249.99, 99.99, 179.99])
        model = LinearRegression()
        model.fit(features, prices)
        prediction = model.predict([[quality_score, demand_score]])
        return round(max(50.0, prediction[0]), 2)
    except Exception as e:
        st.error(f"Error in price prediction: {e}")
        return None

# Rating prediction with linear regression
def predict_rating(sentiment, review_length):
    try:
        # Normalize review_length to 0-100 range
        review_length = min(max(review_length, 0), 100)
        features = np.array([[0.8, 50], [0.6, 30], [0.9, 70], [0.4, 20], [0.7, 40]])  # [sentiment, review_length]
        ratings = np.array([4.5, 3.8, 4.8, 3.2, 4.0])
        model = LinearRegression()
        model.fit(features, ratings)
        prediction = model.predict([[sentiment, review_length]])
        return round(max(1.0, min(5.0, prediction[0])), 2)
    except Exception as e:
        st.error(f"Error in rating prediction: {e}")
        return None

# Sale chart based on user input
def plot_sale_chart(sales_data):
    try:
        months = ['Jan', 'February', 'March', 'April', 'May']
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(months, sales_data, marker='o', color='#0077b6', linewidth=2)
        ax.set_title('Monthly Sales Trend')
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
    max_width = 300  # Medium size
    ratio = max_width / image.width
    new_height = int(image.height * ratio)
    resized_image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)
    st.image(resized_image, caption="Uploaded Product")

    # VQA section
    with st.expander("Ask About the Product"):
        question = st.text_input("", "What is this product?", key="vqa_input", label_visibility="collapsed")
        if st.button("Answer Question", key="vqa_button", help="Analyze with VQA"):
            with st.spinner("Analyzing..."):
                st.markdown('<div class="spinner"></div>', unsafe_allow_html=True)
                vqa_result = vqa_pipeline(resized_image, question, top_k=1)
                st.session_state.vqa_answer = vqa_result[0]["answer"]
                st.markdown(f"**Answer**: {st.session_state.vqa_answer}")

st.markdown('</div>', unsafe_allow_html=True)

# Review analysis
st.markdown('<div class="card">', unsafe_allow_html=True)
with st.expander("Analyze Reviews", expanded=True):
    reviews = st.text_area("", "Great product, love it! But delivery was slow.", key="review_input", label_visibility="collapsed")
    if st.button("Analyze Reviews", key="review_button", help="Check sentiment and correct text"):
        blob = TextBlob(reviews)
        corrected_reviews = blob.correct()
        st.session_state.sentiment = blob.sentiment.polarity
        review_length = len(reviews.split())
        st.markdown(f"**Corrected Reviews**: {corrected_reviews}")
        st.markdown(f"**Sentiment Score**: {st.session_state.sentiment:.2f} (Positive > 0, Negative < 0)")

        # Sentiment visualization
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.bar(['Sentiment'], [st.session_state.sentiment], color='#48cae4')
        ax.set_ylim(-1, 1)
        ax.set_ylabel('Score')
        st.pyplot(fig)

# Rating prediction
with st.expander("Predict Rating"):
    if st.button("Predict Rating", key="rating_button", help="Estimate product rating"):
        if st.session_state.sentiment is not None:
            review_length = len(reviews.split()) if 'reviews' in locals() else 0
            predicted_rating = predict_rating(st.session_state.sentiment, review_length)
            if predicted_rating is not None:
                st.session_state.predicted_rating = predicted_rating
                st.markdown(f"**Predicted Rating**: {st.session_state.predicted_rating}/5")
            else:
                st.error("Rating prediction failed. Please try again with different review text.")
        else:
            st.error("Analyze reviews first.")

st.markdown('</div>', unsafe_allow_html=True)

# Price prediction
st.markdown('<div class="card">', unsafe_allow_html=True)
with st.expander("Predict Price"):
    quality_score = st.number_input("Product Quality Score (1–10):", min_value=1, max_value=10, value=5, step=1)
    demand_score = st.number_input("Demand Score (1–10):", min_value=1, max_value=10, value=5, step=1)
    if st.button("Predict Price", key="price_button", help="Estimate product price"):
        predicted_price = predict_price(quality_score, demand_score)
        if predicted_price:
            st.session_state.predicted_price = predicted_price
            st.markdown(f"**Predicted Price**: ${st.session_state.predicted_price:,.2f}")

st.markdown('</div>', unsafe_allow_html=True)

# Sale trend
st.markdown('<div class="card">', unsafe_allow_html=True)
with st.expander("Sale Trend"):
    st.markdown("Enter sales data for 5 months:")
    sales_data = []
    for i, month in enumerate(['Jan', 'Feb', 'Mar', 'Apr', 'May']):
        sales = st.number_input(f"{month} Sales (Units):", min_value=0, value=100, step=10, key=f"sales_{i}")
        sales_data.append(sales)
    if st.button("Show Sale Trend", key="sale_button", help="View sales trend"):
        plot_sale_chart(sales_data)

st.markdown('</div>', unsafe_allow_html=True)

# Downloadable summary
st.markdown('<div class="card">', unsafe_allow_html=True)
if uploaded_image and st.button("Generate Summary", key="summary_button", help="Download analysis as CSV"):
    summary_data = {
        "Feature": ["Timestamp", "VQA Answer", "Sentiment", "Predicted Price", "Predicted Rating"],
        "Result": [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            str(st.session_state.vqa_answer) if st.session_state.vqa_answer is not None else "Not generated",
            f"{st.session_state.sentiment:.2f}" if st.session_state.sentiment is not None else "Not analyzed",
            f"${st.session_state.predicted_price:,.2f}" if st.session_state.predicted_price is not None else "Not calculated",
            f"{st.session_state.predicted_rating}/5" if st.session_state.predicted_rating is not None else "Not calculated"
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