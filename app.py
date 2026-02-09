import streamlit as st
from transformers import AlbertTokenizerFast, AutoModelForSequenceClassification
import torch
import numpy as np
import time

# --- Set page config MUST be the first Streamlit command ---
st.set_page_config(
    page_title="AirPods Review Sentiment Analyzer",
    page_icon="🎧",
    layout="centered",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# MODEL LOADING
# -----------------------------------------------------------------------------
MODEL_REPO = "IamPradeep/Apple-Airpods-Sentiment-Analysis-ALBERT-base-v2"

@st.cache_resource
def load_model():
    try:
        tokenizer = AlbertTokenizerFast.from_pretrained(MODEL_REPO)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model from Hugging Face: {e}")
        return None, None

tokenizer, model = load_model()

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return probs.detach().numpy()[0]

def get_sentiment_info(probs):
    labels = ["Negative 😡", "Neutral 😐", "Positive 😊"]
    colors = ["#F5C6CB", "#FFE8A1", "#C3E6CB"]
    max_index = np.argmax(probs)
    return labels[max_index], colors[max_index]

# -----------------------------------------------------------------------------
# UI & CSS
# -----------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@700&family=Open+Sans:wght@400;600&display=swap');

    .main {
        background-color: #F0F2F6;
        font-family: 'Open Sans', sans-serif;
        color: #333;
    }

    h1 {
        font-family: 'Nunito', sans-serif;
        color: #6a0572;
        text-align: center;
        font-size: 3em;
        margin-bottom: 15px;
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
    }

    .stButton > button {
        background: linear-gradient(90deg, #ff8a00, #e52e71);
        color: white !important;
        border: none;
        border-radius: 25px;
        padding: 10px 20px;
        font-size: 1.2em;
        font-weight: bold;
        cursor: pointer;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        width: 100%;
    }

    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.3);
        color: white !important;
    }

    .prediction-box {
        border-radius: 25px;
        padding: 10px;
        text-align: center;
        font-size: 18px;
    }

    .stTextArea textarea {
        border-radius: 40px !important;
        border: 2px solid red !important;
        padding: 16px !important;
        background-color: #FFFFFF !important;
        box-shadow: 3px 3px 5px #9E9E9E !important;
        outline: none !important;
    }

    .airpod-card {
        background-color: #FFFFFF;
        border-radius: 15px;
        padding: 10px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.2);
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .airpod-card:hover {
        transform: translateY(-5px);
        box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.3);
    }

    .airpod-card img {
        border-radius: 10px;
        width: 100px;
        height: 100px;
        object-fit: contain;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- App Title ---
st.markdown(
    "<h1>Apple AirPods Sentiment Analysis</h1>",
    unsafe_allow_html=True
)

# --- AirPods Image Row ---
image_urls = [
    "https://i5.walmartimages.com/seo/Apple-AirPods-with-Charging-Case-2nd-Generation_8540ab4f-8062-48d0-9133-323a99ed921d.fb43fa09a0faef3f9495feece1397f8d.jpeg",
    "https://i5.walmartimages.com/asr/b6247579-386a-4bda-99aa-01e44801bc33.jpeg",
    "https://i5.walmartimages.com/asr/0f803868-d25f-4891-b0c8-e27a514ede02.jpeg",
    "https://i5.walmartimages.com/asr/df1b081f-4fa9-4ea5-87f8-413b9cad7a6e.jpeg",
    "https://i5.walmartimages.com/asr/2830c8d7-292d-4b99-b92f-239b15ff1062.jpeg"
]

cols = st.columns(5)
for i, url in enumerate(image_urls):
    with cols[i]:
        st.markdown(
            f"""
            <div class="airpod-card">
                <img src="{url}" alt="AirPods {i+1}">
            </div>
            """,
            unsafe_allow_html=True
        )

st.write("")

# --- User Input ---
user_input = st.text_area("Enter your AirPods review here", height=200)

st.write("")

# --- Analyze Button ---
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    analyze_button = st.button("🔍 Analyze Sentiment")

if analyze_button:
    if not user_input.strip():
        st.error("⚠️ Please enter a review to analyze.")
    elif model is None:
        st.error("Model failed to load. Please check the repo ID.")
    else:
        with st.spinner("Analyzing sentiment..."):
            time.sleep(0.5)
            probs = predict_sentiment(user_input)
            label, bg_color = get_sentiment_info(probs)
            confidence = np.max(probs) * 100

        st.divider()

        label_text = label.split()[0]
        label_emoji = label.split()[1]

        # ✅ FIXED SPACING HERE
        st.markdown(
    f"""
    <div style="background-color:{bg_color}; padding: 12px 15px; border-radius: 25px; text-align: center;" class="prediction-box">
        <span style="font-size: 2.5em; vertical-align: middle;">{label_emoji}</span>
        <span style="font-size: 1.3em; font-weight: bold; vertical-align: middle; margin-left: 10px;">Sentiment: {label_text}</span>
        <p style="font-size: 14px; margin: 5px 0 0 0; color: #555;">(Confidence: {confidence:.2f}%)</p>
    </div>
    """,
    unsafe_allow_html=True)
