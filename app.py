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

MAX_TOKENS = 512

def count_tokens(text):
    tokens = tokenizer(text, truncation=False, add_special_tokens=True)
    return len(tokens["input_ids"])

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_TOKENS)
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

    /* ========== KILL EVERY DEFAULT WRAPPER ========== */

    /* Remove label "Enter your AirPods review here" bottom gap if needed */
    .stTextArea label {
        font-weight: 600;
    }

    /* Outermost Streamlit wrapper */
    .stTextArea > div {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
    }

    /* BaseWeb textarea wrapper — THIS is the culprit with the 4-corner border */
    .stTextArea [data-baseweb="textarea"] {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        border-radius: 0 !important;
        padding: 0 !important;
    }

    /* BaseWeb inner base-input container */
    .stTextArea [data-baseweb="base-input"] {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        border-radius: 0 !important;
        padding: 0 !important;
    }

    /* Remove the focused / active / hovered outlines on wrappers */
    .stTextArea [data-baseweb="textarea"]:focus-within,
    .stTextArea [data-baseweb="textarea"]:hover,
    .stTextArea [data-baseweb="textarea"]:active,
    .stTextArea [data-baseweb="base-input"]:focus-within,
    .stTextArea [data-baseweb="base-input"]:hover,
    .stTextArea [data-baseweb="base-input"]:active {
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }

    /* Nuke any nested divs inside that still carry borders */
    .stTextArea > div > div {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
    }

    .stTextArea > div > div > div {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
    }

    /* ========== STYLE ONLY THE ACTUAL TEXTAREA ========== */
    .stTextArea textarea {
        border-radius: 40px !important;
        border: 2px solid red !important;
        padding: 16px !important;
        background-color: #FFFFFF !important;
        box-shadow: 3px 3px 5px #9E9E9E !important;
        outline: none !important;
    }

    .stTextArea textarea:focus {
        border: 2px solid red !important;
        box-shadow: 3px 3px 5px #9E9E9E !important;
        outline: none !important;
    }

    .stTextArea textarea:hover {
        border: 2px solid red !important;
    }

    .stTextArea textarea::placeholder {
        color: #999;
        font-style: italic;
    }

    /* ========== AIRPODS IMAGE SHADOW CARDS ========== */
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
    """
    <h1 style="font-size: 45px; text-align: center;">Apple AirPods Sentiment Analysis</h1>
    """,
    unsafe_allow_html=True
)

# --- AirPods Image Row with Shadow Cards ---
image_urls = [
    "https://raw.githubusercontent.com/MarpakaPradeepSai/Project-Sentiment-Analysis/refs/heads/main/Data/Images%20%26%20GIFs/Image-1_Apple-AirPods-with-Charging-Case-2nd-Generation.avif ",
    "https://github.com/MarpakaPradeepSai/Project-Sentiment-Analysis/blob/main/Data/Images%20&%20GIFs/Image-2_Apple-AirPods-with-Charging-Case-2nd-Generation.webp?raw=true",
    "https://github.com/MarpakaPradeepSai/Project-Sentiment-Analysis/blob/main/Data/Images%20&%20GIFs/Image-3_Apple-AirPods-with-Charging-Case-2nd-Generation.webp?raw=true",
    "https://github.com/MarpakaPradeepSai/Project-Sentiment-Analysis/blob/main/Data/Images%20&%20GIFs/Image-4_Apple-AirPods-with-Charging-Case-2nd-Generation.webp?raw=true",
    "https://github.com/MarpakaPradeepSai/Project-Sentiment-Analysis/blob/main/Data/Images%20&%20GIFs/Image-5_Apple-AirPods-with-Charging-Case-2nd-Generation.webp?raw=true"
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

# --- User Input Text Area ---
user_input = st.text_area("Enter your AirPods review here", height=200)

st.write("")

# --- Analyze Sentiment Button ---
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    analyze_button = st.button("🔍 Analyze Sentiment")

if analyze_button:
    if not user_input.strip():
        st.error("⚠️ Please enter a review to analyze.")
    elif model is None or tokenizer is None:
        st.error("Model failed to load. Please check the repo ID.")
    else:
        num_tokens = count_tokens(user_input)
        if num_tokens > MAX_TOKENS:
            st.error(
                f"⚠️ Your review is too long! It has **{num_tokens} tokens**, but the model supports a maximum of **{MAX_TOKENS} tokens**. "
                f"Please try a shorter review."
            )
        else:
            with st.spinner("Analyzing sentiment..."):
                time.sleep(0.5)
                probs = predict_sentiment(user_input)
                label, bg_color = get_sentiment_info(probs)
                confidence = np.max(probs) * 100

            st.divider()

            label_text = label.split()[0]
            label_emoji = label.split()[1]

            st.markdown(
                f"""
                <div style="background-color:{bg_color}; padding: 15px; border-radius: 25px; text-align: center;" class="prediction-box">
                    <div style="font-size: 3em; margin-bottom: 0; line-height: 1;">{label_emoji}</div>
                    <h3 style="margin-top: 5px; margin-bottom: 5px;"><strong>Sentiment</strong>: {label_text}</h3>
                    <p style="font-size: 16px; margin: 0;">(Confidence: {confidence:.2f}%)</p>
                </div>
                """,
                unsafe_allow_html=True
            )
