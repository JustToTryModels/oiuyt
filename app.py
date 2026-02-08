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
# Your Hugging Face Repo ID
MODEL_REPO = "IamPradeep/Apple-Airpods-Sentiment-Analysis-ALBERT-base-v2"

@st.cache_resource
def load_model():
    try:
        # Load directly from Hugging Face Hub
        tokenizer = AlbertTokenizerFast.from_pretrained(MODEL_REPO)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model from Hugging Face: {e}")
        return None, None

# Load the model
tokenizer, model = load_model()

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def predict_sentiment(text):
    # Prepare input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
    # Probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return probs.detach().numpy()[0]

def get_sentiment_info(probs):
    # Labels corresponding to model output: 0->Negative, 1->Neutral, 2->Positive
    labels = ["Negative 😡", "Neutral 😐", "Positive 😊"]
    
    # Original reference colors
    colors = ["#F5C6CB", "#FFE8A1", "#C3E6CB"] 
    
    max_index = np.argmax(probs)
    return labels[max_index], colors[max_index]

# -----------------------------------------------------------------------------
# UI & CSS (Matched exactly to reference)
# -----------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* Import Google Fonts - Keeping Nunito and Open Sans for general text */
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@700&family=Open+Sans:wght@400;600&display=swap');

    .main {
        background-color: #F0F2F6; /* Original main background color */
        font-family: 'Open Sans', sans-serif; /* Keep Open Sans for body */
        color: #333;
    }
    h1 {
        font-family: 'Nunito', sans-serif; /* Keep Nunito for title */
        color: #6a0572; /* Original title color */
        text-align: center;
        font-size: 3em; /* Original title size */
        margin-bottom: 15px;
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3); /* Original text shadow */
    }
    .stButton>button {
        background: linear-gradient(90deg, #ff8a00, #e52e71); /* Original button gradient */
        color: white !important;
        border: none;
        border-radius: 25px; /* Original button border-radius */
        padding: 10px 20px;
        font-size: 1.2em; /* Original button font-size */
        font-weight: bold; /* Original button font-weight */
        cursor: pointer;
        transition: transform 0.2s ease, box-shadow 0.2s ease; /* Original button transition */
        width: 100%;
    }
    .stButton>button:hover {
        transform: scale(1.05); /* Original button hover transform */
        box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.3); /* Original button hover box-shadow */
        color: white !important;
    }
    .prediction-box {
        border-radius: 25px; /* Original prediction box border-radius */
        padding: 10px; /* Original prediction box padding */
        text-align: center; /* Original prediction box text-align */
        font-size: 18px; /* Original prediction box font-size */
    }
    .stTextArea textarea {
        border-radius: 15px; /* Keep text area border-radius */
        border: 1px solid #ced4da; /* Keep text area border */
        padding: 10px; /* Keep text area padding */
        background-color: #FFFFFF; /* Keep text area background */
        box-shadow: 3px 3px 5px #9E9E9E; /* Keep text area shadow */
    }
    .stTextArea textarea::placeholder {
        color: #999; /* Light gray placeholder text */
        font-style: italic; /* Italic placeholder text */
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

# --- AirPods Image Row ---
image_urls = [
    "https://i5.walmartimages.com/seo/Apple-AirPods-with-Charging-Case-2nd-Generation_8540ab4f-8062-48d0-9133-323a99ed921d.fb43fa09a0faef3f9495feece1397f8d.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF",
    "https://i5.walmartimages.com/asr/b6247579-386a-4bda-99aa-01e44801bc33.49db04f5e5b8d7f329c6580455e2e010.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF",
    "https://i5.walmartimages.com/asr/0f803868-d25f-4891-b0c8-e27a514ede02.f22c42c1ea17cd4d2b30fdfc89a8797c.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF",
    "https://i5.walmartimages.com/asr/df1b081f-4fa9-4ea5-87f8-413b9cad7a6e.f580d742da0a58bc25dadd30512adf72.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF",
    "https://i5.walmartimages.com/asr/2830c8d7-292d-4b99-b92f-239b15ff1062.ce77d20b2f20a569bfd656d05ca89f7c.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF"
]

cols = st.columns(5)
for i, url in enumerate(image_urls):
    with cols[i]:
        st.image(url, width=100)

st.write("") # Spacer

# --- User Input Text Area ---
user_input = st.text_area("Enter your AirPods review here", height=150)

st.write("") # Spacer

# --- Analyze Sentiment Button ---
if st.button("🔍 Analyze Sentiment"): 
    if not user_input.strip():
        st.error("⚠️ Please enter a review to analyze.")
    elif model is None:
        st.error("Model failed to load. Please check the repo ID.")
    else:
        with st.spinner('Analyzing sentiment...'): 
            time.sleep(0.5) # Simulate processing time
            
            # Predict
            probs = predict_sentiment(user_input)
            label, bg_color = get_sentiment_info(probs)
            confidence = np.max(probs) * 100

        st.divider() 
        
        # --- Output Section (Restored to reference format) ---
        st.markdown(
            f"""
            <div style="background-color:{bg_color}; padding: 10px; border-radius: 25px; text-align: center;" class="prediction-box">
                <h3><span style="font-weight: bold;">Sentiment</span>: {label}</h3>
                <p style="margin-top: 5px; font-size: 16px;">(Confidence: {confidence:.2f}%)</p>
            </div>
            """,
            unsafe_allow_html=True
        )
