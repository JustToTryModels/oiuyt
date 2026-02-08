import streamlit as st
from transformers import AlbertTokenizerFast, AutoModelForSequenceClassification
import torch
import numpy as np
import time

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AirPods Review Analyzer",
    page_icon="🎧",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# CUSTOM CSS & STYLING (The UI Overhaul)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    /* General Body Styling */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Titles and Headers */
    h1 {
        color: #1a1a1a;
        font-weight: 700;
        letter-spacing: -1px;
        text-align: center;
        margin-bottom: 0px;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }

    /* Product Image Styling */
    .product-img {
        border-radius: 15px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        background: white;
        padding: 5px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .product-img:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.15);
    }

    /* Text Area Styling */
    .stTextArea textarea {
        background-color: #ffffff;
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        font-size: 16px;
        color: #333;
        transition: border-color 0.3s;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);
    }
    .stTextArea textarea:focus {
        border-color: #5e60ce;
        box-shadow: 0 0 0 2px rgba(94, 96, 206, 0.2);
    }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #5e60ce, #6930c3);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 12px 30px;
        font-size: 1.1rem;
        font-weight: 600;
        width: 100%;
        box-shadow: 0 4px 15px rgba(105, 48, 195, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(105, 48, 195, 0.6);
        background: linear-gradient(90deg, #6930c3, #5e60ce);
    }

    /* Result Card Styling */
    .result-card {
        background-color: white;
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 20px;
        animation: fadeIn 0.8s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Custom Progress Bars Labels */
    .metric-label {
        font-size: 0.9rem;
        font-weight: 600;
        color: #555;
        margin-bottom: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# MODEL LOADING (Unchanged Logic)
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
# HELPER FUNCTIONS (Unchanged Logic)
# -----------------------------------------------------------------------------
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return probs.detach().numpy()[0]

def get_sentiment_info(probs):
    labels = ["Negative", "Neutral", "Positive"]
    # Modern pastel colors
    colors = ["#ffadad", "#fdffb6", "#caffbf"] 
    # Stronger text colors
    text_colors = ["#c0392b", "#d35400", "#27ae60"]
    
    max_index = np.argmax(probs)
    return labels[max_index], colors[max_index], text_colors[max_index]

# -----------------------------------------------------------------------------
# MAIN APP LAYOUT
# -----------------------------------------------------------------------------

# --- Header Section ---
st.markdown("<h1>🎧 AirPods Insight</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Advanced AI Sentiment Analysis for Customer Reviews</p>", unsafe_allow_html=True)

# --- Product Showcase (Interactive Hover) ---
# Wrapping images in a container for better spacing
with st.container():
    image_urls = [
        "https://i5.walmartimages.com/seo/Apple-AirPods-with-Charging-Case-2nd-Generation_8540ab4f-8062-48d0-9133-323a99ed921d.fb43fa09a0faef3f9495feece1397f8d.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF",
        "https://i5.walmartimages.com/asr/b6247579-386a-4bda-99aa-01e44801bc33.49db04f5e5b8d7f329c6580455e2e010.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF",
        "https://i5.walmartimages.com/asr/0f803868-d25f-4891-b0c8-e27a514ede02.f22c42c1ea17cd4d2b30fdfc89a8797c.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF",
        "https://i5.walmartimages.com/asr/df1b081f-4fa9-4ea5-87f8-413b9cad7a6e.f580d742da0a58bc25dadd30512adf72.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF",
        "https://i5.walmartimages.com/asr/2830c8d7-292d-4b99-b92f-239b15ff1062.ce77d20b2f20a569bfd656d05ca89f7c.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF"
    ]
    
    # Using columns for the grid
    cols = st.columns(5)
    for i, url in enumerate(image_urls):
        with cols[i]:
            # Apply custom class for hover effect
            st.markdown(f'<img src="{url}" class="product-img" width="100%">', unsafe_allow_html=True)

st.write("") # Spacer

# --- Input Section ---
user_input = st.text_area(
    "Paste a review to analyze:", 
    placeholder="e.g., The sound quality is amazing, but the battery life could be better...",
    height=120
)

# --- Action & Results ---
if st.button("✨ Analyze Sentiment"): 
    if not user_input.strip():
        st.warning("⚠️ Please enter some text first.")
    elif model is None:
        st.error("❌ Model unavailable.")
    else:
        with st.spinner('🤖 AI is processing the review...'): 
            time.sleep(0.6) # Slight delay for UX effect
            
            # Predict
            probs = predict_sentiment(user_input)
            label, bg_color, text_color = get_sentiment_info(probs)
            confidence = np.max(probs) * 100
            
            # Define icons based on sentiment
            icon = "😐"
            if label == "Negative": icon = "😡"
            elif label == "Positive": icon = "🥰"

        # --- Beautiful Result Card ---
        st.markdown(
            f"""
            <div class="result-card" style="border-top: 5px solid {text_color};">
                <h4 style="color: #888; margin:0; font-size: 14px; text-transform: uppercase; letter-spacing: 1px;">Detected Sentiment</h4>
                <div style="font-size: 50px; margin: 10px 0;">{icon}</div>
                <h2 style="color: {text_color}; margin: 0; font-size: 32px;">{label}</h2>
                <p style="color: #666; margin-top: 10px;">Confidence Score: <b>{confidence:.2f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # --- Interactive Probability Breakdown ---
        st.write("")
        st.write("")
        st.markdown("##### 📊 Detailed Probability Breakdown")
        
        # Create 3 columns for the progress bars
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown(f"<div class='metric-label' style='color:#e74c3c'>Negative</div>", unsafe_allow_html=True)
            st.progress(float(probs[0]))
            st.caption(f"{probs[0]*100:.1f}%")
            
        with c2:
            st.markdown(f"<div class='metric-label' style='color:#f39c12'>Neutral</div>", unsafe_allow_html=True)
            st.progress(float(probs[1]))
            st.caption(f"{probs[1]*100:.1f}%")

        with c3:
            st.markdown(f"<div class='metric-label' style='color:#27ae60'>Positive</div>", unsafe_allow_html=True)
            st.progress(float(probs[2]))
            st.caption(f"{probs[2]*100:.1f}%")

        # Fun effects based on result
        if label == "Positive":
            st.balloons()
        elif label == "Negative":
            st.toast("Ouch! Sounds like a bad experience.", icon="🩹")
