import streamlit as st
from transformers import AlbertTokenizerFast, AutoModelForSequenceClassification
import torch
import numpy as np
import time

# --- Set page config MUST be the first Streamlit command ---
st.set_page_config(
    page_title="AirPods Sentiment AI",
    page_icon="🎧",
    layout="centered",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# MODEL LOADING (Logic Preserved)
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
# HELPER FUNCTIONS (Logic Preserved)
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
# UI & CSS ENHANCEMENTS
# -----------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* Import Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Poppins:wght@700&display=swap');

    /* General App Styling */
    .stApp {
        background-color: #FAFAFA;
        font-family: 'Inter', sans-serif;
    }

    /* Titles and Headers */
    h1, h2, h3 {
        font-family: 'Poppins', sans-serif;
        color: #1D1D1F;
    }
    
    /* Custom Card Container */
    .css-card {
        background-color: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border: 1px solid #E5E5E5;
    }

    /* Text Area Styling */
    .stTextArea textarea {
        border-radius: 12px;
        border: 1px solid #D1D1D6;
        padding: 15px;
        background-color: #FFFFFF;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);
        font-size: 16px;
        transition: border-color 0.3s;
    }
    .stTextArea textarea:focus {
        border-color: #007AFF;
        box-shadow: 0 0 0 2px rgba(0,122,255,0.2);
    }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #007AFF, #5856D6);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-size: 1.1em;
        font-weight: 600;
        cursor: pointer;
        width: 100%;
        box-shadow: 0 4px 14px rgba(0, 118, 255, 0.39);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 118, 255, 0.23);
    }
    .stButton>button:active {
        transform: scale(0.98);
    }

    /* Prediction Box Styling */
    .prediction-card {
        padding: 20px; 
        border-radius: 20px; 
        text-align: center;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        animation: fadeIn 0.8s ease-in-out;
        border: 2px solid rgba(255,255,255,0.5);
    }

    /* Image Gallery Hover */
    img {
        border-radius: 10px;
        transition: transform 0.3s ease;
    }
    img:hover {
        transform: scale(1.05);
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Apple_logo_grey.svg/172px-Apple_logo_grey.svg.png", width=50)
    st.title("About App")
    st.markdown("""
    This application uses a fine-tuned **ALBERT** model to analyze customer reviews for Apple AirPods.
    
    **How to use:**
    1. Paste a review in the text box.
    2. Click **Analyze Sentiment**.
    3. View the classification and confidence score.
    """)
    
    st.divider()
    st.caption(f"Model: `{MODEL_REPO}`")
    st.caption("v1.0.0 | Enhanced UI")

# -----------------------------------------------------------------------------
# MAIN LAYOUT
# -----------------------------------------------------------------------------

# --- Hero Section ---
st.markdown("<h1 style='text-align: center; margin-bottom: 0px;'>🎧 AirPods Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; font-size: 1.1em;'>Advanced Sentiment Analysis for Customer Reviews</p>", unsafe_allow_html=True)

st.write("") # Spacer

# --- Product Gallery (Containerized) ---
with st.expander("📸 View Product Gallery", expanded=True):
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
            st.image(url, use_container_width=True)

st.write("") # Spacer

# --- Main Input Card ---
st.markdown('<div class="css-card">', unsafe_allow_html=True)
st.subheader("💬 Review Input")
user_input = st.text_area("Paste your review below:", height=150, placeholder="e.g., The sound quality is amazing, but the battery life could be better...")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_btn = st.button("🔍 Analyze Sentiment")
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# LOGIC & RESULTS
# -----------------------------------------------------------------------------

if analyze_btn: 
    if not user_input.strip():
        st.warning("⚠️ Please enter a review to analyze.")
    elif model is None:
        st.error("Model failed to load. Please check the repo ID.")
    else:
        with st.spinner('Running ALBERT Model...'): 
            time.sleep(0.5) # Simulate processing time
            
            # Predict
            probs = predict_sentiment(user_input)
            label, bg_color = get_sentiment_info(probs)
            confidence = np.max(probs)
            confidence_pct = confidence * 100

        # --- Output Section ---
        st.markdown(f"### 📊 Analysis Results")
        
        # Primary Result Card
        st.markdown(
            f"""
            <div style="background-color:{bg_color};" class="prediction-card">
                <h2 style="margin:0; color:#333;">{label}</h2>
                <p style="margin-top: 10px; font-size: 18px; color:#555;">Confidence Score: <b>{confidence_pct:.2f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.write("")
        
        # Visual Progress Bar for Confidence
        st.progress(float(confidence))

        # Advanced Details (Expander for interactivity)
        with st.expander("🔬 View Technical Details"):
            st.info("Raw probability distribution from the ALBERT model:")
            
            # Creating a clean display for all probabilities
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(label="Negative", value=f"{probs[0]*100:.1f}%")
            with c2:
                st.metric(label="Neutral", value=f"{probs[1]*100:.1f}%")
            with c3:
                st.metric(label="Positive", value=f"{probs[2]*100:.1f}%")
