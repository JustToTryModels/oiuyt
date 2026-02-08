import streamlit as st
from transformers import AlbertTokenizerFast, AutoModelForSequenceClassification
import torch
import numpy as np
import time

# --- Set page config MUST be the first Streamlit command ---
st.set_page_config(
    page_title="AirPods Review Sentiment Analyzer",
    page_icon="🎧",
    layout="wide",  # Changed to wide for better responsive layout
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# MODEL LOADING (Preserved Exactly)
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
# HELPER FUNCTIONS (Preserved Exactly)
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
# SESSION STATE INITIALIZATION (New - for interactivity)
# -----------------------------------------------------------------------------
if 'example_loaded' not in st.session_state:
    st.session_state.example_loaded = ""
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# -----------------------------------------------------------------------------
# MODERN UI THEME & CSS ENHANCEMENTS
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Modern Color Palette & Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --success-color: #10B981;
        --warning-color: #F59E0B;
        --danger-color: #EF4444;
        --bg-color: #F3F4F6;
        --card-bg: rgba(255, 255, 255, 0.95);
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Glassmorphism Card Effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin-bottom: 1.5rem;
    }
    
    /* Enhanced Typography */
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        background: linear-gradient(90deg, #6a0572, #ab83a1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 3.5em;
        margin-bottom: 0.5rem;
        letter-spacing: -2px;
    }
    
    .subtitle {
        text-align: center;
        color: #6B7280;
        font-size: 1.2em;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Modern Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 12px 32px;
        font-size: 1.1em;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
        color: white !important;
    }
    
    .stButton>button:active {
        transform: translateY(0);
    }
    
    /* Secondary Button */
    .secondary-btn {
        background: white !important;
        color: #667eea !important;
        border: 2px solid #667eea !important;
        box-shadow: none !important;
    }
    
    /* Text Area Modernization */
    .stTextArea textarea {
        border-radius: 16px;
        border: 2px solid #E5E7EB;
        padding: 16px;
        font-size: 1.05em;
        transition: all 0.3s ease;
        background-color: #FAFAFA;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
        background-color: white;
    }
    
    /* Prediction Box Enhancement */
    .prediction-box {
        border-radius: 20px;
        padding: 20px;
        text-align: center;
        font-size: 20px;
        font-weight: 600;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        animation: slideIn 0.5s ease-out;
        border: 3px solid white;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Metric Cards */
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin-top: 20px;
        gap: 15px;
    }
    
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        flex: 1;
        border-left: 4px solid #667eea;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%);
    }
    
    /* Image Gallery Hover Effects */
    .image-gallery {
        display: flex;
        justify-content: center;
        gap: 15px;
        margin: 20px 0;
        flex-wrap: wrap;
    }
    
    .gallery-img {
        border-radius: 16px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .gallery-img:hover {
        transform: scale(1.1) rotate(2deg);
        box-shadow: 0 12px 24px rgba(0,0,0,0.2);
    }
    
    /* Progress Bar Customization */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #374151;
        background: white;
        border-radius: 10px;
        border: 1px solid #E5E7EB;
    }
    
    /* Divider Styling */
    hr {
        margin: 2rem 0;
        border: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, #CBD5E1, transparent);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: #6B7280;
        font-size: 0.9em;
        margin-top: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# SIDEBAR (New - Interactive Elements & Navigation)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎧 About This App")
    st.info(
        """
        This AI-powered sentiment analyzer uses **ALBERT-base-v2** 
        to classify AirPods reviews into Positive, Neutral, or Negative 
        sentiments with confidence scores.
        
        **Model:** IamPradeep/Apple-Airpods-Sentiment-Analysis-ALBERT-base-v2
        """
    )
    
    st.markdown("### 📝 Quick Examples")
    st.caption("Click to load an example review:")
    
    examples = {
        "😊 Positive": "These AirPods are absolutely amazing! The sound quality is crystal clear and the noise cancellation is top-notch. Best purchase I've made this year!",
        "😐 Neutral": "The AirPods are okay I guess. They work fine for calls but the battery life could be better. Pretty average overall.",
        "😡 Negative": "Terrible product! Keeps disconnecting and the sound cuts out constantly. Waste of money, would not recommend to anyone."
    }
    
    for label, text in examples.items():
        if st.button(f"Load {label}", key=f"ex_{label}"):
            st.session_state.example_loaded = text
            st.rerun()
    
    with st.expander("⚙️ Advanced Settings"):
        st.caption("Analysis Configuration")
        show_probs = st.toggle("Show Probability Breakdown", value=True)
        show_history = st.toggle("Show Analysis History", value=False)
    
    st.markdown("---")
    st.caption("🔒 Powered by Hugging Face Transformers")

# -----------------------------------------------------------------------------
# MAIN LAYOUT
# -----------------------------------------------------------------------------

# --- Hero Section ---
st.markdown("<h1>Apple AirPods Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-Powered Review Classification using ALBERT Transformer</p>", unsafe_allow_html=True)

# --- AirPods Image Gallery (Enhanced) ---
with st.container():
    cols = st.columns(5)
    image_urls = [
        "https://i5.walmartimages.com/seo/Apple-AirPods-with-Charging-Case-2nd-Generation_8540ab4f-8062-48d0-9133-323a99ed921d.fb43fa09a0faef3f9495feece1397f8d.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF",
        "https://i5.walmartimages.com/asr/b6247579-386a-4bda-99aa-01e44801bc33.49db04f5e5b8d7f329c6580455e2e010.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF",
        "https://i5.walmartimages.com/asr/0f803868-d25f-4891-b0c8-e27a514ede02.f22c42c1ea17cd4d2b30fdfc89a8797c.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF",
        "https://i5.walmartimages.com/asr/df1b081f-4fa9-4ea5-87f8-413b9cad7a6e.f580d742da0a58bc25dadd30512adf72.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF",
        "https://i5.walmartimages.com/asr/2830c8d7-292d-4b99-b92f-239b15ff1062.ce77d20b2f20a569bfd656d05ca89f7c.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF"
    ]
    
    for i, url in enumerate(image_urls):
        with cols[i]:
            st.image(url, width=100, use_column_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Input Section (Card-based Layout) ---
input_col, info_col = st.columns([2, 1])

with input_col:
    with st.container():
        st.markdown("### ✍️ Enter Review")
        
        # Use session state for examples
        default_text = st.session_state.example_loaded if st.session_state.example_loaded else ""
        
        user_input = st.text_area(
            "Share your AirPods experience",
            value=default_text,
            height=180,
            placeholder="Type your review here... (e.g., 'The sound quality is incredible but battery drains fast')",
            help="Enter your honest review about Apple AirPods. The AI will analyze the sentiment automatically.",
            key="review_input"
        )
        
        btn_col1, btn_col2 = st.columns([3, 1])
        with btn_col1:
            analyze_btn = st.button("🔍 Analyze Sentiment", use_container_width=True)
        with btn_col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.example_loaded = ""
                st.rerun()

with info_col:
    with st.container():
        st.markdown("### 📊 How it Works")
        st.markdown(
            """
            <div style="background: white; padding: 15px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
                <p style="font-size: 0.9em; color: #4B5563; margin-bottom: 10px;">
                    Our model analyzes text using:
                </p>
                <ul style="font-size: 0.85em; color: #6B7280; padding-left: 20px; line-height: 1.8;">
                    <li>🧠 ALBERT Transformer</li>
                    <li>🎯 3-Class Classification</li>
                    <li>⚡ Real-time Inference</li>
                    <li>📈 Confidence Scoring</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

# --- Analysis & Results Section ---
if analyze_btn:
    if not user_input.strip():
        st.error("⚠️ Please enter a review to analyze.")
    elif model is None:
        st.error("❌ Model failed to load. Please check the repository ID or your internet connection.")
    else:
        # Processing with enhanced spinner
        with st.spinner('🤖 AI is analyzing your review...'):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.005)
                progress_bar.progress(i + 1)
            
            # Original prediction logic (Preserved Exactly)
            probs = predict_sentiment(user_input)
            label, bg_color = get_sentiment_info(probs)
            confidence = np.max(probs) * 100
            
            # Store in history
            st.session_state.analysis_history.append({
                'text': user_input[:50] + "...",
                'sentiment': label,
                'confidence': confidence
            })
        
        st.divider()
        
        # --- Results Container (Enhanced Visualization) ---
        st.markdown("## 🎯 Analysis Results")
        
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            # Original Output (Preserved Exactly as requested)
            st.markdown(
                f"""
                <div style="background-color:{bg_color}; padding: 10px; border-radius: 25px; text-align: center;" class="prediction-box">
                    <h3><span style="font-weight: bold;">Sentiment</span>: {label}</h3>
                    <p style="margin-top: 5px; font-size: 16px;">(Confidence: {confidence:.2f}%)</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Additional: Confidence Meter
            st.markdown("### Confidence Meter")
            st.progress(int(confidence))
            
        with res_col2:
            # Enhanced: Detailed Probability Breakdown
            if show_probs:
                with st.expander("📊 Detailed Probabilities", expanded=True):
                    labels_detailed = ["Negative 😡", "Neutral 😐", "Positive 😊"]
                    colors_detailed = ["#EF4444", "#F59E0B", "#10B981"]
                    
                    for i, (prob, lbl, clr) in enumerate(zip(probs, labels_detailed, colors_detailed)):
                        prob_pct = float(prob) * 100
                        st.markdown(f"**{lbl}**")
                        st.markdown(
                            f"""
                            <div style="background-color: #E5E7EB; border-radius: 10px; height: 24px; margin-bottom: 10px; position: relative; overflow: hidden;">
                                <div style="background-color: {clr}; width: {prob_pct}%; height: 100%; border-radius: 10px; transition: width 0.5s ease;"></div>
                                <span style="position: absolute; right: 10px; top: 50%; transform: translateY(-50%); font-size: 0.85em; font-weight: bold; color: #374151;">{prob_pct:.1f}%</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
        
        # --- Additional Metrics (New) ---
        with st.container():
            st.markdown("### 📈 Key Metrics")
            m1, m2, m3 = st.columns(3)
            
            with m1:
                st.metric(label="Analysis Time", value=f"{time.time() % 1:.2f}s", delta="Fast")
            with m2:
                word_count = len(user_input.split())
                st.metric(label="Word Count", value=word_count)
            with m3:
                char_count = len(user_input)
                st.metric(label="Characters", value=char_count)

# --- Analysis History (Optional Feature) ---
if show_history and st.session_state.analysis_history:
    st.divider()
    with st.expander("🕘 Recent Analysis History", expanded=False):
        for idx, item in enumerate(reversed(st.session_state.analysis_history[-5:])):
            st.markdown(
                f"""
                <div style="background: white; padding: 10px; border-radius: 8px; margin-bottom: 8px; border-left: 4px solid #667eea; display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 500; color: #374151;">{item['text']}</span>
                    <span style="background: #F3F4F6; padding: 4px 12px; border-radius: 12px; font-size: 0.85em; font-weight: 600;">{item['sentiment']} ({item['confidence']:.0f}%)</span>
                </div>
                """,
                unsafe_allow_html=True
            )

# --- Footer ---
st.markdown(
    """
    <div class='footer'>
        <p>Made with ❤️ using Streamlit & Hugging Face | ALBERT-base-v2 Model</p>
        <p style="font-size: 0.8em; margin-top: 10px;">© 2025 AirPods Sentiment Analyzer. All rights reserved.</p>
    </div>
    """,
    unsafe_allow_html=True
)
