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
# CUSTOM CSS & THEME
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Import Modern Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    /* Global Theme */
    .main {
        background-color: #f8f9fa; /* Soft light gray background */
        font-family: 'Poppins', sans-serif;
        color: #333;
    }
    
    /* Container Styling */
    .stApp {
        background-color: #f8f9fa;
    }

    /* Sidebar Styling */
    .css-1d391kg {
        background-color: #ffffff;
        border-right: 1px solid #eaeaea;
    }

    /* Header Styling */
    h1 {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        color: #1f2937;
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* Card Component Styling */
    .card {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 1.5rem;
        border: 1px solid #f3f4f6;
    }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%); /* Modern Indigo to Purple */
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-size: 1.1rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6);
    }

    /* Text Area Styling */
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #e5e7eb;
        padding: 15px;
        background-color: #f9fafb;
        color: #374151;
        font-family: 'Poppins', sans-serif;
        transition: border-color 0.3s;
    }
    .stTextArea textarea:focus {
        border-color: #6366f1;
        background-color: #ffffff;
        outline: none;
    }

    /* Image Gallery Styling */
    .img-container {
        display: flex;
        justify-content: center;
        gap: 15px;
        margin-bottom: 2rem;
        padding: 1rem;
        background: white;
        border-radius: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .img-container img {
        border-radius: 10px;
        transition: transform 0.2s;
        object-fit: contain;
        background: #fff;
    }
    .img-container img:hover {
        transform: scale(1.05);
        cursor: pointer;
    }

    /* Result Badge */
    .result-badge {
        font-size: 1.5rem;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 50px;
        display: inline-block;
        margin-bottom: 10px;
        color: #374151;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# MODEL LOADING (Logic Unchanged)
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
# HELPER FUNCTIONS (Logic Unchanged)
# -----------------------------------------------------------------------------

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return probs.detach().numpy()[0]

def get_sentiment_info(probs):
    # Labels corresponding to model output
    labels = ["Negative 😡", "Neutral 😐", "Positive 😊"]
    # Original reference colors preserved (for functional consistency)
    colors = ["#F5C6CB", "#FFE8A1", "#C3E6CB"] 
    
    max_index = np.argmax(probs)
    return labels[max_index], colors[max_index]

# -----------------------------------------------------------------------------
# SIDEBAR CONTENT
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.info(
        """
        This app uses a fine-tuned **ALBERT** model to analyze the sentiment of Apple AirPods reviews.
        
        **Model:** `Apple-Airpods-Sentiment-Analysis-ALBERT-base-v2`
        """
    )
    st.markdown("---")
    st.markdown("### 🏷️ Labels")
    st.markdown("- 😡 **Negative**")
    st.markdown("- 😐 **Neutral**")
    st.markdown("- 😊 **Positive**")
    st.markdown("---")
    st.markdown("<small>Built with ❤️ using Streamlit & Hugging Face</small>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MAIN APP LAYOUT
# -----------------------------------------------------------------------------

# --- Header Section ---
st.markdown('<h1>AirPods Review Analysis</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter a review below to detect sentiment instantly</p>', unsafe_allow_html=True)

# --- Image Gallery (Wrapped in HTML for cleaner styling) ---
image_urls = [
    "https://i5.walmartimages.com/seo/Apple-AirPods-with-Charging-Case-2nd-Generation_8540ab4f-8062-48d0-9133-323a99ed921d.fb43fa09a0faef3f9495feece1397f8d.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF",
    "https://i5.walmartimages.com/asr/b6247579-386a-4bda-99aa-01e44801bc33.49db04f5e5b8d7f329c6580455e2e010.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF",
    "https://i5.walmartimages.com/asr/0f803868-d25f-4891-b0c8-e27a514ede02.f22c42c1ea17cd4d2b30fdfc89a8797c.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF",
    "https://i5.walmartimages.com/asr/df1b081f-4fa9-4ea5-87f8-413b9cad7a6e.f580d742da0a58bc25dadd30512adf72.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF",
    "https://i5.walmartimages.com/asr/2830c8d7-292d-4b99-b92f-239b15ff1062.ce77d20b2f20a569bfd656d05ca89f7c.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF"
]

# Using st.columns for the images to ensure responsive layout
st.markdown('<div class="img-container">', unsafe_allow_html=True)
cols = st.columns(5)
for i, url in enumerate(image_urls):
    with cols[i]:
        st.image(url, width=110)
st.markdown('</div>', unsafe_allow_html=True)

# --- Input/Output Layout (2 Columns) ---
col_input, col_output = st.columns([1, 1])

with col_input:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 📝 Review Input")
    user_input = st.text_area("Type or paste the review here...", height=200, label_visibility="collapsed")
    analyze_btn = st.button("🔍 Analyze Sentiment", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if not user_input.strip() and analyze_btn:
        st.warning("Please enter text to analyze.")

with col_output:
    # Placeholder for results to maintain layout consistency
    result_container = st.container()
    
    if analyze_btn and user_input.strip() and model:
        with result_container:
            with st.spinner('Processing...'):
                time.sleep(0.5) # Simulate processing time
                
                # Perform Prediction
                probs = predict_sentiment(user_input)
                label, bg_color = get_sentiment_info(probs)
                confidence = np.max(probs) * 100

            # --- Result Display ---
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 📊 Analysis Result")
            
            # Center the main result
            st.markdown(
                f"""
                <div style="text-align: center; margin: 20px 0;">
                    <div class="result-badge" style="background-color: {bg_color};">
                        {label}
                    </div>
                    <h2 style="margin: 10px 0 0 0; color: #374151;">{confidence:.2f}%</h2>
                    <small style="color: #6b7280;">Confidence Score</small>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Detailed Breakdown (Interactive)
            with st.expander("See probability breakdown"):
                # Create progress bars for each sentiment
                labels_list = ["Negative", "Neutral", "Positive"]
                color_map = {"Negative": "#ef4444", "Neutral": "#f59e0b", "Positive": "#10b981"}
                
                for i, l_name in enumerate(labels_list):
                    p_score = probs[i] * 100
                    st.markdown(
                        f"""
                        <div style="display: flex; align-items: center; margin-bottom: 8px;">
                            <span style="width: 80px; font-size: 0.9em;">{l_name}</span>
                            <div style="flex-grow: 1; background: #e5e7eb; border-radius: 4px; height: 10px; margin: 0 10px; overflow: hidden;">
                                <div style="width: {p_score}%; background: {color_map[l_name]}; height: 100%;"></div>
                            </div>
                            <span style="width: 40px; font-size: 0.8em; text-align: right;">{p_score:.1f}%</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            st.markdown('</div>', unsafe_allow_html=True)

    elif analyze_btn and model is None:
         st.error("Model failed to load. Please check the repo ID.")
    else:
        # Empty State for the right column
        with result_container:
            st.markdown('<div class="card" style="text-align: center; color: #9ca3af; padding: 3rem 1rem;">', unsafe_allow_html=True)
            st.markdown("#### 🤖 Waiting for Input")
            st.markdown("Enter a review and click Analyze to see results here.")
            st.markdown('</div>', unsafe_allow_html=True)
