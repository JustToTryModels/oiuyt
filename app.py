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
# ENHANCED UI & CSS
# -----------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&family=Open+Sans:wght@400;600&family=Inter:wght@400;500;600;700&display=swap');

    /* ===== GLOBAL BACKGROUND & BODY ===== */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e9f2 50%, #f0e6f6 100%);
        font-family: 'Inter', 'Open Sans', sans-serif;
        color: #2d3748;
    }

    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ===== ANIMATED GRADIENT TITLE ===== */
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }

    @keyframes shimmer {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }

    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-50px); }
        to { opacity: 1; transform: translateX(0); }
    }

    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(50px); }
        to { opacity: 1; transform: translateX(0); }
    }

    @keyframes bounceIn {
        0% { opacity: 0; transform: scale(0.3); }
        50% { transform: scale(1.05); }
        70% { transform: scale(0.9); }
        100% { opacity: 1; transform: scale(1); }
    }

    @keyframes ripple {
        0% { box-shadow: 0 0 0 0 rgba(106, 5, 114, 0.3); }
        100% { box-shadow: 0 0 0 20px rgba(106, 5, 114, 0); }
    }

    @keyframes barGrow {
        from { width: 0%; }
        to { width: var(--bar-width); }
    }

    /* ===== HERO TITLE SECTION ===== */
    .hero-title-container {
        text-align: center;
        padding: 20px 0 10px 0;
        animation: fadeInDown 1s ease-out;
    }
    .hero-title {
        font-family: 'Nunito', sans-serif;
        font-weight: 900;
        font-size: 3.2em;
        background: linear-gradient(135deg, #6a0572, #e52e71, #ff8a00, #6a0572);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradientShift 4s ease infinite;
        margin-bottom: 0;
        letter-spacing: -1px;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        color: #718096;
        font-size: 1.1em;
        font-weight: 500;
        margin-top: 8px;
        animation: fadeIn 1.5s ease-out;
    }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6a0572, #9b59b6);
        color: white;
        padding: 4px 16px;
        border-radius: 20px;
        font-size: 0.75em;
        font-weight: 600;
        margin-top: 10px;
        letter-spacing: 1px;
        text-transform: uppercase;
        animation: fadeIn 2s ease-out;
    }

    /* ===== IMAGE CAROUSEL CARD ===== */
    .image-showcase {
        background: white;
        border-radius: 24px;
        padding: 24px 16px;
        margin: 20px 0;
        box-shadow: 0 10px 40px rgba(106, 5, 114, 0.08), 0 2px 10px rgba(0,0,0,0.04);
        animation: fadeInUp 0.8s ease-out;
        border: 1px solid rgba(106, 5, 114, 0.06);
        transition: box-shadow 0.3s ease, transform 0.3s ease;
    }
    .image-showcase:hover {
        box-shadow: 0 15px 50px rgba(106, 5, 114, 0.12), 0 5px 15px rgba(0,0,0,0.06);
        transform: translateY(-2px);
    }
    .image-showcase-title {
        text-align: center;
        font-family: 'Nunito', sans-serif;
        font-weight: 700;
        color: #6a0572;
        font-size: 1em;
        margin-bottom: 16px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .image-row {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 16px;
        flex-wrap: wrap;
    }
    .product-img-wrapper {
        background: linear-gradient(135deg, #f8f4ff, #fff);
        border-radius: 18px;
        padding: 10px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        cursor: pointer;
        border: 2px solid transparent;
        position: relative;
        overflow: hidden;
    }
    .product-img-wrapper::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: linear-gradient(135deg, rgba(106,5,114,0.05), rgba(229,46,113,0.05));
        border-radius: 16px;
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    .product-img-wrapper:hover::before {
        opacity: 1;
    }
    .product-img-wrapper:hover {
        transform: translateY(-8px) scale(1.08);
        border-color: rgba(106, 5, 114, 0.2);
        box-shadow: 0 12px 30px rgba(106, 5, 114, 0.15);
    }
    .product-img-wrapper img {
        border-radius: 12px;
        width: 90px;
        height: 90px;
        object-fit: contain;
    }

    /* ===== INPUT SECTION CARD ===== */
    .input-card {
        background: white;
        border-radius: 24px;
        padding: 32px 28px;
        margin: 20px 0;
        box-shadow: 0 10px 40px rgba(106, 5, 114, 0.08), 0 2px 10px rgba(0,0,0,0.04);
        animation: fadeInUp 1s ease-out;
        border: 1px solid rgba(106, 5, 114, 0.06);
        transition: box-shadow 0.3s ease;
    }
    .input-card:hover {
        box-shadow: 0 15px 50px rgba(106, 5, 114, 0.12), 0 5px 15px rgba(0,0,0,0.06);
    }
    .input-label {
        font-family: 'Nunito', sans-serif;
        font-weight: 700;
        color: #4a1a5e;
        font-size: 1.15em;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .input-hint {
        color: #a0aec0;
        font-size: 0.88em;
        margin-bottom: 14px;
        font-style: italic;
    }

    /* ===== TEXT AREA OVERRIDE ===== */
    .stTextArea textarea {
        border-radius: 16px !important;
        border: 2px solid #e2d9f3 !important;
        padding: 16px !important;
        background: linear-gradient(135deg, #fdfcff, #f9f7ff) !important;
        box-shadow: inset 0 2px 8px rgba(106, 5, 114, 0.04) !important;
        font-size: 1em !important;
        font-family: 'Inter', sans-serif !important;
        color: #2d3748 !important;
        transition: all 0.3s ease !important;
        line-height: 1.6 !important;
    }
    .stTextArea textarea:focus {
        border-color: #9b59b6 !important;
        box-shadow: 0 0 0 3px rgba(106, 5, 114, 0.1), inset 0 2px 8px rgba(106, 5, 114, 0.04) !important;
        background: #ffffff !important;
    }
    .stTextArea textarea::placeholder {
        color: #b794d0 !important;
        font-style: italic !important;
    }

    /* Hide default Streamlit label for text_area */
    .stTextArea label {
        display: none !important;
    }

    /* ===== ANALYZE BUTTON ===== */
    .stButton>button {
        background: linear-gradient(135deg, #6a0572, #e52e71, #ff8a00) !important;
        background-size: 200% 200% !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 14px 32px !important;
        font-size: 1.2em !important;
        font-weight: 700 !important;
        font-family: 'Nunito', sans-serif !important;
        cursor: pointer !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        width: 100% !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 8px 25px rgba(229, 46, 113, 0.3) !important;
        text-transform: none !important;
        animation: gradientShift 3s ease infinite !important;
    }
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 12px 35px rgba(229, 46, 113, 0.45) !important;
        color: white !important;
    }
    .stButton>button:active {
        transform: translateY(0px) scale(0.98) !important;
    }

    /* ===== RESULT CARD ===== */
    .result-card {
        background: white;
        border-radius: 28px;
        padding: 0;
        margin: 24px 0;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.08);
        overflow: hidden;
        animation: bounceIn 0.8s ease-out;
        border: 1px solid rgba(0,0,0,0.04);
    }
    .result-header {
        padding: 28px 32px 20px 32px;
        text-align: center;
    }
    .result-emoji {
        font-size: 4em;
        margin-bottom: 8px;
        animation: float 3s ease-in-out infinite;
        display: inline-block;
    }
    .result-sentiment {
        font-family: 'Nunito', sans-serif;
        font-weight: 900;
        font-size: 2em;
        margin: 8px 0 4px 0;
        letter-spacing: -0.5px;
    }
    .result-confidence-label {
        color: #718096;
        font-size: 0.95em;
        font-weight: 500;
        margin-bottom: 4px;
    }
    .result-confidence-value {
        font-family: 'Nunito', sans-serif;
        font-weight: 800;
        font-size: 2.2em;
        margin: 4px 0;
    }

    /* ===== PROBABILITY BARS ===== */
    .prob-section {
        background: #f9fafb;
        border-top: 1px solid #f0f0f0;
        padding: 24px 32px 28px 32px;
    }
    .prob-section-title {
        font-family: 'Nunito', sans-serif;
        font-weight: 700;
        color: #4a5568;
        font-size: 0.95em;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 18px;
        text-align: center;
    }
    .prob-row {
        display: flex;
        align-items: center;
        margin-bottom: 14px;
        gap: 12px;
    }
    .prob-label {
        font-weight: 600;
        font-size: 0.92em;
        min-width: 110px;
        color: #4a5568;
    }
    .prob-bar-container {
        flex: 1;
        background: #edf2f7;
        border-radius: 12px;
        height: 28px;
        overflow: hidden;
        position: relative;
    }
    .prob-bar {
        height: 100%;
        border-radius: 12px;
        transition: width 1.5s cubic-bezier(0.4, 0, 0.2, 1);
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 10px;
        position: relative;
        overflow: hidden;
    }
    .prob-bar::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        background-size: 200% 100%;
        animation: shimmer 2s ease-in-out infinite;
    }
    .prob-bar-negative {
        background: linear-gradient(135deg, #fc5c7d, #e74c3c);
    }
    .prob-bar-neutral {
        background: linear-gradient(135deg, #f9d423, #f0a500);
    }
    .prob-bar-positive {
        background: linear-gradient(135deg, #38ef7d, #11998e);
    }
    .prob-value {
        font-weight: 700;
        font-size: 0.82em;
        color: white;
        text-shadow: 0 1px 2px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
    }

    /* ===== DIVIDER STYLE ===== */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #d6bcfa, transparent);
        margin: 28px 0;
    }

    /* ===== FOOTER ===== */
    .app-footer {
        text-align: center;
        padding: 20px 0 10px 0;
        color: #a0aec0;
        font-size: 0.85em;
        animation: fadeIn 2s ease-out;
    }
    .app-footer a {
        color: #9b59b6;
        text-decoration: none;
        font-weight: 600;
    }

    /* ===== SAMPLE REVIEWS ===== */
    .sample-title {
        font-family: 'Nunito', sans-serif;
        font-weight: 700;
        color: #6a0572;
        font-size: 1em;
        margin-bottom: 10px;
        text-align: center;
    }
    .sample-chip {
        display: inline-block;
        background: linear-gradient(135deg, #f8f4ff, #f0e6f6);
        border: 1px solid #e2d9f3;
        border-radius: 20px;
        padding: 8px 16px;
        margin: 4px;
        font-size: 0.85em;
        color: #6a0572;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .sample-chip:hover {
        background: linear-gradient(135deg, #6a0572, #9b59b6);
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(106, 5, 114, 0.2);
    }

    /* ===== FEATURE PILLS ===== */
    .feature-pills {
        display: flex;
        justify-content: center;
        gap: 12px;
        flex-wrap: wrap;
        margin: 16px 0;
        animation: fadeIn 1.8s ease-out;
    }
    .feature-pill {
        background: rgba(106, 5, 114, 0.06);
        color: #6a0572;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.82em;
        font-weight: 600;
        border: 1px solid rgba(106, 5, 114, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Hero Title ---
st.markdown(
    """
    <div class="hero-title-container">
        <div class="hero-title">🎧 Apple AirPods</div>
        <div class="hero-title" style="font-size: 2em; margin-top: -5px;">Sentiment Analysis</div>
        <div class="hero-subtitle">Powered by ALBERT — Understand the emotion behind every review</div>
        <div class="hero-badge">🤖 AI-Powered NLP</div>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Feature Pills ---
st.markdown(
    """
    <div class="feature-pills">
        <span class="feature-pill">⚡ Real-time Analysis</span>
        <span class="feature-pill">🎯 3-Class Classification</span>
        <span class="feature-pill">📊 Confidence Scores</span>
    </div>
    """,
    unsafe_allow_html=True
)

# --- AirPods Image Showcase ---
image_urls = [
    "https://i5.walmartimages.com/seo/Apple-AirPods-with-Charging-Case-2nd-Generation_8540ab4f-8062-48d0-9133-323a99ed921d.fb43fa09a0faef3f9495feece1397f8d.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF",
    "https://i5.walmartimages.com/asr/b6247579-386a-4bda-99aa-01e44801bc33.49db04f5e5b8d7f329c6580455e2e010.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF",
    "https://i5.walmartimages.com/asr/0f803868-d25f-4891-b0c8-e27a514ede02.f22c42c1ea17cd4d2b30fdfc89a8797c.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF",
    "https://i5.walmartimages.com/asr/df1b081f-4fa9-4ea5-87f8-413b9cad7a6e.f580d742da0a58bc25dadd30512adf72.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF",
    "https://i5.walmartimages.com/asr/2830c8d7-292d-4b99-b92f-239b15ff1062.ce77d20b2f20a569bfd656d05ca89f7c.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF"
]

img_html = ""
for url in image_urls:
    img_html += f'<div class="product-img-wrapper"><img src="{url}" /></div>'

st.markdown(
    f"""
    <div class="image-showcase">
        <div class="image-showcase-title">📦 Featured AirPods Products</div>
        <div class="image-row">{img_html}</div>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Input Section Card ---
st.markdown(
    """
    <div class="input-card">
        <div class="input-label">✍️ Write or paste your review</div>
        <div class="input-hint">Share your experience with Apple AirPods — we'll detect the sentiment instantly</div>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Sample reviews as clickable chips ---
sample_reviews = {
    "👍 Great sound quality and seamless pairing!": "Great sound quality and seamless pairing with my iPhone!",
    "👎 Battery dies too quickly": "The battery life is terrible, dies within 2 hours of use.",
    "😐 They're okay, nothing special": "They're okay for the price, nothing special but they work fine.",
    "❤️ Best purchase ever!": "Absolutely love these AirPods! Best purchase I've ever made, the noise cancellation is incredible!",
    "💔 Fell apart after a month": "Terrible quality, one earbud stopped working after just a month. Complete waste of money."
}

st.markdown('<div style="text-align:center; margin-bottom: 10px;"><span class="sample-title">💡 Try a sample review:</span></div>', unsafe_allow_html=True)

sample_cols = st.columns(len(sample_reviews))
selected_sample = None
for i, (chip_label, review_text) in enumerate(sample_reviews.items()):
    with sample_cols[i]:
        if st.button(chip_label, key=f"sample_{i}", use_container_width=True):
            selected_sample = review_text

# Use session state to manage sample selection
if selected_sample:
    st.session_state["review_text"] = selected_sample

default_text = st.session_state.get("review_text", "")

user_input = st.text_area(
    "Enter your AirPods review here",
    value=default_text,
    height=150,
    placeholder="e.g., 'The sound quality is amazing and they fit perfectly in my ears...'"
)

st.write("")

# --- Analyze Button ---
if st.button("🔍 Analyze Sentiment"):
    if not user_input.strip():
        st.error("⚠️ Please enter a review to analyze.")
    elif model is None:
        st.error("Model failed to load. Please check the repo ID.")
    else:
        # Animated progress
        progress_bar = st.progress(0)
        status_text = st.empty()

        steps = ["🔄 Tokenizing input...", "🧠 Running ALBERT model...", "📊 Computing probabilities...", "✅ Done!"]
        for i, step in enumerate(steps):
            status_text.markdown(f"<p style='text-align:center; color:#6a0572; font-weight:600;'>{step}</p>", unsafe_allow_html=True)
            progress_bar.progress((i + 1) * 25)
            time.sleep(0.3)

        probs = predict_sentiment(user_input)
        label, bg_color = get_sentiment_info(probs)
        confidence = np.max(probs) * 100

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        # Determine result styling
        max_index = np.argmax(probs)
        emojis = ["😡", "😐", "😊"]
        sentiment_names = ["Negative", "Neutral", "Positive"]
        sentiment_colors = ["#e74c3c", "#f0a500", "#11998e"]
        result_bg_gradients = [
            "linear-gradient(135deg, #fff5f5, #fed7d7)",
            "linear-gradient(135deg, #fffff0, #fefcbf)",
            "linear-gradient(135deg, #f0fff4, #c6f6d5)"
        ]

        emoji = emojis[max_index]
        sentiment_name = sentiment_names[max_index]
        sentiment_color = sentiment_colors[max_index]
        result_bg = result_bg_gradients[max_index]

        # Build probability bars
        prob_bars_html = ""
        bar_classes = ["prob-bar-negative", "prob-bar-neutral", "prob-bar-positive"]
        bar_labels = ["😡 Negative", "😐 Neutral", "😊 Positive"]

        for i in range(3):
            pct = probs[i] * 100
            prob_bars_html += f"""
            <div class="prob-row">
                <span class="prob-label">{bar_labels[i]}</span>
                <div class="prob-bar-container">
                    <div class="prob-bar {bar_classes[i]}" style="width: {pct:.1f}%;">
                        <span class="prob-value">{pct:.1f}%</span>
                    </div>
                </div>
            </div>
            """

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-header" style="background: {result_bg};">
                    <div class="result-emoji">{emoji}</div>
                    <div class="result-sentiment" style="color: {sentiment_color};">{sentiment_name}</div>
                    <div class="result-confidence-label">Confidence Score</div>
                    <div class="result-confidence-value" style="color: {sentiment_color};">{confidence:.1f}%</div>
                </div>
                <div class="prob-section">
                    <div class="prob-section-title">📊 Probability Distribution</div>
                    {prob_bars_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Clear the stored sample after analysis
        if "review_text" in st.session_state:
            del st.session_state["review_text"]

# --- Footer ---
st.markdown(
    """
    <div class="app-footer">
        <p>Built with ❤️ using <a href="https://streamlit.io" target="_blank">Streamlit</a> & 
        <a href="https://huggingface.co" target="_blank">🤗 Hugging Face Transformers</a></p>
        <p style="font-size: 0.8em; margin-top: 4px;">Model: ALBERT-base-v2 fine-tuned on Apple AirPods reviews</p>
    </div>
    """,
    unsafe_allow_html=True
)
