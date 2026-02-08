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
# MODEL LOADING (UNCHANGED CORE LOGIC)
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

# Load the model
tokenizer, model = load_model()

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS (UNCHANGED CORE LOGIC)
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
# INITIALIZE SESSION STATE FOR HISTORY
# -----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------------------------------------------------------
# GLOBAL CSS — MODERN, POLISHED THEME
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&family=Open+Sans:wght@400;500;600&family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Root variables for easy theming ── */
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --accent-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --button-gradient: linear-gradient(135deg, #ff6a00 0%, #ee0979 100%);
        --card-bg: rgba(255, 255, 255, 0.95);
        --card-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
        --card-radius: 20px;
        --text-primary: #1a1a2e;
        --text-secondary: #555;
        --bg-main: #f0f2f6;
    }

    /* ── Main background ── */
    .main {
        background: linear-gradient(160deg, #f0f2f6 0%, #e8eaf6 50%, #f3e5f5 100%);
        font-family: 'Inter', 'Open Sans', sans-serif;
        color: var(--text-primary);
    }

    /* ── Hide default Streamlit header/footer for cleaner look ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Animated gradient hero banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-size: 200% 200%;
        animation: gradientShift 6s ease infinite;
        border-radius: 24px;
        padding: 40px 30px 30px 30px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 60%);
        animation: shimmer 4s ease-in-out infinite;
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes shimmer {
        0%, 100% { transform: translateX(-30%) translateY(-30%); }
        50% { transform: translateX(10%) translateY(10%); }
    }
    .hero-banner h1 {
        font-family: 'Nunito', sans-serif;
        color: #ffffff !important;
        font-size: 2.6em;
        font-weight: 900;
        margin: 0 0 8px 0;
        text-shadow: 2px 4px 12px rgba(0, 0, 0, 0.25);
        position: relative;
        z-index: 1;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1em;
        font-weight: 400;
        position: relative;
        z-index: 1;
        margin-top: 0;
    }

    /* ── Glass-morphism card style ── */
    .glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(10px);
        border-radius: var(--card-radius);
        padding: 28px 28px 22px 28px;
        box-shadow: var(--card-shadow);
        border: 1px solid rgba(255, 255, 255, 0.6);
        margin-bottom: 24px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.12);
    }
    .card-header {
        font-family: 'Nunito', sans-serif;
        font-size: 1.3em;
        font-weight: 800;
        color: var(--text-primary);
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .card-header-icon {
        font-size: 1.4em;
    }

    /* ── Styled button ── */
    .stButton>button {
        background: var(--button-gradient) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 14px 32px !important;
        font-size: 1.15em !important;
        font-weight: 700 !important;
        font-family: 'Nunito', sans-serif !important;
        cursor: pointer !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
        box-shadow: 0 6px 20px rgba(238, 9, 121, 0.3) !important;
        letter-spacing: 0.5px !important;
    }
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 10px 30px rgba(238, 9, 121, 0.4) !important;
        color: white !important;
    }
    .stButton>button:active {
        transform: translateY(0px) scale(0.98) !important;
    }

    /* ── Text area styling ── */
    .stTextArea textarea {
        border-radius: 16px !important;
        border: 2px solid #e0e0e0 !important;
        padding: 16px !important;
        background-color: #FAFBFC !important;
        box-shadow: inset 0 2px 6px rgba(0,0,0,0.04) !important;
        font-size: 15px !important;
        font-family: 'Inter', sans-serif !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
        line-height: 1.6 !important;
    }
    .stTextArea textarea:focus {
        border-color: #764ba2 !important;
        box-shadow: 0 0 0 3px rgba(118, 75, 162, 0.15), inset 0 2px 6px rgba(0,0,0,0.04) !important;
    }
    .stTextArea textarea::placeholder {
        color: #aaa !important;
        font-style: italic !important;
    }

    /* ── Prediction Box (Imported from Code-2) ── */
    .prediction-box {
        border-radius: 25px;
        padding: 10px;
        text-align: center;
        font-size: 18px;
    }

    /* ── Probability bars ── */
    .prob-bar-container {
        margin: 8px 0;
        animation: fadeSlideUp 0.7s ease-out;
    }
    .prob-label-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.92em;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 5px;
    }
    .prob-bar-bg {
        background: #eee;
        border-radius: 10px;
        height: 14px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* ── Sidebar styling ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] .stMarkdown span {
        color: rgba(255, 255, 255, 0.85) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.15) !important;
    }

    /* ── Image row styling ── */
    .airpod-img-wrapper {
        background: white;
        border-radius: 16px;
        padding: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .airpod-img-wrapper:hover {
        transform: translateY(-5px) scale(1.05);
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    }

    /* ── Expander styling ── */
    .streamlit-expanderHeader {
        font-family: 'Nunito', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.05em !important;
        color: var(--text-primary) !important;
    }

    /* ── Example chip buttons ── */
    .example-chip {
        display: inline-block;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 1px solid #dee2e6;
        border-radius: 50px;
        padding: 8px 18px;
        margin: 4px;
        font-size: 0.85em;
        color: var(--text-primary);
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .example-chip:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: transparent;
        transform: translateY(-1px);
    }

    /* ── Stat metric cards ── */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 16px;
        padding: 16px;
        text-align: center;
        border: 1px solid rgba(0,0,0,0.06);
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .metric-value {
        font-family: 'Nunito', sans-serif;
        font-size: 1.6em;
        font-weight: 900;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-label {
        font-size: 0.82em;
        color: var(--text-secondary);
        margin-top: 4px;
        font-weight: 500;
    }

    /* ── Toast / info strip ── */
    .info-strip {
        background: linear-gradient(90deg, #667eea20, #764ba220);
        border-left: 4px solid #764ba2;
        border-radius: 0 12px 12px 0;
        padding: 12px 18px;
        margin: 12px 0;
        font-size: 0.92em;
        color: var(--text-primary);
    }

    /* ── History item ── */
    .history-item {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 12px 16px;
        margin: 6px 0;
        border-left: 4px solid;
        font-size: 0.9em;
    }

    /* ── Scrollbar styling ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #764ba2; border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# SIDEBAR — App info, navigation, and settings
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 10px 0 20px 0;">
            <span style="font-size: 3.5em;">🎧</span>
            <h2 style="margin: 8px 0 0 0; font-family: 'Nunito', sans-serif; font-weight: 900;
                        background: linear-gradient(135deg, #667eea, #f093fb);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                        background-clip: text;">
                AirPods Analyzer
            </h2>
            <p style="font-size: 0.85em; opacity: 0.7; margin-top: 4px;">
                Powered by ALBERT-base-v2
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # --- About section ---
    st.markdown("### ℹ️ About")
    st.markdown(
        """
        This app uses a **fine-tuned ALBERT** model to classify
        Apple AirPods reviews into **Positive**, **Neutral**, or
        **Negative** sentiments in real-time.
        """
    )

    st.divider()

    # --- How to use section ---
    st.markdown("### 🚀 How to Use")
    st.markdown(
        """
        1. ✍️ Type or paste a review  
        2. 🔍 Click **Analyze Sentiment**  
        3. 📊 View results & probabilities  
        """
    )

    st.divider()

    # --- Model details expander ---
    with st.expander("🧠 Model Details"):
        st.markdown(
            f"""
            | Property | Value |
            |----------|-------|
            | **Architecture** | ALBERT-base-v2 |
            | **Task** | Sequence Classification |
            | **Classes** | 3 (Neg / Neu / Pos) |
            | **Max Length** | 512 tokens |
            | **Source** | [HuggingFace Hub]({f'https://huggingface.co/{MODEL_REPO}'}) |
            """
        )

    st.divider()

    # --- Show analysis history toggle ---
    show_history = st.toggle("📜 Show Analysis History", value=False,
                             help="Toggle to display past analyses from this session")

    st.divider()

    # --- Footer ---
    st.markdown(
        """
        <div style="text-align: center; padding-top: 10px; opacity: 0.6; font-size: 0.78em;">
            Built with ❤️ using<br>
            <strong>Streamlit</strong> · <strong>HuggingFace</strong> · <strong>PyTorch</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# MAIN CONTENT — Hero Banner
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <h1>🎧 Apple AirPods Sentiment Analysis</h1>
        <p class="hero-subtitle">Instantly understand the sentiment behind any AirPods review using AI</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- AirPods Image Row inside a glass card ---
image_urls = [
    "https://i5.walmartimages.com/seo/Apple-AirPods-with-Charging-Case-2nd-Generation_8540ab4f-8062-48d0-9133-323a99ed921d.fb43fa09a0faef3f9495feece1397f8d.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF",
    "https://i5.walmartimages.com/asr/b6247579-386a-4bda-99aa-01e44801bc33.49db04f5e5b8d7f329c6580455e2e010.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF",
    "https://i5.walmartimages.com/asr/0f803868-d25f-4891-b0c8-e27a514ede02.f22c42c1ea17cd4d2b30fdfc89a8797c.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF",
    "https://i5.walmartimages.com/asr/df1b081f-4fa9-4ea5-87f8-413b9cad7a6e.f580d742da0a58bc25dadd30512adf72.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF",
    "https://i5.walmartimages.com/asr/2830c8d7-292d-4b99-b92f-239b15ff1062.ce77d20b2f20a569bfd656d05ca89f7c.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF",
]

cols = st.columns(5)
for i, url in enumerate(image_urls):
    with cols[i]:
        st.markdown('<div class="airpod-img-wrapper">', unsafe_allow_html=True)
        st.image(url, width=100)
        st.markdown("</div>", unsafe_allow_html=True)

st.write("")  # Spacer

# -----------------------------------------------------------------------------
# INPUT SECTION — Review entry
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="card-header">
        <span class="card-header-icon">✍️</span> Enter Your Review
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="info-strip">💡 <strong>Tip:</strong> Write a detailed review for more accurate sentiment predictions. You can also try the example reviews below!</div>',
    unsafe_allow_html=True,
)

# --- Example review selector ---
example_reviews = {
    "👍 Positive Example": "These AirPods are absolutely incredible! The sound quality is crystal clear, the noise cancellation is top-notch, and they fit perfectly in my ears. Best purchase I've made this year!",
    "😐 Neutral Example": "The AirPods work fine for the price. Sound quality is decent and battery life is okay. Nothing special but they get the job done for everyday use.",
    "👎 Negative Example": "Very disappointed with these AirPods. They keep disconnecting from my phone, the sound quality is mediocre at best, and they fell out of my ears constantly. Not worth the money.",
}

with st.expander("📝 Try an Example Review", expanded=False):
    selected_example = st.radio(
        "Choose an example:",
        options=list(example_reviews.keys()),
        horizontal=True,
        label_visibility="collapsed",
    )
    if st.button("📋 Load Example", key="load_example"):
        st.session_state["review_text"] = example_reviews[selected_example]

# --- User Input Text Area (core functionality preserved) ---
user_input = st.text_area(
    "Enter your AirPods review here",
    height=150,
    value=st.session_state.get("review_text", ""),
    placeholder="e.g., The AirPods Pro 2 have amazing noise cancellation and the sound quality is superb...",
    label_visibility="collapsed",
)

# --- Character counter ---
char_count = len(user_input.strip())
word_count = len(user_input.strip().split()) if user_input.strip() else 0
col_c1, col_c2, col_c3 = st.columns(3)
with col_c1:
    st.caption(f"📝 **{char_count}** characters")
with col_c2:
    st.caption(f"📖 **{word_count}** words")
with col_c3:
    if char_count > 0:
        st.caption("✅ Ready to analyze")
    else:
        st.caption("⏳ Waiting for input")

st.write("")  # Spacer

# -----------------------------------------------------------------------------
# ANALYZE BUTTON & RESULTS (CORE LOGIC UNCHANGED)
# -----------------------------------------------------------------------------
if st.button("🔍 Analyze Sentiment"):
    if not user_input.strip():
        st.error("⚠️ Please enter a review to analyze.")
    elif model is None:
        st.error("Model failed to load. Please check the repo ID.")
    else:
        with st.spinner("🧠 Analyzing sentiment..."):
            time.sleep(0.5)  # Simulate processing time

            # Predict (CORE LOGIC — UNCHANGED)
            probs = predict_sentiment(user_input)
            label, bg_color = get_sentiment_info(probs)
            confidence = np.max(probs) * 100

        # --- Save to history ---
        st.session_state.history.insert(
            0,
            {
                "text": user_input[:80] + ("..." if len(user_input) > 80 else ""),
                "label": label,
                "confidence": confidence,
                "color": bg_color,
            },
        )

        st.divider()

        # --- Results Section Header ---
        st.markdown(
            """
            <div class="card-header" style="justify-content: center; margin-top: 8px;">
                <span class="card-header-icon">📊</span> Analysis Results
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --- Main Result Card (REPLACED WITH CODE-2 STRUCTURE) ---
        st.markdown(
            f"""
            <div style="background-color:{bg_color}; padding: 10px; border-radius: 25px; text-align: center;" class="prediction-box">
                <h3><span style="font-weight: bold;">Sentiment</span>: {label}</h3>
                <p style="margin-top: 5px; font-size: 16px;">(Confidence: {confidence:.2f}%)</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")  # Spacer

        # --- Detailed Probability Breakdown ---
        with st.expander("📈 Detailed Probability Breakdown", expanded=True):
            prob_labels = ["Negative 😡", "Neutral 😐", "Positive 😊"]
            prob_colors = ["#e74c3c", "#f39c12", "#27ae60"]

            for idx, (p_label, p_color, p_val) in enumerate(
                zip(prob_labels, prob_colors, probs)
            ):
                pct = p_val * 100
                st.markdown(
                    f"""
                    <div class="prob-bar-container">
                        <div class="prob-label-row">
                            <span>{p_label}</span>
                            <span>{pct:.1f}%</span>
                        </div>
                        <div class="prob-bar-bg">
                            <div class="prob-bar-fill" style="width: {pct}%; background: {p_color};"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # --- Quick Stats Row ---
        st.write("")
        stat1, stat2, stat3 = st.columns(3)
        with stat1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{confidence:.1f}%</div>
                    <div class="metric-label">Confidence Score</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with stat2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{word_count}</div>
                    <div class="metric-label">Words Analyzed</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with stat3:
            dominant = label.split()[0]
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{dominant}</div>
                    <div class="metric-label">Dominant Sentiment</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.balloons()

# -----------------------------------------------------------------------------
# ANALYSIS HISTORY — Shown via sidebar toggle
# -----------------------------------------------------------------------------
if show_history and st.session_state.history:
    st.write("")
    st.divider()
    st.markdown(
        """
        <div class="card-header">
            <span class="card-header-icon">📜</span> Analysis History
        </div>
        """,
        unsafe_allow_html=True,
    )

    for idx, item in enumerate(st.session_state.history[:10]):  # Show last 10
        border_color = item["color"]
        st.markdown(
            f"""
            <div class="history-item" style="border-left-color: {border_color};">
                <strong>{item['label']}</strong> — {item['confidence']:.1f}% confidence<br>
                <span style="color: #777; font-size: 0.85em;">"{item['text']}"</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.button("🗑️ Clear History", key="clear_history"):
        st.session_state.history = []
        st.rerun()

elif show_history and not st.session_state.history:
    st.info("📭 No analyses yet. Enter a review and click **Analyze Sentiment** to get started!")

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.write("")
st.write("")
st.markdown(
    """
    <div style="text-align: center; padding: 30px 0 10px 0; opacity: 0.5; font-size: 0.82em;">
        🎧 AirPods Sentiment Analyzer &nbsp;·&nbsp; Built with
        <a href="https://streamlit.io" target="_blank" style="color: #764ba2; text-decoration: none;">Streamlit</a> &
        <a href="https://huggingface.co" target="_blank" style="color: #764ba2; text-decoration: none;">HuggingFace</a>
    </div>
    """,
    unsafe_allow_html=True,
)
