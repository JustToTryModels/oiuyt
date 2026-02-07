import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time  # For adding a loading spinner

# --- Set page config MUST be the first Streamlit command ---
st.set_page_config(
    page_title="AirPods Review Sentiment Analyzer",
    page_icon=":headphones:",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- Hugging Face Model Repository ---
MODEL_NAME = "IamPradeep/Apple-Airpods-Sentiment-Analysis-ALBERT-base-v2"

# --- Cache the model loading to avoid reloading on every interaction ---
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

# --- Load tokenizer and model from Hugging Face Hub ---
try:
    with st.spinner('Loading model from Hugging Face Hub...'):
        tokenizer, model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- Function to predict sentiment ---
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return probs.numpy()

# --- Function to map probabilities to sentiment labels and emojis ---
def get_sentiment_label(probs):
    sentiment_mapping = ["Negative 😡", "Neutral 😐", "Positive 😊"]
    max_index = probs.argmax()
    return sentiment_mapping[max_index]

# --- Function to get background color based on sentiment ---
def get_background_color(label):
    if "Positive" in label:
        return "#C3E6CB"
    elif "Neutral" in label:
        return "#FFE8A1"
    else:
        return "#F5C6CB"

# --- Custom CSS for a more attractive look ---
st.markdown(
    """
    <style>
    /* Import Google Fonts - Keeping Nunito and Open Sans for general text */
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
    .stButton>button {
        background: linear-gradient(90deg, #ff8a00, #e52e71);
        color: white !important;
        border: none;
        border-radius: 25px;
        padding: 10px 20px;
        font-size: 1.2em;
        font-weight: bold;
        cursor: pointer;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stButton>button:hover {
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
        border-radius: 15px;
        border: 1px solid #ced4da;
        padding: 10px;
        background-color: #FFFFFF;
        box-shadow: 3px 3px 5px #9E9E9E;
    }
    .stTextArea textarea::placeholder {
        color: #999;
        font-style: italic;
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

# --- User Input Text Area ---
user_input = st.text_area("Enter your AirPods review here")

# --- Analyze Sentiment Button ---
if st.button("🔍 Analyze Sentiment"):
    if user_input:
        with st.spinner('Analyzing sentiment...'):
            time.sleep(0.5)
            sentiment_probs = predict_sentiment(user_input)
            sentiment_label = get_sentiment_label(sentiment_probs[0])
            background_color = get_background_color(sentiment_label)

        st.divider()
        st.markdown(
            f"""
            <div style="background-color:{background_color}; padding: 10px; border-radius: 25px; text-align: center;" class="prediction-box">
                <h3><span style="font-weight: bold;">Sentiment</span>: {sentiment_label}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.error("⚠️ Please enter a review to analyze.")
