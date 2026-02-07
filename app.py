import streamlit as st
import torch
import pandas as pd
import numpy as np
from transformers import AlbertTokenizerFast, AutoModelForSequenceClassification

# -----------------------------------------------------------------------------
# 1. APP CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AirPods Sentiment Analyzer",
    page_icon="🎧",
    layout="centered"
)

# Define the Hugging Face Model Repo
MODEL_REPO = "IamPradeep/Apple-Airpods-Sentiment-Analysis-ALBERT-base-v2"

# Map labels (Must match your training configuration: 0=Neg, 1=Neu, 2=Pos)
LABEL_MAP = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

# -----------------------------------------------------------------------------
# 2. LOAD MODEL & TOKENIZER
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        # Load tokenizer and model directly from Hugging Face Hub
        tokenizer = AlbertTokenizerFast.from_pretrained(MODEL_REPO)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
        return tokenizer, model
    except Exception as e:
        return None, None

# Load the model
tokenizer, model = load_model()

# -----------------------------------------------------------------------------
# 3. PREDICTION FUNCTION
# -----------------------------------------------------------------------------
def predict_sentiment(text):
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=512
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    probs_np = probs.detach().numpy()[0] # Returns array like [0.1, 0.8, 0.1]
    
    # Get the predicted class index
    pred_idx = np.argmax(probs_np)
    
    return pred_idx, probs_np

# -----------------------------------------------------------------------------
# 4. USER INTERFACE
# -----------------------------------------------------------------------------
st.title("🎧 AirPods Sentiment Analysis")
st.markdown("""
This app analyzes reviews using a fine-tuned **ALBERT** model.
""")

# Input Area
user_input = st.text_area("Enter Review Text:", height=150, placeholder="Type your review here...")

if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    elif model is None:
        st.error(f"Could not load model from {MODEL_REPO}. Check the repo ID.")
    else:
        with st.spinner("Analyzing..."):
            try:
                # Get Predictions
                pred_idx, probabilities = predict_sentiment(user_input)
                sentiment_label = LABEL_MAP[pred_idx]
                confidence = probabilities[pred_idx] * 100

                # Display Text Result
                st.divider()
                
                # Determine color for text
                if sentiment_label == "Positive":
                    st.success(f"**Sentiment: Positive** (Confidence: {confidence:.2f}%)")
                elif sentiment_label == "Negative":
                    st.error(f"**Sentiment: Negative** (Confidence: {confidence:.2f}%)")
                else:
                    st.warning(f"**Sentiment: Neutral** (Confidence: {confidence:.2f}%)")

                # ---------------------------------------------------------
                # VISUALIZATION FIX
                # ---------------------------------------------------------
                st.markdown("#### Confidence Levels")

                # Reshape data into a DataFrame with 1 row and 3 columns
                # Columns: Negative, Neutral, Positive
                chart_data = pd.DataFrame(
                    [probabilities], 
                    columns=["Negative", "Neutral", "Positive"]
                )

                # Now we have 3 columns, so we can provide a list of 3 colors
                # Red for Negative, Orange for Neutral, Green for Positive
                st.bar_chart(
                    chart_data, 
                    color=["#FF4B4B", "#FFA500", "#4CAF50"]
                )

            except Exception as e:
                st.error(f"An error occurred: {e}")

st.markdown("---")
st.caption("Model: ALBERT-base-v2 | Finetuned on Walmart AirPods Reviews")
