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
# 2. LOAD MODEL & TOKENIZER (Cached for performance)
# -----------------------------------------------------------------------------
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
# 3. PREDICTION FUNCTION
# -----------------------------------------------------------------------------
def predict_sentiment(text):
    # Preprocess text
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=512
    )
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Apply softmax to get probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    probs_np = probs.detach().numpy()[0]
    
    # Get the predicted class index
    pred_idx = np.argmax(probs_np)
    
    return pred_idx, probs_np

# -----------------------------------------------------------------------------
# 4. USER INTERFACE
# -----------------------------------------------------------------------------
st.title("🎧 AirPods Sentiment Analysis")
st.markdown("""
This app analyzes reviews for Walmart AirPods using a fine-tuned **ALBERT** model.
Enter a review below to see if it is **Positive**, **Neutral**, or **Negative**.
""")

# Input Text Area
user_input = st.text_area("Enter Review Text:", height=150, placeholder="e.g., The sound quality is amazing, but the battery life could be better.")

if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    elif model is None:
        st.error("Model failed to load. Please check your Hugging Face repo ID.")
    else:
        with st.spinner("Analyzing..."):
            try:
                # Run prediction
                pred_idx, probabilities = predict_sentiment(user_input)
                sentiment_label = LABEL_MAP[pred_idx]
                confidence = probabilities[pred_idx] * 100

                # Display Result
                st.divider()
                st.subheader("Results")

                # Dynamic Color Coding
                if sentiment_label == "Positive":
                    color = "green"
                    emoji = "😃"
                elif sentiment_label == "Negative":
                    color = "red"
                    emoji = "😡"
                else:
                    color = "orange"
                    emoji = "😐"

                st.markdown(f"### Sentiment: :{color}[{sentiment_label} {emoji}]")
                st.markdown(f"**Confidence Score:** {confidence:.2f}%")

                # Visualization: Bar Chart of Probabilities
                st.markdown("#### Probability Distribution")
                
                chart_data = pd.DataFrame({
                    "Sentiment": ["Negative", "Neutral", "Positive"],
                    "Probability": probabilities
                })
                
                # Create a simple bar chart
                st.bar_chart(
                    chart_data, 
                    x="Sentiment", 
                    y="Probability", 
                    color=["#FF4B4B", "#FFA500", "#4CAF50"] # Red, Orange, Green hex codes roughly
                )

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

# Footer
st.markdown("---")
st.caption("Model trained on Walmart AirPods Reviews | Powered by Hugging Face & Streamlit")
