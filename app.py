import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px
import torch

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AirPods Sentiment Analyzer",
    page_icon="🎧",
    layout="centered"
)

# -----------------------------------------------------------------------------
# 2. LOAD MODEL FUNCTION (Cached for Performance)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_sentiment_pipeline():
    """
    Loads the fine-tuned ALBERT model from Hugging Face.
    Repository: IamPradeep/Apple-Airpods-Sentiment-Analysis-ALBERT-base-v2
    """
    model_id = "IamPradeep/Apple-Airpods-Sentiment-Analysis-ALBERT-base-v2"
    
    # Check if CUDA (GPU) is available
    device = 0 if torch.cuda.is_available() else -1
    
    # Load pipeline
    # top_k=None ensures we get scores for all labels (Negative, Neutral, Positive)
    classifier = pipeline("text-classification", model=model_id, device=device, top_k=None)
    return classifier

# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def map_labels(prediction_list):
    """
    Maps model outputs (LABEL_0, LABEL_1, LABEL_2) to readable sentiments.
    Based on your training: 0=Negative, 1=Neutral, 2=Positive
    """
    # Define mapping based on your training code
    label_map = {
        "LABEL_0": "Negative",
        "LABEL_1": "Neutral",
        "LABEL_2": "Positive",
        # Adding lowercase variants just in case HF config differs
        "label_0": "Negative",
        "label_1": "Neutral",
        "label_2": "Positive",
        "Negative": "Negative",
        "Neutral": "Neutral",
        "Positive": "Positive"
    }
    
    formatted_data = {}
    for item in prediction_list:
        label_str = item['label']
        clean_label = label_map.get(label_str, label_str) # Fallback to original if not found
        formatted_data[clean_label] = item['score']
        
    return formatted_data

# -----------------------------------------------------------------------------
# 4. STREAMLIT UI LAYOUT
# -----------------------------------------------------------------------------

# Title and Subtitle
st.title("🎧 Apple AirPods Sentiment Analysis")
st.markdown("### Powered by Fine-Tuned ALBERT (albert-base-v2)")
st.markdown("Enter a customer review below to classify it as **Positive**, **Neutral**, or **Negative**.")

# Sidebar with info and examples
with st.sidebar:
    st.header("About the Model")
    st.info(
        """
        **Model Architecture:** ALBERT Base v2  
        **Source:** Hugging Face Hub  
        **Repo:** `IamPradeep/Apple-Airpods-Sentiment-Analysis-ALBERT-base-v2`  
        **Training Data:** Walmart AirPods Reviews  
        """
    )
    
    st.markdown("---")
    st.markdown("**Try these examples:**")
    example_reviews = [
        "The sound quality is amazing and they connect instantly!",
        "They are okay, but the battery life could be better.",
        "Terrible product. One side stopped working after a week."
    ]
    
    # Button to quickly fill example (Streamlit workaround using session state)
    if "review_text" not in st.session_state:
        st.session_state.review_text = ""

    for ex in example_reviews:
        if st.button(f"📝 {ex[:30]}..."):
            st.session_state.review_text = ex
            st.rerun()

# Main Input Area
user_input = st.text_area("Enter Review Text:", value=st.session_state.get("review_text", ""), height=100, placeholder="Type your review here...")

# Analyze Button
if st.button("Analyze Sentiment", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Loading model and analyzing..."):
            try:
                # Load Model
                classifier = load_sentiment_pipeline()
                
                # Get Prediction
                # The pipeline returns a list of lists because input is a single string
                raw_predictions = classifier(user_input)[0]
                
                # Process Results
                scores = map_labels(raw_predictions)
                
                # Determine the highest score
                best_label = max(scores, key=scores.get)
                confidence = scores[best_label]
                
                # ---------------------------------------------------------
                # DISPLAY RESULTS
                # ---------------------------------------------------------
                st.markdown("---")
                
                # Create columns for result display
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("Prediction")
                    
                    # Dynamic Coloring and Emoji
                    if best_label == "Positive":
                        st.success(f"**{best_label}** 😃")
                    elif best_label == "Negative":
                        st.error(f"**{best_label}** 😠")
                    else:
                        st.warning(f"**{best_label}** 😐")
                        
                    st.metric("Confidence Score", f"{confidence:.2%}")

                with col2:
                    st.subheader("Probability Distribution")
                    # Prepare data for Plotly
                    chart_data = pd.DataFrame({
                        "Sentiment": list(scores.keys()),
                        "Score": list(scores.values())
                    })
                    
                    # Custom colors
                    color_map = {"Negative": "#ff4b4b", "Neutral": "#ffa421", "Positive": "#21c354"}
                    
                    fig = px.bar(
                        chart_data, 
                        x="Score", 
                        y="Sentiment", 
                        orientation='h', 
                        color="Sentiment",
                        color_discrete_map=color_map,
                        text_auto='.2%',
                        range_x=[0, 1]
                    )
                    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0), height=200)
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error loading model or processing text: {str(e)}")
                st.markdown("Make sure the repository `IamPradeep/Apple-Airpods-Sentiment-Analysis-ALBERT-base-v2` is public on Hugging Face.")

# Footer
st.markdown("---")
st.caption("Developed with Streamlit and Transformers")
