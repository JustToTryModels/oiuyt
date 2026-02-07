import streamlit as st
import torch
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from transformers import AlbertTokenizerFast, AutoModelForSequenceClassification
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AirPods Review Sentiment Analyzer",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e3d59;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #5eb3d6;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background-color: #1e3d59;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #5eb3d6;
        transform: scale(1.05);
    }
    .sentiment-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .positive {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .negative {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    .neutral {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# Cache the model loading
@st.cache_resource
def load_model():
    """Load the fine-tuned ALBERT model and tokenizer from Hugging Face"""
    with st.spinner("🔄 Loading ALBERT model from Hugging Face..."):
        model_name = "IamPradeep/Apple-Airpods-Sentiment-Analysis-ALBERT-base-v2"
        
        try:
            tokenizer = AlbertTokenizerFast.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Move model to appropriate device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            model.eval()
            
            return model, tokenizer, device
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None, None, None

def predict_sentiment(text, model, tokenizer, device):
    """Predict sentiment for a given text"""
    # Tokenize the input text
    encodings = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors='pt'
    )
    
    # Move tensors to device
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        prediction = torch.argmax(logits, dim=-1)
        confidence = torch.max(probabilities, dim=-1).values
    
    # Map predictions to labels
    label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    predicted_label = label_map[prediction.item()]
    confidence_score = confidence.item()
    
    # Get all class probabilities
    all_probs = probabilities.squeeze().cpu().numpy()
    class_probabilities = {
        'Negative': all_probs[0],
        'Neutral': all_probs[1],
        'Positive': all_probs[2]
    }
    
    return predicted_label, confidence_score, class_probabilities

def create_gauge_chart(confidence):
    """Create a gauge chart for confidence score"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence * 100,
        title = {'text': "Confidence Score"},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def create_probability_chart(class_probabilities):
    """Create a bar chart for class probabilities"""
    df = pd.DataFrame({
        'Sentiment': list(class_probabilities.keys()),
        'Probability': list(class_probabilities.values())
    })
    
    colors = {'Negative': '#ff6b6b', 'Neutral': '#ffd93d', 'Positive': '#6bcf63'}
    
    fig = px.bar(
        df, 
        x='Sentiment', 
        y='Probability',
        color='Sentiment',
        color_discrete_map=colors,
        title='Sentiment Class Probabilities',
        labels={'Probability': 'Probability (%)'},
    )
    
    fig.update_layout(
        showlegend=False,
        yaxis_tickformat='.0%',
        yaxis_range=[0, 1],
        height=400
    )
    
    fig.update_traces(
        texttemplate='%{y:.1%}',
        textposition='outside'
    )
    
    return fig

def display_sentiment_result(sentiment, confidence):
    """Display sentiment result with custom styling"""
    sentiment_emoji = {
        'Positive': '😊',
        'Negative': '😞',
        'Neutral': '😐'
    }
    
    sentiment_class = sentiment.lower()
    
    st.markdown(f"""
    <div class="sentiment-box {sentiment_class}">
        <h2 style="margin: 0;">Predicted Sentiment</h2>
        <h1 style="margin: 0.5rem 0; font-size: 3rem;">
            {sentiment_emoji[sentiment]} {sentiment}
        </h1>
        <p style="margin: 0; font-size: 1.2rem;">
            Confidence: {confidence*100:.2f}%
        </p>
    </div>
    """, unsafe_allow_html=True)

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🎧 Apple AirPods Review Sentiment Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by ALBERT-base-v2 Fine-tuned Model</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://store.storeimages.cdn-apple.com/4982/as-images.apple.com/is/airpods-3rd-gen-hero-202110?wid=940&hei=1112&fmt=png-alpha&.v=1633783104000", width=200)
        st.markdown("### About")
        st.markdown("""
        This application analyzes the sentiment of Apple AirPods reviews using a fine-tuned ALBERT model.
        
        **Model Details:**
        - Base Model: ALBERT-base-v2
        - Fine-tuned on: Walmart AirPods Reviews
        - Classes: Positive, Neutral, Negative
        - Accuracy: ~85%
        
        **Features:**
        - Real-time sentiment prediction
        - Confidence scores
        - Batch processing
        - Export results
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Model Performance")
        st.markdown("""
        - **Precision**: 0.85
        - **Recall**: 0.84
        - **F1-Score**: 0.84
        """)
        
        st.markdown("---")
        st.markdown("### 🔗 Links")
        st.markdown("[🤗 Model on HuggingFace](https://huggingface.co/IamPradeep/Apple-Airpods-Sentiment-Analysis-ALBERT-base-v2)")
        st.markdown("[📚 Documentation](https://github.com/yourusername/airpods-sentiment)")
    
    # Load model
    model, tokenizer, device = load_model()
    
    if model is None:
        st.error("Failed to load the model. Please check your internet connection and try again.")
        return
    
    # Success message
    st.success("✅ Model loaded successfully!")
    
    # Tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Single Review", "Batch Analysis", "Examples"])
    
    with tab1:
        st.markdown("### Analyze a Single Review")
        
        # Text input
        review_text = st.text_area(
            "Enter your AirPods review:",
            height=150,
            placeholder="Type or paste your review here... (e.g., 'The sound quality is amazing and the battery life is impressive!')"
        )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            analyze_button = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
        
        if analyze_button and review_text:
            with st.spinner("Analyzing sentiment..."):
                # Simulate processing time for better UX
                time.sleep(0.5)
                
                # Get prediction
                sentiment, confidence, class_probs = predict_sentiment(
                    review_text, model, tokenizer, device
                )
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    display_sentiment_result(sentiment, confidence)
                    
                with col2:
                    st.plotly_chart(create_gauge_chart(confidence), use_container_width=True)
                
                # Show probability distribution
                st.plotly_chart(create_probability_chart(class_probs), use_container_width=True)
                
                # Additional insights
                with st.expander("📝 Review Analysis Details"):
                    st.markdown("**Original Review:**")
                    st.write(review_text)
                    st.markdown("**Character Count:**")
                    st.write(len(review_text))
                    st.markdown("**Word Count:**")
                    st.write(len(review_text.split()))
                    st.markdown("**Timestamp:**")
                    st.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        elif analyze_button:
            st.warning("Please enter a review to analyze.")
    
    with tab2:
        st.markdown("### Batch Review Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload a CSV file with reviews",
            type=['csv'],
            help="CSV should have a column named 'review' or 'text'"
        )
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            # Try to find the review column
            review_column = None
            for col in df.columns:
                if col.lower() in ['review', 'text', 'reviews', 'review_text']:
                    review_column = col
                    break
            
            if review_column:
                st.success(f"Found {len(df)} reviews in column '{review_column}'")
                
                if st.button("🚀 Analyze All Reviews", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    sentiments = []
                    confidences = []
                    
                    for i, review in enumerate(df[review_column]):
                        status_text.text(f'Processing review {i+1}/{len(df)}...')
                        progress_bar.progress((i + 1) / len(df))
                        
                        sentiment, confidence, _ = predict_sentiment(
                            str(review), model, tokenizer, device
                        )
                        sentiments.append(sentiment)
                        confidences.append(confidence)
                    
                    df['Predicted_Sentiment'] = sentiments
                    df['Confidence'] = confidences
                    
                    status_text.text('Analysis complete!')
                    
                    # Display results
                    st.markdown("### Results")
                    st.dataframe(df)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Reviews", len(df))
                    with col2:
                        st.metric("Average Confidence", f"{np.mean(confidences)*100:.2f}%")
                    with col3:
                        st.metric("Most Common Sentiment", df['Predicted_Sentiment'].mode()[0])
                    
                    # Sentiment distribution
                    sentiment_counts = df['Predicted_Sentiment'].value_counts()
                    fig = px.pie(
                        values=sentiment_counts.values,
                        names=sentiment_counts.index,
                        title="Sentiment Distribution",
                        color_discrete_map={'Positive': '#6bcf63', 'Negative': '#ff6b6b', 'Neutral': '#ffd93d'}
                    )
                    st.plotly_chart(fig)
                    
                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"sentiment_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            else:
                st.error("Could not find a review column. Please ensure your CSV has a column named 'review' or 'text'.")
    
    with tab3:
        st.markdown("### Example Reviews")
        st.markdown("Click on any example to analyze it:")
        
        examples = {
            "Positive": [
                "I absolutely love my Apple AirPods! The sound quality is incredible, and the noise cancellation is perfect.",
                "These AirPods exceeded my expectations! The audio is sharp, and the connection never drops.",
                "Best wireless earbuds I've ever owned! Crystal clear sound and amazing battery life."
            ],
            "Neutral": [
                "The Apple AirPods are decent, but I expected a bit more for the price.",
                "They're okay wireless earbuds. Nothing special but they get the job done.",
                "Average product. Sound is fine, battery life is standard."
            ],
            "Negative": [
                "Very disappointed with my AirPods. They don't stay in my ears and the battery drains quickly.",
                "The AirPods are overrated. Poor sound quality for such an expensive product.",
                "Terrible experience. Connection drops constantly and they're uncomfortable."
            ]
        }
        
        for sentiment_type, reviews in examples.items():
            st.markdown(f"**{sentiment_type} Examples:**")
            for review in reviews:
                if st.button(f"📝 {review[:50]}...", key=f"example_{review[:20]}"):
                    with st.spinner("Analyzing..."):
                        sentiment, confidence, class_probs = predict_sentiment(
                            review, model, tokenizer, device
                        )
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            display_sentiment_result(sentiment, confidence)
                        with col2:
                            st.plotly_chart(create_probability_chart(class_probs), use_container_width=True)
            st.markdown("---")

if __name__ == "__main__":
    main()
