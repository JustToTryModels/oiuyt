<div align="center">

# 🎧 Sentiment Analysis of Apple AirPods Reviews

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-154f3c?style=for-the-badge&logo=python&logoColor=white)](https://www.nltk.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

An end-to-end NLP project that collects **16,000+ customer reviews** of Apple AirPods (2nd Generation) from Walmart, performs comprehensive text preprocessing and exploratory data analysis, and benchmarks **two distinct sentiment analysis approaches** — traditional Machine Learning (Logistic Regression, Naive Bayes, Random Forest) and a fine-tuned **ALBERT transformer** — to classify reviews as Positive, Neutral, or Negative with up to **100% accuracy on manual test reviews**.

</div>

<br>

---

## 📋 Table of Contents

- [What is Sentiment Analysis?](#-what-is-sentiment-analysis)
- [About the Product](#-about-the-product)
- [Project Overview](#-project-overview)
- [Demo](#-demo)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
  - [Data Collection & Extraction](#-1-data-collection--extraction)
  - [Exploratory Data Analysis](#-2-exploratory-data-analysis)
  - [Text Preprocessing Pipeline](#%EF%B8%8F-3-text-preprocessing-pipeline)
  - [Method 1 — ML-Based Sentiment Analysis](#-4-method-1--ml-based-sentiment-analysis)
  - [Method 2 — Fine-Tuned ALBERT Transformer](#-5-method-2--fine-tuned-albert-transformer)
- [Model Comparison & Results](#%EF%B8%8F-model-comparison--results)
- [Key Findings](#-key-findings)
- [Installation & Usage](#%EF%B8%8F-installation--usage)

<br>

---

## ❓ What is Sentiment Analysis?

**Sentiment Analysis** (also known as opinion mining) is a Natural Language Processing (NLP) technique used to determine the emotional tone behind a body of text. It classifies text into categories such as:

- 😊 **Positive** — Expresses satisfaction, happiness, or appreciation
- 😐 **Neutral** — Balanced, informative, or lacks strong emotion
- 😠 **Negative** — Expresses dissatisfaction, frustration, or disappointment

This project applies sentiment analysis to real-world **product reviews**, enabling businesses to understand customer perception at scale, identify product strengths and weaknesses, and make data-driven decisions to improve customer satisfaction.

<br>

---

## 🎧 About the Product

<div align="center">

This project analyzes customer sentiment for the **Apple AirPods with Charging Case (2nd Generation)** as sold on Walmart.com.

<br>

### Product Gallery

<table>
<tr>
<td align="center"><img src="https://raw.githubusercontent.com/JustToTryModels/oiuyt/refs/heads/main/Image-1_Apple-AirPods-with-Charging-Case-2nd-Generation.avif" alt="AirPods - Main Product" width="180"/></td>
<td align="center"><img src="https://github.com/JustToTryModels/oiuyt/blob/main/Image-2_Apple-AirPods-with-Charging-Case-2nd-Generation.webp?raw=true" alt="AirPods - Charging Case" width="180"/></td>
<td align="center"><img src="https://github.com/JustToTryModels/oiuyt/blob/main/Image-3_Apple-AirPods-with-Charging-Case-2nd-Generation.webp?raw=true" alt="AirPods - In Case" width="180"/></td>
<td align="center"><img src="https://github.com/JustToTryModels/oiuyt/blob/main/Image-4_Apple-AirPods-with-Charging-Case-2nd-Generation.webp?raw=true" alt="AirPods - Detail View" width="180"/></td>
<td align="center"><img src="https://github.com/JustToTryModels/oiuyt/blob/main/Image-5_Apple-AirPods-with-Charging-Case-2nd-Generation.webp?raw=true" alt="AirPods - Package" width="180"/></td>
</tr>
</table>

<br>

### Product Overview

**Apple AirPods with Charging Case (2nd Generation)** deliver an incredible wireless headphone experience with the power of the Apple H1 headphone chip. Simply take them out and they're ready to use with all your Apple devices. Put them in your ears and they connect immediately, immersing you in rich, high-quality sound.

</div>

> **Note:** This model contains the standard Lightning charging case, NOT the wireless charging case.

<br>

<div align="center">

### Key Specifications

| Specification | Details |
|---|---|
| **Model** | Apple AirPods (2nd Generation) |
| **Chip** | Apple H1 Headphone Chip |
| **Battery Life** | 5 hours listening time (single charge) |
| **Connectivity** | Bluetooth / True Wireless |
| **Charging** | Lightning Connector (Non-Wireless) |
| **Virtual Assistant** | Siri ("Hey Siri" voice-activated) |
| **Color** | White |
| **Weight** | 1.34 oz |
| **Price** | ~~$229.00~~ **Now $89.00** (You save $140.00) |

</div>

<br>

### Key Features

- **Automatically on, automatically connected** — Seamless pairing with all Apple devices
- **Quick access to Siri** — Just say "Hey Siri" for hands-free control
- **All-day battery life** — Multiple charges from the Charging Case
- **Fast charging** — 15 minutes in the case = 3 hours of listening time
- **Rich, high-quality audio** — Powered by Apple H1 chip
- **Seamless device switching** — Move between iPhone, iPad, Mac, and Apple Watch
- **Crystal-clear calls** — Dual beamforming microphones filter out background noise
- **Double-tap control** — Play, skip, or answer calls with a simple tap

<br>


### What's Included

- AirPods (2nd Generation)
- Charging Case with Lightning Connector
- Lightning to USB-A Cable
- Documentation

<br>

---

## 🎯 Project Overview

### Objective

Collect, clean, and analyze Apple AirPods (2nd Generation) reviews from Walmart, then build and compare multiple sentiment classification models to identify the most accurate approach.

<div align="center">

### 🛣️ Approach

| Component | Description |
|---|---|
| **Data Source** | Walmart.com (16,849 reviews collected) |
| **Collection Method** | HTML page download + JSON extraction (rate-limit compliant) |
| **After Cleaning** | 11,569 unique reviews |
| **Sentiment Labeling** | Llama 3.2-1B-Instruct (for ALBERT training data) + VADER (for ML methods) |
| **ML Models Evaluated** | Logistic Regression, Multinomial Naive Bayes, Random Forest |
| **Feature Extraction** | Bag of Words (Unigram), BoW with N-grams (1-3), TF-IDF |
| **Deep Learning Model** | Fine-tuned ALBERT (`albert-base-v2`) |
| **Imbalance Handling** | Class weighting + Weighted Cross-Entropy Loss |
| **Deployment** | Streamlit Web Application |

</div>

<br>

---

## 🚀 Demo

Try the live sentiment analysis app here:

<div align="center">

[![Open Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://project-sentiment-analysis.streamlit.app/)

<table>
<tr>
<td align="center"><img src="https://github.com/JustToTryModels/oiuyt/blob/main/Screenshot%202026-02-09%20100355.png?raw=true" alt="Positive Sentiment Example" width="380" /></td>
<td align="center"><img src="https://github.com/JustToTryModels/oiuyt/blob/main/Screenshot%202026-02-09%20100112.png?raw=true" alt="Negative Sentiment Example" width="380" /></td>
<td align="center"><img src="https://github.com/JustToTryModels/oiuyt/blob/main/Screenshot%202026-02-09%20100315.png?raw=true" alt="Neutral Sentiment Example" width="380" /></td>
</tr>
<tr>
<td align="center"><b>😊 Positive Sentiment</b></td>
<td align="center"><b>😠 Negative Sentiment</b></td>
<td align="center"><b>😐 Neutral Sentiment</b></td>
</tr>
</table>

</div>

> Enter any Apple AirPods review and get an instant sentiment prediction powered by fine-tuned ALBERT!

<br>

---

## 📁 Project Structure

```
Project-Sentiment-Analysis/
├── Notebooks/
│   ├── Web_Scraping_Walmart_Reviews.ipynb          # HTML download + JSON extraction
│   └── Sentiment_Analysis_Complete.ipynb            # Full analysis & modeling notebook
├── Data/
│   ├── Walmart_AirPods_All_Reviews.csv              # Extracted reviews dataset
│   └── Walmart_AirPods_Sentiment_Llama_3.2_1B_Instruct.csv  # Llama-labeled reviews
├── Model/
│   └── AirPods_fine_tuned_ALBERT_base_v2_model_balanced/     # Saved fine-tuned ALBERT model
├── app.py                                           # Streamlit web application
├── requirements.txt                                 # Python dependencies
├── LICENSE                                          # MIT License
└── README.md                                        # Project documentation
```

<br>

---

## 📊 Dataset

<div align="center">

### Overview

| Metric | Value |
|---|---|
| **Total HTML Pages Downloaded** | 1,686 pages |
| **Reviews Extracted** | 16,849 |
| **After Removing Nulls** | 16,833 |
| **After Removing Duplicates** | 11,569 |
| **Date Range** | March 2019 — January 2025 |
| **Source** | Walmart.com |
| **Product** | Apple AirPods (2nd Generation) |

</div>

<br>

<div align="center">

### Features

| Feature | Description |
|---|---|
| `Review_Title` | Short summary title of the review |
| `User_Name` | Reviewer's username |
| `Rating` | Star rating (1–5) |
| `Review_Date` | Date of the review |
| `Review_Text` | Full review body text |

</div>

<br>

<div align="center">

### Rating Distribution

| Rating | Count | Percentage |
|:---:|:---:|:---:|
| ⭐⭐⭐⭐⭐ (5) | 6,722 | 58.1% |
| ⭐⭐⭐⭐ (4) | 2,386 | 20.6% |
| ⭐⭐⭐ (3) | 855 | 7.4% |
| ⭐⭐ (2) | 356 | 3.1% |
| ⭐ (1) | 1,250 | 10.8% |

</div>

> **Mean Rating: 4.12 / 5** — Strong overall customer satisfaction with a notable segment of 1-star reviews.

<br>

---

## 🔬 Methodology

### 🌐 1. Data Collection & Extraction

Since Walmart prohibits direct web scraping, we employed a **two-step data collection strategy** that respects rate limits while gathering comprehensive review data:

#### **Step 1: HTML Page Download**

- **Method:** Automated download of review pages as static HTML files using the `requests` library
- **Anti-Bot Measures Implemented:**
  - Rotating user agents (4 different browser signatures)
  - Random delays between requests (2-20 seconds)
  - Proxy rotation (4 proxy servers)
  - CAPTCHA detection and automatic retry logic
  - Maximum 5 retry attempts per page with exponential backoff

<div align="center">

| Rating Filter | Pages Downloaded |
|:---:|:---:|
| 1-3 Stars | 339 pages |
| 4 Stars | 347 pages |
| 5 Stars (Batch 1) | 502 pages |
| 5 Stars (Batch 2) | 498 pages |
| **Total** | **1,686 pages** |

</div>

> **Note:** 5-star reviews were split into two batches (502 + 498 pages) to avoid triggering rate-limiting on the 1,000-page dataset.

<br>

#### **Step 2: JSON Data Extraction**

- **Method:** Parsed downloaded HTML files using BeautifulSoup to extract embedded JSON data
- **Target:** Located `<script id="__NEXT_DATA__" type="application/json">` tags containing structured review data
- **Extracted Fields:**
  - Review text and title
  - Star rating
  - Reviewer username
  - Submission date
- **Result:** 16,849 raw reviews consolidated into a single CSV file

<br>

### 📈 2. Exploratory Data Analysis

Comprehensive EDA was performed across multiple dimensions:

<div align="center">

| Analysis | Key Insight |
|---|---|
| **Rating Distribution** | ~79% of reviews are 4-5 stars; highly positive skew |
| **Average Rating Over Time** | U-shaped trend: dip in 2021–2022, recovery in 2023–2025 |
| **Review Volume Over Time** | Significant growth starting early 2023, peaking in late 2024 |
| **Monthly Patterns** | November peak (1,642 reviews) — likely holiday/promotional season |
| **Review Length** | Titles mostly < 50 chars; bodies vary widely with a long tail |
| **Top Review Titles** | "Great" and "Good" dominate — strong positive sentiment signal |

</div>

<br>

### ✂️ 3. Text Preprocessing Pipeline

A rigorous multi-step text cleaning pipeline was applied before ML-based modeling:

```
Raw Review → Lowercasing → URL Removal → Emoji Removal → Punctuation Removal
→ Special Character Removal → Tokenization → Stopword Removal (Custom + NLTK)
→ Lemmatization (WordNet) → Cleaned Review
```

<div align="center">

| Step | Technique | Example |
|---|---|---|
| **Normalization** | `str.lower()` | "They Don't Work" → "they don't work" |
| **URL Removal** | Regex `r'http\S+\|www\S+\|https\S+'` | Removes any web links |
| **Emoji Removal** | `emoji.replace_emoji()` | 🎧👍 → *(removed)* |
| **Punctuation** | `str.maketrans` | "don't" → "dont" |
| **Special Chars** | Regex `r'[^a-zA-Z\s]'` | "2nd" → "nd" |
| **Tokenization** | `word_tokenize()` | "love easy great" → ["love", "easy", "great"] |
| **Stopwords** | Custom 500+ stopword list | Removes "the", "is", "and", etc. |
| **Lemmatization** | `WordNetLemmatizer` | "loves" → "love", "running" → "running" |

</div>

<br>

### Post-Preprocessing Visualizations

After cleaning, text analytics revealed what customers talk about most:

<div align="center">

| Visualization | Key Takeaway |
|---|---|
| **Word Cloud** | "sound", "quality", "love", "battery", "great" dominate |
| **Top 20 Unigrams** | `love` (3,500+), `sound` (3,200+), `quality` (2,800+), `great` (2,500+) |
| **Top 20 Bigrams** | "sound quality", "battery life", "noise cancellation", "easy use" |
| **Top 20 Trigrams** | "sound quality great", "battery life good", "noise cancellation great" |
| **POS Tag Distribution** | Nouns (NN) and Adjectives (JJ) dominate — feature-focused reviews |

</div>

<br>

---

### 🤖 4. Method 1 — ML-Based Sentiment Analysis

#### Sentiment Labeling (VADER)

Reviews were labeled using **NLTK's VADER** (Valence Aware Dictionary and sEntiment Reasoner):
- **Compound score ≥ 0.05** → Positive
- **Compound score ≤ -0.05** → Negative
- **Otherwise** → Neutral

#### Feature Extraction

Three vectorization techniques were used to convert text into numerical features:

<div align="center">

| Technique | Max Features | Description |
|---|:---:|---|
| **CountVectorizer (Unigram)** | 500 | Word frequency counts (single words) |
| **Bag of Words (N-gram)** | 3,000 | Unigrams + Bigrams + Trigrams |
| **TF-IDF** | 6,979 | Term frequency weighted by inverse document frequency |

</div>

#### Models Trained

Three classifiers were trained on each of the three feature sets (9 total model-feature combinations):

1. **Logistic Regression** — `solver='liblinear'`, `class_weight='balanced'`
2. **Multinomial Naive Bayes** — with computed sample weights
3. **Random Forest** — `n_estimators=200`, `class_weight='balanced'`

<br>

### 🧠 5. Method 2 — Fine-Tuned ALBERT Transformer

#### Sentiment Labeling (Llama 3.2-1B-Instruct)

For the deep learning approach, reviews were labeled using **Meta's Llama 3.2-1B-Instruct** model with a carefully crafted prompt to classify each review as Positive, Negative, or Neutral.

#### Class Distribution (After Llama Labeling)

<div align="center">

| Class | Weight | Interpretation |
|---|:---:|---|
| Negative (0) | 1.4711 | Upweighted |
| Neutral (1) | **3.0424** | Heavily upweighted (minority class) |
| Positive (2) | 0.5021 | Downweighted (majority class) |

**Imbalance Ratio: 6.06x** (max/min class)

</div>

#### Model Architecture & Training

<div align="center">

| Parameter | Value |
|---|---|
| **Base Model** | `albert-base-v2` (12M parameters) |
| **Task** | Sequence Classification (3 classes) |
| **Max Sequence Length** | 512 tokens |
| **Epochs** | 10 |
| **Batch Size** | 8 (train) / 16 (eval) |
| **Learning Rate** | 2e-5 (linear schedule) |
| **Warmup Steps** | 500 |
| **Weight Decay** | 0.01 |
| **Imbalance Strategy** | Weighted Cross-Entropy Loss |
| **Best Model Selection** | Macro F1-Score |

</div>

<br>

---

## ⚔️ Model Comparison & Results

### 📊 ML Models — Test Set Performance (80/20 Split)

<div align="center">

| Feature Set | Model | Accuracy | Macro F1 | Neg. F1 | Neu. F1 | Pos. F1 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| CountVectorizer | Logistic Regression | 89.07% | 0.78 | 0.62 | 0.77 | 0.95 |
| CountVectorizer | Naive Bayes | 78.69% | 0.64 | 0.45 | 0.59 | 0.88 |
| CountVectorizer | Random Forest | 89.46% | 0.75 | 0.50 | 0.78 | 0.95 |
| **BoW (N-gram)** | **Logistic Regression** | **90.54%** | **0.81** | **0.68** | **0.80** | **0.95** |
| BoW (N-gram) | Naive Bayes | 78.65% | 0.66 | 0.49 | 0.60 | 0.88 |
| BoW (N-gram) | Random Forest | 91.10% | 0.79 | 0.60 | 0.83 | 0.95 |
| TF-IDF | Logistic Regression | 90.67% | 0.81 | 0.69 | 0.79 | 0.95 |
| TF-IDF | Naive Bayes | 83.36% | 0.70 | 0.57 | 0.62 | 0.91 |
| TF-IDF | Random Forest | 90.06% | 0.75 | 0.50 | 0.81 | 0.95 |

</div>

<br>

### 🧠 Fine-Tuned ALBERT — Validation Set Performance

<div align="center">

| Metric | Negative | Neutral | Positive | Overall |
|---|:---:|:---:|:---:|:---:|
| **Precision** | 0.87 | 0.59 | 0.94 | — |
| **Recall** | 0.84 | 0.64 | 0.94 | — |
| **F1-Score** | 0.85 | 0.61 | 0.94 | — |
| **Accuracy** | — | — | — | **88.50%** |
| **Macro F1** | — | — | — | **0.80** |
| **Weighted F1** | — | — | — | **0.89** |

</div>

<br>

### 🏆 Manual Review Evaluation (15 Hand-Crafted Reviews)

To truly test generalization, all models were evaluated on **15 manually written reviews** (5 positive, 5 neutral, 5 negative):

<div align="center">

| Model | Correct | Total | Accuracy | Macro F1 |
|:---|:---:|:---:|:---:|:---:|
| **Fine-Tuned ALBERT** 🏆 | **15** | **15** | **100.00%** | **1.0000** |
| BoW (N-gram) + LR | 10 | 15 | 66.67% | 0.6551 |
| CountVectorizer + LR | 9 | 15 | 60.00% | 0.5889 |
| TF-IDF + LR | 8 | 15 | 53.33% | 0.5004 |

</div>

<br>

### 🎯 Why ALBERT Wins

<div align="center">

| Criteria | ML Models (Best: BoW+LR) | Fine-Tuned ALBERT |
|---|---|---|
| **Contextual Understanding** | ❌ Bag-of-words; no word order | ✅ Full context via self-attention |
| **Neutral Detection** | ❌ 40% accuracy on neutral reviews | ✅ 100% on manual, 64% on val set |
| **Manual Review Accuracy** | 66.67% | **100.00%** |
| **Preprocessing Required** | Extensive (7-step pipeline) | ✅ Minimal (tokenizer handles it) |
| **Handles Sarcasm/Nuance** | ❌ Struggles with mixed sentiment | ✅ Better nuance understanding |

</div>

> The fine-tuned ALBERT model correctly classified all 15 manual test reviews, whereas the best ML model struggled with **neutral** and **mixed-sentiment** reviews, misclassifying 5 out of 15.

<br>

---

## 💡 Key Findings

### 🔑 Customer Sentiment Summary

<div align="center">

| Sentiment | Percentage | Primary Themes |
|---|:---:|---|
| 😊 **Positive** | ~53% | Sound quality, ease of use, Apple ecosystem integration, battery life |
| 😠 **Negative** | ~22% | Disconnection issues, durability (left AirPod failure), battery degradation |
| 😐 **Neutral** | ~9% | Decent but nothing exceptional, price concerns, average bass |

</div>

<br>

### 📌 What Customers Love
- **Sound quality** — The most frequently praised aspect (`"sound quality"` = top bigram)
- **Ease of use** — Seamless pairing with Apple devices
- **Battery life** — Consistently highlighted as a strong point
- **Comfort & design** — Lightweight, portable, and sleek

### ⚠️ What Customers Complain About
- **Connectivity issues** — Frequent disconnections, especially after several months
- **Left AirPod failure** — A recurring hardware defect pattern
- **Durability concerns** — Charging case issues within the first year
- **Noise cancellation** — Perceived as underwhelming vs. competitors
- **Price vs. value** — Some customers feel the product is overpriced for AirPods Gen 2

### 📈 Temporal Trends
- **U-shaped rating trend:** Ratings dipped in 2021–2022 (possibly due to aging hardware or increased competition), then recovered in 2023–2025.
- **Review volume surged** starting in 2023, with a massive peak in late 2024, indicating sustained market interest.

<br>

---

## 🛠️ Installation & Usage

### Prerequisites

- Python 3.10 or higher
- pip package manager
- GPU recommended for ALBERT inference (optional — works on CPU)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/MarpakaPradeepSai/Project-Sentiment-Analysis.git
   cd Project-Sentiment-Analysis
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Run the Streamlit App Locally

```bash
streamlit run app.py
```

The app will open in the default browser at `http://localhost:8501`

<br>

---

## 🙏 Thank You

<div align="center">
  <img src="https://github.com/MarpakaPradeepSai/Employee-Churn-Prediction/blob/main/Data/Images%20&%20GIFs/thank-you-33.gif?raw=true" alt="Thank You" width="400">
  
  If this project was helpful, please consider giving it a ⭐
</div>
