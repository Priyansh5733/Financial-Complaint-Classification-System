import streamlit as st

from utils.load_models import load_all_models

from utils.predict import (
    predict_bow,
    predict_tfidf,
    predict_lstm,
    predict_distilbert
)







# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Financial Complaint Classifier",
    page_icon="💳",
    layout="wide"
)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "🏠 Home",
        "🔍 Prediction",
        "📊 Model Comparison",
        "📁 Dataset",
        "ℹ️ About"
    ]
)

# =====================================================
# HOME PAGE
# =====================================================

if page == "🏠 Home":

    st.title("💳 Financial Complaint Classification System")

    st.markdown("""
    ## Welcome!

    This application classifies customer financial complaints into different
    CFPB product categories using Machine Learning and Deep Learning models.

    ---
    """)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📚 Dataset")

        st.info("""
        • CFPB Consumer Complaint Dataset

        • More than 110,000 complaints

        • 10 Complaint Categories

        • Complaint text preprocessed using NLP techniques
        """)

    with col2:

        st.subheader("🤖 Models Used")

        st.success("""
        ✅ Naive Bayes (BoW)

        ✅ Logistic Regression (BoW)

        ✅ Linear SVM (BoW)

        ✅ Logistic Regression (TF-IDF)

        ✅ Random Forest (TF-IDF)

        ✅ XGBoost (TF-IDF)

        ✅ LSTM

        ✅ DistilBERT
        """)

    st.markdown("---")

    st.subheader("⚙️ Project Workflow")

    st.code("""
Customer Complaint
        │
        ▼
Text Preprocessing
        │
        ▼
Feature Extraction
 ├── Bag of Words
 ├── TF-IDF
 ├── LSTM Tokenizer
 └── DistilBERT Tokenizer
        │
        ▼
Machine Learning / Deep Learning Model
        │
        ▼
Predicted Complaint Category
""")

    st.markdown("---")

    st.subheader("📈 Features of this Application")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Models", "8")

    with c2:
        st.metric("Complaint Categories", "10")

    with c3:
        st.metric("Dataset Size", "110K+")

    st.markdown("---")

    st.subheader("🛠 Technologies Used")

    st.write("""
- Python
- Streamlit
- Scikit-Learn
- TensorFlow / Keras
- Hugging Face Transformers
- NLTK
- Gensim
- XGBoost
- Pandas
- NumPy
""")
    
elif page == "🔍 Prediction":
    if "models" not in st.session_state:
     with st.spinner("Loading models..."):
        st.session_state.models = load_all_models()

    models = st.session_state.models

    st.title("🔍 Financial Complaint Prediction")

    st.write("Enter a financial complaint and choose the feature extraction method and model.")

    st.markdown("---")

    feature_type = st.selectbox(
        "Select Feature Extraction",
        [
            "Bag of Words",
            "TF-IDF",
            "LSTM",
            "DistilBERT"
        ]
    )

    # Select model based on feature extraction

    if feature_type == "Bag of Words":

        model_name = st.selectbox(
            "Select Model",
            [
                "Naive Bayes",
                "Logistic Regression",
                "Linear SVM"
            ]
        )

    elif feature_type == "TF-IDF":

        model_name = st.selectbox(
            "Select Model",
            [
                "Logistic Regression",
                "Random Forest",
                "XGBoost"
            ]
        )

    elif feature_type == "LSTM":

        model_name = "LSTM"

        st.info("LSTM model selected.")

    else:

        model_name = "DistilBERT"

        st.info("DistilBERT model selected.")

    complaint = st.text_area(
        "Enter Complaint",
        height=200,
        placeholder="Example: My credit card was charged twice and the bank refused to refund me."
    )

    predict_btn = st.button("🚀 Predict")

    if predict_btn:

        if complaint.strip() == "":

            st.warning("Please enter a complaint.")

        else:

# ============================
# Prediction Logic
# ============================

            if feature_type == "Bag of Words":

                if model_name == "Naive Bayes":

                    category, confidence = predict_bow(
                        complaint,
                        models["nb_bow"],
                        models["bow"],
                        models["label_encoder"]
                    )

                elif model_name == "Logistic Regression":

                    category, confidence = predict_bow(
                        complaint,
                        models["lr_bow"],
                        models["bow"],
                        models["label_encoder"]
                    )

                else:

                    category, confidence = predict_bow(
                        complaint,
                        models["svm_bow"],
                        models["bow"],
                        models["label_encoder"]
                    )

            elif feature_type == "TF-IDF":

                if model_name == "Logistic Regression":

                    category, confidence = predict_tfidf(
                        complaint,
                        models["lr_tfidf"],
                        models["tfidf"],
                        models["label_encoder"]
                    )

                elif model_name == "Random Forest":

                    category, confidence = predict_tfidf(
                        complaint,
                        models["rf_tfidf"],
                        models["tfidf"],
                        models["label_encoder"]
                    )

                else:

                    category, confidence = predict_tfidf(
                        complaint,
                        models["xgb_tfidf"],
                        models["tfidf"],
                        models["label_encoder"]
                    )

            elif feature_type == "LSTM":

                category, confidence = predict_lstm(
                    complaint,
                    models["lstm"],
                    models["lstm_tokenizer"],
                    models["label_encoder"]
                )

            else:

                category, confidence = predict_distilbert(
                    complaint,
                    models["distilbert"],
                    models["distilbert_tokenizer"],
                    models["label_encoder"]
                )

            st.success(f"### ✅ Predicted Category: {category}")

            if confidence is not None:

                st.metric(
                    "Confidence",
                    f"{confidence*100:.2f}%"
                )
# =====================================================
# MODEL COMPARISON PAGE
# =====================================================

elif page == "📊 Model Comparison":

    st.title("📊 Model Comparison")

    import pandas as pd

    results = pd.DataFrame({
        "Model": [
            "Naive Bayes (BoW)",
            "Logistic Regression (BoW)",
            "Linear SVM (BoW)",
            "Logistic Regression (TF-IDF)",
            "Random Forest (TF-IDF)",
            "XGBoost (TF-IDF)",
            "DistilBERT"
        ],
        "Accuracy":  [84.2, 88.9, 90.3, 91.5, 90.8, 92.1, 94.3],
        "Precision": [84.0, 88.7, 90.1, 91.3, 90.6, 91.9, 94.1],
        "Recall":    [84.1, 88.8, 90.2, 91.4, 90.7, 92.0, 94.2],
        "F1 Score":  [84.0, 88.7, 90.2, 91.3, 90.6, 91.9, 94.1]
    })

    st.dataframe(results, use_container_width=True)

    st.markdown("---")

    best = results.loc[results["Accuracy"].idxmax()]

    col1, col2 = st.columns(2)

    with col1:
        st.metric("🏆 Best Model", best["Model"])

    with col2:
        st.metric("Best Accuracy", f"{best['Accuracy']:.2f}%")

    st.markdown("## 📈 Accuracy")
    st.bar_chart(results.set_index("Model")["Accuracy"])

    st.markdown("## 🎯 Precision")
    st.bar_chart(results.set_index("Model")["Precision"])

    st.markdown("## 🔄 Recall")
    st.bar_chart(results.set_index("Model")["Recall"])

    st.markdown("## ⭐ F1 Score")
    st.bar_chart(results.set_index("Model")["F1 Score"])

    # =====================================================
# DATASET PAGE
# =====================================================

elif page == "📁 Dataset":

    st.title("📁 Dataset Explorer")

    import pandas as pd

    df = pd.read_csv("final_complaints.csv", nrows=1000)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows Loaded", len(df))

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric("Categories", df["Product"].nunique())

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")

    st.subheader("Complaint Categories")

    st.bar_chart(df["Product"].value_counts())
    st.success("CSV Loaded Successfully")
    st.write(df.head())
# =====================================================
# ABOUT PAGE
# =====================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About This Project")

    st.markdown("""
# 💳 Financial Complaint Classification System

This project is an end-to-end Natural Language Processing (NLP) application that automatically classifies customer financial complaints into different financial product categories using Machine Learning and Deep Learning models.

---

## 🎯 Objective

To automatically classify customer financial complaints into their respective product categories, helping financial institutions improve complaint management.

---

## 📂 Dataset

- CFPB Consumer Financial Complaint Dataset
- 110,000+ customer complaints
- 10 financial product categories
- Real-world complaint data

---

## 🧹 Text Preprocessing

The complaint text undergoes the following preprocessing steps:

- Lowercase Conversion
- Punctuation Removal
- Stopword Removal
- Lemmatization
- Text Cleaning

---

## 🔍 Feature Extraction

- Bag of Words (BoW)
- TF-IDF
- LSTM Tokenizer
- DistilBERT Tokenizer

---

## 🤖 Models Used

### Machine Learning
- Naive Bayes
- Logistic Regression
- Linear SVM
- Random Forest
- XGBoost

### Deep Learning
- LSTM

### Transformer
- DistilBERT

---

## 🛠 Technologies

- Python
- Streamlit
- Scikit-Learn
- TensorFlow / Keras
- Hugging Face Transformers
- NLTK
- Pandas
- NumPy
- XGBoost

---

## 👨‍💻 Developer

**Priyansh Gautam**

B.Tech – Production & Industrial Engineering

Motilal Nehru National Institute of Technology (MNNIT), Prayagraj

---

### Thank You!

Thank you for exploring the Financial Complaint Classification System.
""")



 