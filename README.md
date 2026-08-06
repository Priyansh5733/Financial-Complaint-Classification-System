# 🏦 Customer Complaint Classification System using Machine Learning, Deep Learning & Transformers

An end-to-end **Natural Language Processing (NLP)** project that automatically classifies financial customer complaints into their respective product categories using **Machine Learning, Deep Learning, and Transformer-based models**.

---

# 📌 Overview

The **Customer Complaint Classification System** is an end-to-end NLP project developed to automatically classify financial customer complaints into their respective product categories. The project compares multiple text representation techniques and classification algorithms to determine the most effective approach for complaint classification.

The project includes an interactive **Streamlit** application that enables users to classify custom complaint text using different trained models.

---

# 🎯 Problem Statement

Financial institutions receive thousands of customer complaints every day across multiple financial products. Manually categorizing these complaints is time-consuming, inconsistent, and difficult to scale.

This project automates complaint classification using Natural Language Processing (NLP), enabling faster complaint routing and reducing manual effort.

---

# ✨ Key Features

- Automated financial complaint classification using NLP
- Comparison of **9 Machine Learning, Deep Learning & Transformer models**
- Multiple feature extraction techniques
  - Bag of Words (BoW)
  - TF-IDF
  - Word2Vec
  - LSTM Tokenization
  - DistilBERT Tokenization
- Interactive Streamlit application
- Model performance comparison
- End-to-end NLP pipeline

---

# 📂 Dataset

The project uses the **Consumer Complaints Financial Products Dataset**, which contains customer complaint descriptions along with their corresponding financial product categories.

### Categories

- Credit Card
- Mortgage
- Student Loan
- Debt Collection
- Consumer Loan
- Bank Account
- Money Transfer
- Credit Reporting
- Payday Loan
- Other Financial Services

---

# 🛠️ Technologies Used

### Programming Language

- Python

### Libraries & Frameworks

- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- TensorFlow / Keras
- Hugging Face Transformers
- Streamlit
- Joblib

---

# 🔄 Project Workflow

1. Data Collection
2. Data Cleaning
3. Text Preprocessing
4. Label Encoding
5. Train-Test Split
6. Feature Extraction
7. Model Training
8. Model Evaluation
9. Complaint Prediction
10. Streamlit Integration

---

# 🧠 Feature Extraction Techniques

| Technique | Description | Models Used |
|-----------|-------------|-------------|
| **Bag of Words (BoW)** | Converts text into word frequency vectors. | Naive Bayes, Logistic Regression, Linear SVM |
| **TF-IDF** | Assigns importance to words based on their frequency across the corpus. | Logistic Regression, Random Forest, XGBoost |
| **Word2Vec** | Generates dense semantic word embeddings that capture contextual relationships between words. | Machine Learning Classifier |
| **Tokenization & Padding** | Converts text into fixed-length integer sequences for deep learning models. | LSTM |
| **DistilBERT Tokenizer** | Generates contextual embeddings for transformer-based classification. | DistilBERT |

---

# 🤖 Models Implemented

## Machine Learning

- Naive Bayes
- Logistic Regression (BoW)
- Linear Support Vector Machine (SVM)
- Logistic Regression (TF-IDF)
- Random Forest
- XGBoost
- Word2Vec-based Classifier

## Deep Learning

- Long Short-Term Memory (LSTM)

## Transformer

- DistilBERT

---

# 📊 Model Evaluation

The performance of all models was evaluated using standard classification metrics.

- Accuracy
- Precision
- Recall
- F1-Score

The project compares traditional Machine Learning algorithms, Deep Learning architectures, and Transformer models to determine the most effective approach for financial complaint classification.

---

# 💻 Streamlit Application

The application provides an intuitive interface for testing the trained models.

### Features

- Predict customer complaint categories
- Select different trained models
- Compare model performance
- Dataset overview
- Project information

---

# 📁 Project Structure

```text
Customer-Complaint-Classification/
│
├── Dataset/
├── Models/
├── Notebooks/
├── app.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 🚀 Installation

### Clone the Repository

```bash
git clone <repository-url>
```

### Navigate to the Project Directory

```bash
cd Customer-Complaint-Classification
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit Application

```bash
streamlit run app.py
```

---

# 📈 Future Improvements

- Hyperparameter Optimization
- Fine-tuning Larger Transformer Models
- Explainable AI using SHAP/LIME
- REST API Development using FastAPI
- Cloud Deployment

---

# 🎓 Skills Demonstrated

- Natural Language Processing (NLP)
- Text Preprocessing
- Feature Engineering
- Machine Learning
- Deep Learning
- Transformer Models
- Word Embeddings
- Model Evaluation
- Streamlit Application Development
- Comparative Model Analysis

---

# ⭐ Project Highlights

- End-to-end NLP Pipeline
- Comparison of 9 Classification Models
- Multiple Feature Extraction Techniques
- Interactive Streamlit Interface
- Machine Learning, Deep Learning & Transformer-based Classification
- Real-world Financial Complaint Dataset
- Performance Evaluation using Multiple Metrics
