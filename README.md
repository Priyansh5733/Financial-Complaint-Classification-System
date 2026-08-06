🏦 Customer Complaint Classification System using Machine Learning, Deep Learning & Transformers.









📌 Overview

The Customer Complaint Classification System is an end-to-end NLP project that automatically classifies financial customer complaints into their respective product categories using Machine Learning, Deep Learning, and Transformer-based models.

The project compares multiple text representation techniques and classification algorithms to identify the best-performing approach for complaint classification. An interactive Streamlit application was developed to allow users to test different models on custom complaint text.

🎯 Problem Statement

Financial institutions receive thousands of customer complaints daily. Manually categorizing these complaints is time-consuming and inefficient.

This project automates the classification process using Natural Language Processing (NLP), reducing manual effort while enabling faster complaint routing and analysis.

✨ Key Features
Automated financial complaint classification
Comparison of 9 Machine Learning & Deep Learning models
Multiple text representation techniques
Bag of Words (BoW)
TF-IDF
Word2Vec
LSTM Tokenization
DistilBERT Tokenization
Interactive Streamlit application
Model comparison dashboard
End-to-end NLP pipeline
📂 Dataset

Dataset: Consumer Complaints Financial Products Dataset

The dataset contains customer complaint text along with the associated financial product category.

Example categories include:

Credit Card
Mortgage
Student Loan
Debt Collection
Consumer Loan
Bank Account
Money Transfer
Credit Reporting
Payday Loan
Other Financial Services
🛠 Tech Stack
Programming Language
Python
Libraries
Pandas
NumPy
Matplotlib
Scikit-learn
TensorFlow / Keras
Hugging Face Transformers
Streamlit
Joblib
🔄 Project Workflow
Data Collection
Data Cleaning
Text Preprocessing
Label Encoding
Train-Test Split
Feature Extraction
Model Training
Model Evaluation
Streamlit Deployment
🧠 Feature Extraction Techniques
Technique	Purpose	Models
Bag of Words	Word frequency representation	Naive Bayes, Logistic Regression, Linear SVM
TF-IDF	Importance-weighted representation	Logistic Regression, Random Forest, XGBoost
Word2Vec	Dense semantic embeddings	Machine Learning classifier
Tokenization & Padding	Sequential representation	LSTM
DistilBERT Tokenizer	Context-aware embeddings	DistilBERT
🤖 Models Implemented
Machine Learning
Naive Bayes
Logistic Regression (BoW)
Linear Support Vector Machine (SVM)
Logistic Regression (TF-IDF)
Random Forest
XGBoost
Word2Vec-based Classifier
Deep Learning
Long Short-Term Memory (LSTM)
Transformer
DistilBERT
📊 Model Evaluation

The models were evaluated using standard classification metrics:

Accuracy
Precision
Recall
F1-Score

The project compares traditional Machine Learning algorithms, Deep Learning architectures, and Transformer models to identify the most effective solution for complaint classification.

💻 Streamlit Application

The application provides an intuitive interface for testing the trained models.

Features
Complaint Prediction
Model Selection
Performance Comparison
Dataset Overview
Project Information

Users can input a complaint and obtain predictions from multiple trained models.

📁 Project Structure
Customer-Complaint-Classification/
│
├── dataset/
├── notebooks/
├── models/
│   ├── bow_vectorizer.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── word2vec_model.pkl
│   ├── label_encoder.pkl
│   ├── lstm_model.h5
│   ├── distilbert_model/
│
├── streamlit_app/
│   ├── app.py
│
├── requirements.txt
├── README.md
└── LICENSE
🚀 Installation

Clone the repository

git clone https://github.com/your-username/Customer-Complaint-Classification.git

Move into the project folder

cd Customer-Complaint-Classification

Install dependencies

pip install -r requirements.txt

Run the application

streamlit run app.py
📸 Application Screenshots

Include screenshots of:

Home Page
Prediction Interface
Model Comparison
Dataset Overview
About Page
📈 Future Enhancements
Fine-tune larger Transformer models
Hyperparameter optimization
Explainable AI using SHAP/LIME
REST API with FastAPI
Cloud deployment using Streamlit Community Cloud or Hugging Face Spaces
🎓 Skills Demonstrated
Natural Language Processing (NLP)
Text Preprocessing
Feature Engineering
Machine Learning
Deep Learning
Transformer Models
Word Embeddings
Model Evaluation
Streamlit Development
Model Comparison
Python Programming
