# 📚 Kindle Review Sentiment Analysis using NLP

<p align="center">
  <img src="https://img.shields.io/badge/Python-NLP-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/Machine%20Learning-Text%20Classification-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/Streamlit-Interactive%20App-red?style=for-the-badge&logo=streamlit">
  <img src="https://img.shields.io/badge/Project-End--to--End%20NLP-black?style=for-the-badge">
</p>

<p align="center">
🚀 An end-to-end Natural Language Processing project that analyzes Kindle book reviews  
and predicts whether the sentiment is <b>Positive</b> or <b>Negative</b> using Machine Learning.
</p>

---

## 📌 Table of Contents

* ✨ Features
* 🧠 Project Overview
* ⚙️ Tech Stack
* 📂 Project Structure
* 🔄 NLP Pipeline
* 🤖 Model Details
* 📊 Dataset Information
* 🚀 Streamlit App
* ▶️ Getting Started
* 🧩 Skills Demonstrated
* 🔮 Future Improvements
* 🧑‍💻 Author

---

## ✨ Features

* 📚 Sentiment analysis on Kindle reviews
* 🔍 Text preprocessing using NLP techniques
* ⚡ TF-IDF feature extraction
* 🤖 Machine Learning classification model
* 🌐 Interactive Streamlit web application
* 🎯 Clean UI with custom styling

---

## 🧠 Project Overview

This project demonstrates a complete **NLP workflow**, starting from raw text data to a deployed interactive application.

The goal is to understand user sentiment from Kindle reviews and showcase strong fundamentals in:

* Text preprocessing
* Feature engineering
* Machine learning modeling
* Application deployment

The project bridges **Machine Learning + NLP + UI development**, reflecting real-world AI application building.

---

## ⚙️ Tech Stack

| Technology     | Purpose             |
| -------------- | ------------------- |
| Python         | Core programming    |
| NLTK           | Text preprocessing  |
| Scikit-Learn   | Machine Learning    |
| Pandas & NumPy | Data handling       |
| Streamlit      | Web App Interface   |
| Pickle         | Model serialization |

---

## 📂 Project Structure

```
Kindel-Review-Using-NLP
│
├── app.py                # Streamlit application
├── nb_model.pkl          # Trained ML model
├── tfidf.pkl             # Vectorizer
├── notebooks/            # Model training notebooks
├── dataset/              # Data files
├── assets/               # Images or UI files
└── README.md
```

---

## 🔄 NLP Pipeline

```
Raw Text
   ↓
Lowercasing
   ↓
Stopword Removal
   ↓
Lemmatization
   ↓
TF-IDF Vectorization
   ↓
Naive Bayes Model
   ↓
Sentiment Prediction
```

Steps Included:

* Cleaning special characters
* Removing stopwords
* Lemmatizing tokens
* Converting text into numeric vectors

---

## 🤖 Model Details

Model Used:

* Multinomial Naive Bayes

Why Naive Bayes?

* Efficient for text classification
* Works well with TF-IDF features
* Fast training and prediction

Evaluation Focus:

* Accuracy
* Precision
* Recall

---

## 📊 Dataset Information

The dataset contains Kindle product reviews including:

* Review text
* Sentiment label

Goal:

Predict whether a review expresses a **positive** or **negative** sentiment.

---

## 🚀 Streamlit App

The project includes a fully interactive Streamlit interface where users can:

* Enter a custom review
* Click **Analyze Sentiment**
* Instantly view prediction results

Run locally:

```
streamlit run app.py
```

---

## ▶️ Getting Started

Clone repository:

```
git clone https://github.com/Vashishtha05/Kindel-Review-Using-NLP.git
```

Install dependencies:

```
pip install streamlit nltk scikit-learn pandas numpy
```

Download NLTK resources:

```
python -m nltk.downloader stopwords wordnet
```

Run application:

```
streamlit run app.py
```

---

## 🧩 Skills Demonstrated

* Natural Language Processing
* Text Cleaning & Tokenization
* Feature Engineering (TF-IDF)
* Machine Learning Classification
* Streamlit App Development
* End-to-End ML Workflow

---

## 🔮 Future Improvements

* Deep Learning models (LSTM / Transformers)
* Live model confidence visualization

---

## 🧑‍💻 Author

**Vashishtha Verma**

* AI / Machine Learning Enthusiast
* Generative AI & Agentic AI Explorer
* Strong foundation in DSA and Software Engineering

---

<p align="center">
⭐ If you found this project useful, consider giving it a star!
</p>
