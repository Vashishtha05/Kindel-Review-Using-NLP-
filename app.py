import streamlit as st
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ==========================
# Page Config
# ==========================
st.set_page_config(
    page_title="Kindle Review Sentiment Analyzer",
    page_icon="📚",
    layout="centered"
)

# ==========================
# Custom Dark Gradient UI
# ==========================
st.markdown("""
<style>

.stApp {
    background-color: #C9CFCC;  /* grey-green tone like your image */
}

/* Text area styling */
textarea {
    background-color: #E5E7EB !important;
    color: black !important;
}

/* Sidebar background */
section[data-testid="stSidebar"] {
    background-color: #BFC6C3;
}

/* Button styling */
div.stButton > button {
    background-color: #6B7280;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
    border: none;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# Title Section
# ==========================
st.markdown("""
<h1 style='text-align:center;'>📚 Kindle Review Sentiment Analyzer</h1>
<p style='text-align:center;font-size:18px;color:black;'>
Analyze whether a Kindle review is Positive 😊 or Negative 😞 using Machine Learning
</p>
""", unsafe_allow_html=True)

# ==========================
# Sidebar Info
# ==========================
st.sidebar.title("🧠 Project Info")

st.sidebar.write("""
**Model:** Naive Bayes  
**Vectorizer:** TF-IDF  

**NLP Pipeline:**
- Lowercasing
- Stopword Removal
- Lemmatization
""")

# ==========================
# Load Model & Vectorizer
# ==========================
model = pickle.load(open("nb_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf.pkl", "rb"))

# ==========================
# NLP Setup
# ==========================
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub('[^a-z A-Z 0-9-]+', '', text)
    text = " ".join([w for w in text.split() if w not in stopwords.words('english')])
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

# ==========================
# Input Section
# ==========================
st.subheader("✍️ Enter Kindle Review")

user_input = st.text_area(
    "Type your review here",
    height=150,
    placeholder="Example: This book is amazing and very helpful..."
)

# ==========================
# Prediction
# ==========================
if st.button("🔍 Analyze Sentiment"):

    if user_input.strip() == "":
        st.warning("Please enter a review first.")
    else:
        clean_text = preprocess(user_input)
        vectorized = vectorizer.transform([clean_text]).toarray()
        prediction = model.predict(vectorized)[0]

        st.markdown("---")

        if prediction == 1:
            st.success("✅ Positive Review Detected")
        else:
            st.error("❌ Negative Review Detected")

# ==========================
# Footer
# ==========================
st.markdown("""
<hr>
<p style='text-align:center;color:#94a3b8;font-size:14px;'>
NLP • ML • Python
</p>
""", unsafe_allow_html=True)
