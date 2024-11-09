import streamlit as st
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re

# Streamlit page configuration for a modern look
st.set_page_config(
    page_title="Fake News Detector 📰",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Download necessary NLTK data
nltk.download("punkt")
nltk.download("stopwords")

# Load the models and vectorizer from the specified directory
tfidf = pickle.load(
    open("D:\\Projects\\Fake News Detection\\Model\\vectorizer.pkl", "rb")
)
XGB_model = pickle.load(
    open("D:\\Projects\\Fake News Detection\\Model\\Xgb_model.pkl", "rb")
)
model_title = pickle.load(
    open("D:\\Projects\\Fake News Detection\\Model\\title_model.pkl", "rb")
)


# Define the process_text function
def process_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [re.sub(r"[^a-zA-Z0-9]", "", token) for token in tokens if token]
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    return " ".join(tokens)


# Predict function with weightage for title and text
def predict_label(title, text, tfidf, model_title, XGB_model):
    text_weight = 0.9
    title_weight = 0.1
    final_label = "Real"  # Default assumption

    processed_text = process_text(text)
    X_text = tfidf.transform([processed_text]).toarray()
    label_text = XGB_model.predict(X_text)[0]

    # If title is given, use both title and text for prediction
    if title:
        processed_title = process_text(title)
        X_title = tfidf.transform([processed_title]).toarray()
        label_title = model_title.predict(X_title)[0]

        # Aggregate predictions with weights
        final_score = label_text * text_weight + label_title * title_weight
        final_label = "Fake" if final_score >= 0.5 else "Real"
    else:
        # If no title, rely solely on text prediction
        final_label = "Fake" if label_text == 1 else "Real"

    return final_label


# Streamlit UI with enhanced look
def main():
    st.markdown(
        "<h1 style='text-align: center; color: black;'>📰 Fake News Detector</h1>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("##")
        title_input = st.text_input("", placeholder="Enter the news title (optional)")
    with col2:
        st.write("##")
        text_input = st.text_area("", placeholder="Enter the news text", height=300)

    if st.button("Classify News", key="classify"):
        with st.spinner("Analyzing..."):
            label = predict_label(
                title_input, text_input, tfidf, model_title, XGB_model
            )
            if label == "Real":
                st.success("The news is Real 🟢")
            else:
                st.error("The news is Fake 🔴")


if __name__ == "__main__":
    main()
