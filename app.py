import streamlit as st
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import nltk

nltk.download('stopwords')

# Title of the app
st.title('Fake News Detection App')

# Load the trained model and vectorizer
model = joblib.load('fake_news_detection_model.sav')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Stemming function
port_stem = PorterStemmer()

def stemming(content):
    # Pre-process the content
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    return ' '.join(stemmed_content)

# Input for prediction
st.subheader("Paste your news article below:")
input_content = st.text_area("Enter news content:")

if st.button("Predict"):
    if input_content:
        input_content_stemmed = stemming(input_content)
        input_vector = vectorizer.transform([input_content_stemmed])

        # Make prediction
        prediction = model.predict(input_vector)

        if prediction[0] == 0:
            st.success('The news is Real')
        else:
            st.error('The news is Fake')
    else:
        st.warning("Please enter some content in the text box.")
        
