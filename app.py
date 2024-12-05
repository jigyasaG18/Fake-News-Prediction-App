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

# File uploader for CSV
uploaded_file = st.file_uploader("Choose a CSV file containing news articles", type="csv")
if uploaded_file is not None:
    # Load and process the dataset
    news_dataset = pd.read_csv(uploaded_file, encoding='unicode_escape')
  
    # Data Pre-processing
    news_dataset = news_dataset.fillna('')
    news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']
    
    # Stemming function
    port_stem = PorterStemmer()
    
    def stemming(content):
        stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
        stemmed_content = stemmed_content.lower()
        stemmed_content = stemmed_content.split()
        stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
        return ' '.join(stemmed_content)

    news_dataset['content'] = news_dataset['content'].apply(stemming)

    # Load the trained model and vectorizer
    model = joblib.load('fake_news_detection_model.sav')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')

    # Input for prediction
    st.subheader("News Content")
    if st.text_area("Enter news content for prediction:"):
        input_content = st.text_area("Enter news content:")
        input_content_stemmed = stemming(input_content)
        input_vector = vectorizer.transform([input_content_stemmed])

        # Make prediction
        prediction = model.predict(input_vector)

        if prediction[0] == 0:
            st.write('The news is Real')
        else:
            st.write('The news is Fake')
            
