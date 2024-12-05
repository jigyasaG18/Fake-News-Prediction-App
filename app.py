import numpy as np
import pandas as pd
import re
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')

# Initialize the PorterStemmer
port_stem = PorterStemmer()

# Load the dataset
@st.cache(allow_output_mutation=True)
def load_data():
    news_dataset = pd.read_csv(r'C:\Users\Jigyasa\Desktop\Project\dataset\train.csv' , encoding='unicode_escape')
    news_dataset = news_dataset.fillna('')
    news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']
    return news_dataset

# Stemming function
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    return ' '.join(stemmed_content)

# Load and preprocess the data
news_dataset = load_data()
news_dataset['content'] = news_dataset['content'].apply(stemming)
X = news_dataset['content'].values
Y = news_dataset['label'].values

# Convert the textual data to numerical data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Splitting the dataset into training & test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Training the model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Streamlit UI
st.title("Fake News Classification")
st.subheader("Enter a News Article Below:")

input_text = st.text_area("Article Text", "")
if st.button("Predict"):
    if input_text:
        # Preprocess the input text
        input_stemmed = stemming(input_text)
        input_vectorized = vectorizer.transform([input_stemmed])
        
        # Make prediction
        prediction = model.predict(input_vectorized)
        
        # Display the result
        if prediction[0] == 0:
            st.write("The news is Real")
        else:
            st.write("The news is Fake")
    else:
        st.write("Please enter a news article.")