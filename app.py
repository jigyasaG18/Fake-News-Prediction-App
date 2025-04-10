import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Function to download required nltk datasets
def download_nltk_data():
    try:
        stopwords.words('english')  # Attempt to load stopwords
    except LookupError:
        nltk.download('stopwords')

    try:
        WordNetLemmatizer()  # Attempt to create a lemmatizer
    except LookupError:
        nltk.download('wordnet')

# Download NLTK data
download_nltk_data()

# Initialize stop words and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load the trained model and vectorizer (make sure these files exist)
model = joblib.load('random_forest_model.sav')  # Update with your model file path
vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Update with your vectorizer file path

# Function to preprocess the input text
def preprocess_input(content):
    # Remove non-alphabetic characters and convert to lower case
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower()
    content = content.split()
    
    # Apply lemmatization and remove stopwords
    content = [lemmatizer.lemmatize(word) for word in content if word not in stop_words]
    return ' '.join(content)

# Title of the Streamlit app
st.title("Fake News Prediction")

# User input for news article
full_text = st.text_area("Enter the news article text here:")

# Button to make prediction
if st.button("Predict"):
    if full_text:
        # Preprocess the input text
        processed_text = preprocess_input(full_text)
        
        # Vectorize the processed text
        X = vectorizer.transform([processed_text])
        
        # Make the prediction
        prediction = model.predict(X)
        
        # Display the prediction result
        if prediction[0] == 1:
            st.write("The news article is predicted to be **FAKE**.")
        else:
            st.write("The news article is predicted to be **REAL**.")
    else:
        st.write("Please enter some text to analyze.")
