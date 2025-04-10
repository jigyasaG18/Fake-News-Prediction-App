import streamlit as st
import joblib
import re
from nltk.corpus import stopword
from nltk.stem import WordNetLemmatizer

# Load the trained model and vectorizer
model = joblib.load('random_forest_model.sav')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Initialize the WordNetLemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function for preprocessing the input text
def preprocess_input(content):
    # Remove non-alphabetic characters and convert to lower case
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower()
    content = content.split()
    
    # Apply lemmatization and remove stopwords
    content = [lemmatizer.lemmatize(word) for word in content if word not in stop_words]
    return ' '.join(content)

# Set a nice style for the app
st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="wide")

# Title for the web app
st.title("📰 Fake News Detection App")

# Input fields for the title, author, and content of the news article
st.header("Input Article Details")
title = st.text_input("**Enter the title of the news article**", placeholder="Title")
author = st.text_input("**Enter the author of the news article**", placeholder="Author")
content = st.text_area("**Enter the content of the news article**", placeholder="Content")

# Button to make predictions
if st.button("Check News", key="check_news"):
    # Merge author, title, and content for the prediction
    full_text = f"{author} {title} {content}"
    
    # Preprocess the combined text
    processed_text = preprocess_input(full_text)

    # Transform the input using the same fitted vectorizer
    vectorized_text = vectorizer.transform([processed_text])  # Make sure to pass a list

    # Make prediction using the model
    prediction = model.predict(vectorized_text)

    # Display the prediction result with custom colors
    if prediction[0] == 0:
        st.success("✨ **The news is Real!** 😃", icon="✅")
    else:
        st.error("🚫 **The news is Fake!** 😱", icon="❌")

# Sidebar for additional information about the app
st.sidebar.title("About the App")

# Create a selection box to view various information
app_info = st.sidebar.selectbox("Select Information", ["Overview", "How it Works", "Use Cases"])

# Display information based on the user's selection
if app_info == "Overview":
    st.sidebar.markdown("""
    This application leverages machine learning algorithms to detect the authenticity of news articles.
    By analyzing the text provided by the user, it concludes whether the news is real or fake.
    """)
elif app_info == "How it Works":
    st.sidebar.markdown("""
    The app processes user input by removing noise and applying text preprocessing techniques 
    such as lemmatization and stop word removal. The cleaned text is then vectorized using TF-IDF 
    and passed into a pre-trained model to make predictions.
    """)
elif app_info == "Use Cases":
    st.sidebar.markdown("""
    - **Journalists** can use the app to verify the authenticity of news articles.
    - **Students** can benefit from it while researching news sources for projects.
    - **General users** can check news articles to improve their media literacy.
    """)
