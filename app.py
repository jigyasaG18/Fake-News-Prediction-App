import streamlit as st
import joblib

# Load the trained model and vectorizer
model = joblib.load('trained_model.sav')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Title for the web app
st.title("Fake News Detection")

# Input fields for the title, author, and content of the news article
title = st.text_input("Enter the title of the news article")
author = st.text_input("Enter the author of the news article")
content = st.text_area("Enter the content of the news article")

# Button to make predictions
if st.button("Check News"):
    # Merge author, title, and content for the prediction
    full_text = f"{author} {title} {content}"

    # Transform the input using the vectorizer
    vectorized_text = vectorizer.transform([full_text])

    # Make prediction using the model
    prediction = model.predict(vectorized_text)

    # Display the prediction result
 if prediction[0] == 0:
        st.success("The news is Real")
    else:
        st.error("The news is Fake")
