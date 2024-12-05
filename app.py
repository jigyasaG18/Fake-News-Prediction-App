# Button to make predictions
if st.button("Check News"):
    # Merge author, title, and content for the prediction
    full_text = f"{author} {title} {content}"
    
    # (Optional) Debug: Print the full text being analyzed
    print("Full Text for Prediction:", full_text)

    # Transform the input using the vectorizer
    vectorized_text = vectorizer.transform([full_text])
    
    # (Optional) Debug: Print the vectorized text
    print("Vectorized Text Shape:", vectorized_text.shape)
    
    # Make prediction using the model
    prediction = model.predict(vectorized_text)
    
    # (Optional) Debug: Print the actual prediction
    print("Prediction Output:", prediction)

    # Display the prediction result
    if prediction[0] == 0:
        st.success("The news is Real")
    else:
        st.error("The news is Fake")
