import streamlit as st
import pickle

# Load the model and vectorizer
with open('spam_classifier.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tf_vec = pickle.load(vectorizer_file)

def predict_message(message):
    # Transform the input message using the loaded vectorizer
    message_vec = tf_vec.transform([message]).toarray()
    # Predict the label using the loaded model
    prediction = model.predict(message_vec)
    # Return 'ham' or 'spam'
    return 'ham' if prediction[0] == 0 else 'spam'

# Streamlit UI
st.title("Spam Classifier")
st.write("Enter a message to classify it as 'ham' or 'spam'.")

message = st.text_area("Message")

if st.button("Classify"):
    if message:
        prediction = predict_message(message)
        st.write(f"The message is classified as: **{prediction}**")
    else:
        st.write("Please enter a message to classify.")
