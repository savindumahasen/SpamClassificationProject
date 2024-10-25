from flask import Flask, request, jsonify, render_template
import pickle
import os
from helper import vectorizer,get_prediction, preprocessing

app = Flask(__name__)

# Load the model and the vectorizer from disk
model_path = os.path.join(os.path.dirname(__file__), 'spam_classifier.pkl')

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data['message']
    print(f"Received message: {message}")  # Log the received message
    preprocessed_text = preprocessing(message);
    print(f'preprocessed_text: {preprocessed_text}')
    message_vector = vectorizer([message])
    predictions = get_prediction(message_vector)
    #prediction = model.predict(message_vector)[0]
    print(f"Prediction: {predictions}")  # Log the prediction result

    # Map prediction result to label
    prediction_label = 'spam' if predictions == 'Spam'else 'Ham'
    
    return jsonify({'prediction': prediction_label})

if __name__ == '__main__':
    app.run(debug=True)
