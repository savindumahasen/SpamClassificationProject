from flask import Flask, request, jsonify, render_template
import pickle
import os

app = Flask(__name__)

# Load the model and the vectorizer from disk
model_path = os.path.join(os.path.dirname(__file__), 'spam_classifier.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), 'tfidf_vectorizer.pkl')

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(vectorizer_path, 'rb') as vectorizer_file:
    tf_vec = pickle.load(vectorizer_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data['message']
    message_vector = tf_vec.transform([message])
    prediction = model.predict(message_vector)[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
