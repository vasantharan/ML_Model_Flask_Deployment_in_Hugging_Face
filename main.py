from flask import Flask, request, jsonify
import pickle
import numpy as np
from sklearn.datasets import load_iris

targets = load_iris().target_names

app = Flask(__name__)

with open('model.pkl','rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return "Welcome to ML Model Flask App!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_data = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(input_data)
    output = targets[int(prediction[0])]
    return jsonify({
        'Predicted Class': output
    })

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 7860,debug=True)