from flask import Flask,jsonify,request
from model import Network
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)


NeuralNetwork = Network([784,50,30,10])


@app.route('/predict', methods=['POST'])
def take_input():
    data = request.get_json()
    
    if not data or 'pixels' not in data:
        return jsonify({"error": "No Pixel data provided"}), 400
    
    pixel_input = data['pixels']
    pixel_array = np.array(pixel_input).reshape(784,1)
    
    network_output = NeuralNetwork.predict(pixel_array)
    predicted_digit = int(network_output)
    
    return jsonify({"digit": predicted_digit})
    

if __name__ == "__main__":
    app.run()