from flask import Flask,jsonify
from model import Network
import numpy as np

app = Flask(__name__)

pixel_array = np.zeros((784,1))

NeuralNetwork = Network([784,50,30,10])


@app.route('/', methods=['GET'])
def take_input(pixel_input):
    global pixel_array
    pixel_array = np.array(pixel_input)
    predicted_digit = {
        "digit": NeuralNetwork.predict(pixel_array)
    }
    return jsonify({predicted_digit})
    
    

if __name__ == "__main__":
    app.run(debug=True)