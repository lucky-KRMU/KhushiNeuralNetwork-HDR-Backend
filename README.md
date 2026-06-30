# KNN-HDR Backend: Khushi Neural Network - Handwritten Digit Recognizer

This is the backend server for the **KNN-HDR (Khushi Neural Network - Handwritten Digit Recognizer)** application. It is built using Python, Flask, and NumPy.

Rather than relying on high-level deep learning frameworks (like TensorFlow or PyTorch), this backend implements a **Feedforward Neural Network (Multi-Layer Perceptron) completely from scratch** using NumPy.

## 🛠️ Neural Network Architecture

The model is configured with a network topology of `[784, 50, 30, 10]`:

*   **Input Layer:** $784$ neurons (accepting a flattened $28 \times 28$ grayscale pixel grid).
*   **Hidden Layer 1:** $50$ neurons.
*   **Hidden Layer 2:** $30$ neurons.
*   **Output Layer:** $10$ neurons (representing the classification confidence for digits `0` to `9`).
*   **Activation Function:** Sigmoid activation $\sigma(z) = \frac{1}{1 + e^{-z}}$.
*   **Optimization:** Stochastic Gradient Descent (SGD) with Backpropagation.

The trained network weights and biases are saved as NumPy array binaries in the `parameters/` directory.

---

## 📂 Project Structure

```
backend/
├── parameters/
│   ├── biases.npy       # Saved bias vectors for the layers
│   └── weights.npy      # Saved weight matrices for the layers
├── .gitignore           # Git ignore configurations
├── model.py             # Neural Network implementation (SGD, Backpropagation)
└── server.py            # Flask API hosting the network endpoints
```


## 🚀 Getting Started

### Prerequisites

Make sure you have Python installed (version 3.8+ recommended).

### Install Dependencies

Install the required Python modules:

```bash
pip install flask flask-cors numpy
```

### Run the Server

Start the Flask development server:

```bash
python server.py
```

By default, the server runs on `http://127.0.0.1:5000` in debug mode.

---

## 🔌 API Endpoint

### `POST /predict`

Receives pixel inputs from the frontend and returns the predicted digit.

*   **Headers:** `Content-Type: application/json`
*   **Request Body:** A JSON object with a `pixels` key holding a flat $784$-element array of values representing pixel intensities $[0, 255]$.
    ```json
    {
      "pixels": [0, 0, 0, ..., 255, 128, 0, ..., 0]
    }
    ```
*   **Response Body:** A JSON object with the predicted digit.
    ```json
    {
      "digit": 3
    }
    ```
