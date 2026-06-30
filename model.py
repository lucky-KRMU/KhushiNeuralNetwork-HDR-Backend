
#importing the necessary modules/libraries


import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


#creating the neural network

class Network:
    # Defining the constructor
    def __init__(self, net_size: list):
        self.net_size: list = net_size
        self.layer_num: int = len(net_size)
        
        self.biases = [ np.random.randn(y, 1) for y in (net_size[1:]) ] # Defining the biases of the entire network\
        self.weights = [ np.random.randn(y,x) / np.sqrt(x) for y,x in zip(net_size[ 1: ], net_size[ :-1 ]) ] # Definging the weights for each layer of the entire network
        
    # The sigmoid/activatoin function
    def sigmoid(self, z: float) -> float:
        return 1/( 1 + np.exp(-z) )
    
    # The sigmoid prime function is the derivative of sigmoid function
    def sigmoid_prime(self, a: float) -> float:
        '''
        a is already the final activation(sigmoid) 
        '''
        
        return a * ( 1 - a )
    
    # Defining the feedforward function which will also be used to train the certain data (later)
    def feedforward(self, a: np.ndarray) -> np.ndarray:
        '''
        For this specific network, we know that:
        the weight matricies are [784,50] and [50,10] (dimensions/sizes)
        and the bias matrices are [50,1] and [10,1] (dimensions/sizez)
        1. The first layer(hidden layer) consists of 50 neurons and each neuron has a single bias and 784 weights.
        2. The second layer(output layer) consists of 10 neurons and each neuron has a single bias and 50 weights.
        total parameters of this Neural Network would turn out ot be 784x50 + 50 + 50x10 + 10 = 39,760
        Hence, this model would have 39,760 parameters.
        '''
        a = a/255.0 # -> this is used for normalizing the value of each pixel. pixel activation lies in [0,255]
        self.io_layer = [a] # This is actually the list that would consists of all the input/output of the network for each layer.
       
        for i in range(self.layer_num - 1):
            w = self.weights[i]
            b = self.biases[i]
            x = self.io_layer[i]
            z = w @ x + b
            a = self.sigmoid(z)
            self.io_layer.append(a)
        return a
    
    def cost(self, a:float, y: int) -> float:
        return .5 * ( a - y ) ** 2
    
    def vectorize(self, label: int) -> list:
        vec = np.zeros((10,1))
        vec[label] = 1
        return vec
    
    
        
    
    def train(self, training_data: list, lr: float = .01, epochs: int = 20):
        
        # necessary variables for SGD
        batch_size = 64
        if (len(training_data)%batch_size == 0):
            training_data_size = len(training_data)//batch_size 
        else:
            training_data_size = len(training_data)//batch_size + 1
        
        # This is the loop for back propagation (investigating the weights)
        
        # This is for plotting the graph
        self.accuracy_percentage = []
        
        
        for i in range(epochs):
            
            # applying SGD (Stochastic gradient descent)
            np.random.shuffle(training_data)
            
            
            
            old_index = 0
            
            correct = 0
            total = 0
            
            for j in range(training_data_size):
                SGD_batch = training_data[old_index:old_index+batch_size]
                old_index += batch_size
            
                
                
                layer_error_gradient = [ np.zeros_like(x) for x in self.weights ] # This is for storing the values of updated gradients of weights
                layer_bias_gradient = [ np.zeros_like(x) for x in self.biases ] # This is for storing the values of updated gradients of biases
            
                for data in SGD_batch:
                    
                    y = data[0]
                    train_inputs = np.array(data[1:]).reshape(784,1)
                    # print(train_inputs.shape) 
                    
                    # running the feedforward loop
                    self.feedforward(train_inputs)
                    
                    if self.predict(train_inputs) == y:
                        correct += 1
                    total += 1
                    
                    
                    # Back propagation code strats from here
                    error_gradient = ( self.io_layer[self.layer_num - 1] - self.vectorize(int(y)) ) * self.sigmoid_prime(self.io_layer[self.layer_num - 1])
                    for k in range(self.layer_num - 1, 0, -1):
                        error_weight_gradient  = error_gradient @ self.io_layer[k-1].T
                        
                        layer_error_gradient[k-1] += (error_weight_gradient)
                        layer_bias_gradient[k-1] += (error_gradient)
                        
                        # updating the error gradient
                        error_gradient = self.weights[k-1].T @ error_gradient * self.sigmoid_prime(self.io_layer[k -1])
                        
                        
                        
                # Gradient Descent starts here
                for k in range(self.layer_num-1, 0, -1):
                    self.weights[k-1] -= lr * ((layer_error_gradient[k-1])/len(SGD_batch))
                    self.biases[k-1] -= lr * ((layer_bias_gradient[k-1])/len(SGD_batch))
                    
            # adding the accuracy meter
            print(f"Accuracy of {i+1} epoch: ", correct/total*100, " %")
            
            self.accuracy_percentage.append((correct/total*100))
        
        self.epoch_range = [x+1 for x in range(epochs)]
        
        # Actual Saving the graph
        plt.style.use("dark_background")
        plt.plot(self.epoch_range, self.accuracy_percentage, marker='o', color='cyan', linewidth=3, markersize=8, label='Accuracy')
        plt.xlabel("epochs")
        plt.ylabel("accuracy (%)")
        plt.title("KNN - MNIST Training Accuracy", color="red", fontsize=18)
        plt.grid(True, linestyle=":")
        plt.legend()
        plt.savefig("graphs/accuracy.png")
        plt.show()
                    
    def predict(self, x):
        output = self.feedforward(x)
        return np.argmax(output)
                
        

# Loading the MNIST Dataset for the sake of testing a single prediction out of the network (without training)
test_data = np.loadtxt(
    "data/train.csv",
    delimiter=',',
    skiprows=1
)


N = Network([784, 50, 30, 10])
N.train(test_data[1:], epochs=50, lr=.5)

for i in range(10):
    output = N.predict(test_data[i][1:].reshape(784,1))
    print(output)

# storing the weights and biases in a text file
np.save("parameters/weights.npy", np.array(N.weights, dtype=object))
np.save("parameters/biases.npy", np.array(N.biases, dtype=object))
    