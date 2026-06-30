
#importing the necessary modules/libraries


import numpy as np



#creating the neural network

class Network:
    # Defining the constructor
    def __init__(self, net_size: list):
        self.net_size: list = net_size
        self.layer_num: int = len(net_size)
        
        self.biases = np.load('parameters/biases.npy', allow_pickle=True)
        self.weights = np.load('parameters/weights.npy', allow_pickle=True)
        
    # The sigmoid/activatoin function
    def sigmoid(self, z: float) -> float:
        return 1/( 1 + np.exp(-z) )
    
    # The sigmoid prime function is the derivative of sigmoid function
    def sigmoid_prime(self, a: float) -> float:
      
        return a * ( 1 - a )
    
    # Defining the feedforward function which will also be used to train the certain data (later)
    def feedforward(self, a: np.ndarray) -> np.ndarray:
        
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
        
       
                    
    def predict(self, x):
        output = self.feedforward(x)
        return np.argmax(output)
                
        
