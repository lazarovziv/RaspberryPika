import numpy as np

class Layer:
    def __init__(self):
        self.activation_func = None

    def calculate(self, inputs):
        return self.forward(inputs)

    def add_activation_func(self, activation_func):
        self.activation_func = activation_func

class Dense(Layer):
    def __init__(self, n_inputs, n_neurons):
        # initialize weights and biases
        # setting shape as (n_inputs, n_neurons) to prevent transposing every dot product call
        super().__init__()
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) # using np.random.randn to set random weights between -1, 1 keeping the mean close to 0
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # calculating outputs from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases
