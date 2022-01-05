import numpy as np

class Dense:
    def __init__(self, n_inputs, n_neurons):
        # initialize weights and biases
        # setting shape as (n_inputs, n_neurons) to prevent transposing every dot product call
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) # using np.random.randn to set random weights between -1, 1 keeping the mean close to 0
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # calculating outputs from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases
