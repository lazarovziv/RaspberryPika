import numpy as np


class Dense:
    def __init__(self, n_inputs, n_neurons):
        # initialize weights and biases
        # setting shape as (n_inputs, n_neurons) to prevent transposing every dot product call
        # super().__init__()
        self.output = None
        self.inputs = None
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)  # using np.random.randn to set random weights between -1, 1 keeping the mean close to 0
        self.biases = np.zeros((1, n_neurons))

        self.d_inputs = None
        self.d_biases = None
        self.d_weights = None

    def forward(self, inputs):
        self.inputs = inputs
        # calculating outputs from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, d_values):
        self.d_weights = np.dot(self.inputs.T, d_values)
        self.d_biases = np.sum(d_values, axis=0, keepdims=True)
        self.d_inputs = np.dot(d_values, self.weights.T)
