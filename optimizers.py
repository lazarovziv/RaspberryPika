import numpy as np


class SGD:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    def optimize(self, layer):
        layer.weights = layer.weights - self.learning_rate * layer.d_weights
        layer.biases = layer.biases - self.learning_rate * layer.d_biases
