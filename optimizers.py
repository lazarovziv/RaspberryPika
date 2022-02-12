import numpy as np


class SGD:
    def __init__(self, learning_rate=1.0, decay=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0

    def optimize(self, layer):
        layer.weights = layer.weights - self.learning_rate * layer.d_weights
        layer.biases = layer.biases - self.learning_rate * layer.d_biases

    def pre_optimize(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def post_optimize(self):
        self.iterations += 1
