import numpy as np

class ReLU:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Softmax:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        # exponentiate input values
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalizing
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)