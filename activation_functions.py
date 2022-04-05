import numpy as np

class ReLU:

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, d_values):
        self.d_inputs = d_values.copy()
        # zero gradient where values are non-positive
        self.d_inputs[self.inputs <= 0] = 0


class Softmax:

    def forward(self, inputs):
        self.inputs = inputs
        # exponentiate input values
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalizing
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def backward(self, d_values):
        # initialize an array
        self.d_inputs = np.empty_like(d_values)
        for index, (output, d_value) in enumerate(zip(self.output, d_values)):
            # flatten output array
            output = output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(output) - np.dot(output, output.T)
            # calculate gradient
            self.d_inputs[index] = np.dot(jacobian_matrix, d_value)
