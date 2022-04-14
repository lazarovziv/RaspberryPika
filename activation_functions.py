import numpy as np


class ReLU:

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, d_values):
        self.d_inputs = d_values.copy()
        # zero gradient where values are non-positive
        self.d_inputs[self.inputs <= 0] = 0

    def predictions(self, outputs):
        return outputs


class Softmax:

    def forward(self, inputs, training):
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

    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)


class Sigmoid:

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, d_values):
        self.d_inputs = d_values * (1 - self.output) * self.output

    def predictions(self, outputs):
        return (outputs > 0.5) * 1


class Linear:

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = inputs

    def backward(self, d_values):
        # derivative of linear function is 1
        self.d_inputs = d_values.copy()  # ( * 1 )

    def predictions(self, outputs):
        return outputs
