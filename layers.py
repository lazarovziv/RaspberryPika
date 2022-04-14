import numpy as np


class Dense:
    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        # initialize weights and biases
        # setting shape as (n_inputs, n_neurons) to prevent transposing every dot product call
        # super().__init__()
        self.output = None
        self.inputs = None
        self.d_inputs = None
        self.d_biases = None
        self.d_weights = None

        # initially multiplied by 0.01, but with glorot uniform it suited best to use 0.1
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)  # using np.random.randn to set random weights between -1, 1 keeping the mean close to 0
        self.biases = np.zeros((1, n_neurons))

        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs, training):
        self.inputs = inputs
        # calculating outputs from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, d_values):
        self.d_weights = np.dot(self.inputs.T, d_values)
        self.d_biases = np.sum(d_values, axis=0, keepdims=True)

        # gradients on regularization
        # l1 - weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            # setting the gradient values according to weights' positivity
            dL1[self.weights < 0] = -1
            self.d_weights = self.d_weights + self.weight_regularizer_l1 * dL1

        # l2 - weights
        if self.weight_regularizer_l2 > 0:
            self.d_weights = self.d_weights + 2 * self.weight_regularizer_l2 * self.weights

        # l1 - biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.d_biases = self.d_biases + self.bias_regularizer_l1 * dL1

        # l2 - biases
        if self.bias_regularizer_l2 > 0:
            self.d_biases = self.d_biases + 2 * self.bias_regularizer_l2 * self.biases

        self.d_inputs = np.dot(d_values, self.weights.T)


class Dropout:
    def __init__(self, rate):
        # rate parameter is like pytorch, percent to keep
        self.rate = 1 - rate

    def forward(self, inputs, training):
        self.inputs = inputs

        if not training:
            self.output = inputs.copy()
            return

        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate

        self.output = inputs * self.binary_mask

    def backward(self, d_values):
        self.d_inputs = d_values * self.binary_mask


class Input:

    def forward(self, inputs, training):
        self.output = inputs