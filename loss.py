import numpy as np
from activation_functions import Softmax


class Loss:

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        loss = np.mean(sample_losses)
        return loss


class CategoricalCrossentropy(Loss):

    def __init__(self):
        self.d_inputs = None

    def forward(self, y_pred, y_true):
        n_samples = len(y_pred)

        # clipping data to prevent division by 0
        # clipping lowest and highest values to prevent mean being dragged to a specific value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(n_samples), y_true]
        # elif len(y_true.shape) == 2:
        else:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        probabilities = -np.log(correct_confidences)
        return probabilities

    def backward(self, d_values, y_true):
        samples = len(d_values)
        labels = len(d_values[0])

        # turn labels into one-hot vectors
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # calculate gradient
        self.d_inputs = -y_true / d_values
        # normalize
        self.d_inputs = self.d_inputs / samples


class SoftmaxCategoricalCrossentropy(Loss):

    def __init__(self):
        self.d_inputs = None
        self.activation = Softmax()
        self.loss = CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output

        return self.loss.forward(self.output, y_true)

    def backward(self, d_values, y_true):
        samples = len(d_values)

        # if one-hot encoded, make them discrete
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.d_inputs = d_values.copy()
        # calculate gradient
        self.d_inputs[range(samples), y_true] -= 1
        # normalize it
        self.d_inputs = self.d_inputs / samples
