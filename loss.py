import numpy as np
from activation_functions import Softmax


class Loss:

    def __init__(self):
        self.accumulated_count = 0
        self.accumulated_sum = 0
        self.trainable_layers = None

    def set_trainable_layers(self, layers):
        self.trainable_layers = layers

    def calculate(self, output, y, *, include_regularization=False):
        sample_losses = self.forward(output, y)
        loss = np.mean(sample_losses)

        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        if not include_regularization:
            return loss
        return loss, self.regularization_loss()

    def calculate_accumulated(self, *, include_regularization=False):
        data_loss = self.accumulated_sum / self.accumulated_count

        if not include_regularization:
            return data_loss

        return data_loss, self.regularization_loss()

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def regularization_loss(self):
        regularization_loss = 0  # default value

        for layer in self.trainable_layers:
            # l1 - weights
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

            # l2 - weights
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(np.power(layer.weights, 2))

            # l1 - biases
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

            # l2 - weights
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(np.power(layer.biases, 2))

        return regularization_loss


class CategoricalCrossentropy(Loss):

    def __init__(self):
        super().__init__()
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

    # def __init__(self):
    #     self.d_inputs = None
    #     self.activation = Softmax()
    #     self.loss = CategoricalCrossentropy()
    #
    # def forward(self, inputs, y_true):
    #     self.activation.forward(inputs)
    #     self.output = self.activation.output
    #
    #     return self.loss.forward(self.output, y_true)

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


class BinaryCrossEntropy(Loss):

    def __init__(self):
        super().__init__()
        self.d_inputs = None

    def forward(self, y_pred, y_true):
        # clipping to prevent division by zero
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        return np.mean(sample_losses, axis=1)

    def backward(self, d_values, y_true):
        samples = len(d_values)
        outputs = len(d_values[0])

        # clipping to prevent division by zero
        clipped_values = np.clip(d_values, 1e-7, 1 - 1e-7)
        # gradients
        self.d_inputs = -(y_true / clipped_values - (1 - y_true) / (1 - clipped_values)) / outputs
        # normalize gradient
        self.d_inputs = self.d_inputs / samples


class MSE(Loss):

    def __init__(self):
        super().__init__()
        self.d_inputs = None

    def forward(self, y_pred, y_true):
        return np.mean(np.power(y_true - y_pred, 2), axis=-1)

    def backward(self, d_values, y_true):
        samples = len(d_values)
        outputs = len(d_values[0])

        # gradients
        self.d_inputs = -2 * (y_true - d_values) / outputs
        # normalize gradients
        self.d_inputs = self.d_inputs / samples


class MAE(Loss):

    def __init__(self):
        super().__init__()
        self.d_inputs = None

    def forward(self, y_pred, y_true):
        return np.mean(np.abs(y_true - y_pred), axis=-1)

    def backward(self, d_values, y_true):
        samples = len(d_values)
        outputs = len(d_values[0])

        # gradients (sign() return positivity of the parameter)
        self.d_inputs = np.sign(y_true - d_values) / outputs
        # normalize gradients
        self.d_inputs = self.d_inputs / samples
