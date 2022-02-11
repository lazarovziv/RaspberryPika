import numpy as np


class CategoricalCrossentropy:
    def __init__(self):
        self.d_inputs = None

    def forward(self, y_pred, y_true):
        global correct_confidences
        n_samples = len(y_pred)

        # clipping data to prevent division by 0
        # clipping lowest and highest values to prevent mean being dragged to a specific value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(n_samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        return -np.log(correct_confidences)

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
