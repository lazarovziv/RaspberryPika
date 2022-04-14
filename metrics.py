import numpy as np


class Accuracy:

    def calculate(self, predictions, y):
        comparisons = self.compare(predictions, y)
        return np.mean(comparisons)


class RegressionAccuracy(Accuracy):

    def __init__(self):
        self.precision = None

    def init(self, y, reinit=False, decimal_accuracy=50):
        if self.precision is None or reinit:
            self.precision = np.std(y) / decimal_accuracy

    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision


class CategoricalAccuracy(Accuracy):

    def __init__(self, *, binary=False):
        # if classification is binary
        self.binary = binary

    # no initialization required
    def init(self, y):
        pass

    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y
