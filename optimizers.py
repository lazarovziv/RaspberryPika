import numpy as np


class SGD:
    def __init__(self, learning_rate=1.0, decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_optimize(self):
        # updating learning rate
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def optimize(self, layer):
        # using momentum for weights and biases optimization
        if self.momentum:
            # if layer doesn't contain momentum attribute, create it as zero filled arrays (for weights and for biases)
            if not hasattr(layer, 'weights_momentum'):
                layer.weights_momentum = np.zeros_like(layer.weights)
                layer.biases_momentum = np.zeros_like(layer.biases)

            weights_updates = self.momentum * layer.weights_momentum - self.current_learning_rate * layer.d_weights
            layer.weights_momentum = weights_updates
            biases_updates = self.momentum * layer.biases_momentum - self.current_learning_rate * layer.d_biases
            layer.biases_momentum = biases_updates
        # "classic-no-momentum" SGD optimization
        else:
            weights_updates = -self.current_learning_rate * layer.d_weights
            biases_updates = -self.current_learning_rate * layer.d_biases

        # updating layer's weights and biases whether we used momentum or not
        layer.weights = layer.weights + weights_updates
        layer.biases = layer.biases + biases_updates

    def post_optimize(self):
        self.iterations += 1
