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


class AdaGrad:
    def __init__(self, learning_rate=1.0, decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def pre_optimize(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def optimize(self, layer):
        # if layer doesn't contain cache arrays, create with zeros
        if not hasattr(layer, 'weights_cache'):
            layer.weights_cache = np.zeros_like(layer.weights)
            # layer.weights_cache = np.full_like(shape=layer.weights.shape, fill_value=0.1)
            layer.biases_cache = np.zeros_like(layer.biases)
            # layer.biases_cache = np.full_like(shape=layer.biases.shape, fill_value=0.1)

        # update cache with squared current gradients
        layer.weights_cache = layer.weights_cache + np.power(layer.d_weights, 2)
        layer.biases_cache = layer.biases_cache + np.power(layer.d_biases, 2)

        # SGD update pattern and normalization with square rooted cache
        layer.weights = layer.weights + -self.current_learning_rate * layer.d_weights / (
                np.sqrt(layer.weights_cache) + self.epsilon)
        layer.biases = layer.biases + -self.current_learning_rate * layer.d_biases / (
                np.sqrt(layer.biases_cache) + self.epsilon)

    def post_optimize(self):
        self.iterations += 1


class RMSProp:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        # cache memory decay rate
        self.rho = rho

    def pre_optimize(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def optimize(self, layer):
        if not hasattr(layer, 'weights_cache'):
            layer.weights_cache = np.zeros_like(layer.weights)
            # layer.weights_cache = np.full(shape=layer.weights.shape, fill_value=0.1)
            layer.biases_cache = np.zeros_like(layer.biases)
            # layer.biases_cache = np.full(shape=layer.biases.shape, fill_value=0.1)

        layer.weights_cache = self.rho * layer.weights_cache + (1 - self.rho) * np.power(layer.d_weights, 2)
        layer.biases_cache = self.rho * layer.biases_cache + (1 - self.rho) * np.power(layer.d_biases, 2)

        layer.weights = layer.weights + -self.current_learning_rate * layer.d_weights / (np.sqrt(layer.weights_cache) + self.epsilon)
        layer.biases = layer.biases + -self.current_learning_rate * layer.d_biases / (np.sqrt(layer.biases_cache) + self.epsilon)

    def post_optimize(self):
        self.iterations += 1


class Adam:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.iterations = 0
        self.beta_1 = beta_1
        # similar to rho in RMSProp
        self.beta_2 = beta_2

    def pre_optimize(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def optimize(self, layer):
        if not hasattr(layer, 'weights_cache'):
            layer.weights_momentum = np.zeros_like(layer.weights)
            layer.weights_cache = np.zeros_like(layer.weights)
            # layer.weights_momentum = np.full(shape=layer.weights.shape, fill_value=0.1)
            # layer.weights_cache = np.full(shape=layer.weights.shape, fill_value=0.1)

            layer.biases_momentum = np.zeros_like(layer.biases)
            layer.biases_cache = np.zeros_like(layer.biases)
            # layer.biases_momentum = np.full(shape=layer.biases.shape, fill_value=0.1)
            # layer.biases_cache = np.full(shape=layer.biases.shape, fill_value=0.1)

        # update momentum with the gradients
        layer.weights_momentum = self.beta_1 * layer.weights_momentum + (1 - self.beta_1) * layer.d_weights
        layer.biases_momentum = self.beta_1 * layer.biases_momentum + (1 - self.beta_1) * layer.d_biases

        # corrected momentum
        weight_momentum_corrected = layer.weights_momentum / (1 - self.beta_1 ** (self.iterations + 1))  # prevent power by 0
        bias_momentum_corrected = layer.biases_momentum / (1 - self.beta_1 ** (self.iterations + 1))

        # update cache with squared current gradients
        layer.weights_cache = self.beta_2 * layer.weights_cache + (1 - self.beta_2) * np.power(layer.d_weights, 2)
        layer.biases_cache = self.beta_2 * layer.biases_cache + (1 - self.beta_2) * np.power(layer.d_biases, 2)

        # corrected cache
        weight_cache_corrected = layer.weights_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.biases_cache / (1 - self.beta_2 ** (self.iterations + 1))

        layer.weights = layer.weights + -self.current_learning_rate * weight_momentum_corrected / (
                    np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases = layer.biases + -self.current_learning_rate * bias_momentum_corrected / (
                    np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_optimize(self):
        self.iterations += 1
