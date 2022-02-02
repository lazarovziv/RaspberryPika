import layers

class Model:
    def __init__(self):
        self.num_layers = 0
        self.layers = []
        self.optimizer = None
        self.calc_metrics = []
        self.loss_func = None

    def compile(self, optimizer, loss_func, metrics):
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.calc_metrics = metrics

    def add(self, layer):
        if isinstance(layer, layers.Layer):
            self.layers.append(layer)
        else:
            raise RuntimeError('Parameter not of type Layer.')

    def train(self, X, y):
        for (idx, layer) in enumerate(self.layers):
            if idx == 0:
                layer.calculate(X)
                layer.activation_func.forward(layer.output)
            else:
                layer.calculate(self.layers[idx-1].output)
                layer.activation_func.forward(layer.output)

            self.loss_func.calculate(self.layers[-1].activation_func.output, y)
            print(self.loss_func.output)
