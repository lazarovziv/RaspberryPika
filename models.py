import numpy as np
from layers import Input
from activation_functions import Softmax
from loss import CategoricalCrossentropy, SoftmaxCategoricalCrossentropy


class Model:

    def __init__(self):
        self.layers = []
        self.softmax_classifier_output = None
        self.accuracy_graph = {}
        self.loss_graph = {}

    def add(self, layer):
        self.layers.append(layer)

    def fit(self, X, y, *, epochs=1, print_every=1, validation_data=None):
        # init accuracy
        self.accuracy.init(y)

        for epoch in range(epochs + 1):
            output = self.predict(X, training=True)

            data_loss, regularization_loss = self.loss.calculate(output, y, include_regularization=True)
            loss = data_loss + regularization_loss

            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)

            self.backpropagate(output, y)

            self.optimizer.pre_optimize()
            for layer in self.trainable_layers:
                self.optimizer.optimize(layer)
            self.optimizer.post_optimize()

            if not epoch % print_every:
                print(f'epoch: {epoch}')
                print(f'loss: {loss:.3f}')
                print(f'data loss: {data_loss:.3f}')
                print(f'regularization loss: {regularization_loss:.3f}')
                print(f'accuracy: {accuracy:.3f}')
                print(f'lr: {self.optimizer.current_learning_rate}')
                print('---------------')

                # add values to metrics' graphs for plotting
                self.accuracy_graph[epoch] = accuracy
                self.loss_graph[epoch] = loss

        if validation_data is not None:
            X_val, y_val = validation_data

            output = self.predict(X_val, training=False)

            loss = self.loss.calculate(output, y_val)

            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y_val)

            print(f'validation: ')
            print(f'accuracy: {accuracy:.3f}')
            print(f'loss: {loss:.3f}')

    # needs to be called before fit() and after add()
    def compile(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

        self.input_layer = Input()

        self.trainable_layers = []

        for i in range(len(self.layers)):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]

            elif i < len(self.layers) - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]

            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        self.loss.set_trainable_layers(self.trainable_layers)

        if isinstance(self.layers[-1], Softmax) and isinstance(self.loss, CategoricalCrossentropy):
            self.softmax_classifier_output = SoftmaxCategoricalCrossentropy()

    # needs to be called after fit()
    def predict(self, X, training):
        self.input_layer.forward(X, training)

        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        # returning output of last layer (which is the last activation function)
        return self.layers[-1].output

    def backpropagate(self, output, y):
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)
            self.layers[-1].d_inputs = self.softmax_classifier_output.d_inputs

            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.d_inputs)

            return

        self.loss.backward(output, y)
        for layer in reversed(self.layers):
            layer.backward(layer.next.d_inputs)