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

    def fit(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None):
        # init accuracy
        self.accuracy.init(y)

        train_steps = 1

        if validation_data is not None:
            validation_steps = 1
            X_val, y_val = validation_data

        if batch_size is not None:
            train_steps = len(X) // batch_size

            if train_steps * batch_size < len(X):
                train_steps += 1

            if validation_data is not None:
                validation_steps = len(X_val) // batch_size

                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1

        # training loop
        for epoch in range(epochs + 1):
            print(f'epoch: {epoch+1}')
            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(train_steps):
                # if batch_size wasn't specified, setting batch_size as entire dataset
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step * batch_size:(step+1) * batch_size]
                    batch_y = y[step * batch_size:(step+1) * batch_size]

                output = self.predict(batch_X, training=True)

                data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss

                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                self.backpropagate(output, batch_y)

                self.optimizer.pre_optimize()
                for layer in self.trainable_layers:
                    self.optimizer.optimize(layer)
                self.optimizer.post_optimize()

                if not epoch % print_every or step == train_steps - 1:
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

            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            print('Training: ')
            print(f'epoch accuracy: {epoch_accuracy:.3f}')
            print(f'epoch loss: {epoch_loss:.3f}')
            print(f'epoch data loss: {epoch_data_loss:.3f}')
            print(f'epoch regularization loss: {epoch_regularization_loss:.3f}')
            print(f'learning rate: {self.optimizer.current_learning_rate:.3f}')

            if validation_data is not None:
                self.loss.new_pass()
                self.accuracy.new_pass()

                for step in range(validation_steps):
                    if batch_size is None:
                        batch_X = X
                        batch_y = y
                    else:
                        batch_X = X_val[step * batch_size:(step+1) * batch_size]
                        batch_y = y_val[step * batch_size:(step+1) * batch_size]

                    output = self.predict(batch_X, training=False)

                    self.loss.calculate(output, batch_y)

                    predictions = self.output_layer_activation.predictions(output)
                    self.accuracy.calculate(predictions, batch_y)

                validation_loss = self.loss.calculate_accumulated()
                validation_accuracy = self.accuracy.calculate_accumulated()

                print(f'\nValidation: ')
                print(f'accuracy: {validation_accuracy:.3f}')
                print(f'loss: {validation_loss:.3f}')

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