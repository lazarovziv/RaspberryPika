import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
from activation_functions import Softmax, ReLU, Sigmoid
from layers import *
from loss import CategoricalCrossentropy, SoftmaxCategoricalCrossentropy, BinaryCrossEntropy
from optimizers import *
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X, y = spiral_data(samples=1000, classes=2)
y = y.reshape(-1, 1)

X_test, y_test = spiral_data(samples=100, classes=2)
y_test = y_test.reshape(-1, 1)


def forward_network(X, y, layers, activations, loss, optimizer, epochs=10000):
    dense1 = layers[0]
    activation1 = activations[0]
    dense2 = layers[1]
    activation2 = activations[1]

    loss_graph_values = []
    accuracy_graph_values = []

    for epoch in range(epochs+1):
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)
        data_loss = loss.calculate(activation2.output, y)
        regularization_loss = loss.regularization_loss(dense1) + loss.regularization_loss(dense2)
        loss_value = data_loss + regularization_loss

        predictions = (activation2.output > 0.5) * 1
        accuracy = np.mean(predictions == y)

        if not epoch % 100:
            print(f'epoch: {epoch}')
            print(f'loss: {loss_value.mean()}')
            print(f'data loss: {data_loss.mean()}')
            print(f'regularization loss: {regularization_loss.mean()}')
            print(f'accuracy: {accuracy:.2f}')
            print(f'lr: {optimizer.current_learning_rate}')
            print('---------------')

            # plotting accuracy and loss values over time
            loss_graph_values.append(loss_value.mean())
            accuracy_graph_values.append(accuracy)

            plt.plot(loss_graph_values, label='Loss', color='blue')
            plt.plot(accuracy_graph_values, label='Accuracy', color='red')
            plt.xlabel('Epoch x 100')

            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            plt.pause(0.05)

        # backpropagation (calculating gradients)
        loss.backward(activation2.output, y)
        activation2.backward(loss.d_inputs)
        dense2.backward(activation2.d_inputs)
        activation1.backward(dense2.d_inputs)
        dense1.backward(activation1.d_inputs)

        optimizer.pre_optimize()
        optimizer.optimize(dense1)
        optimizer.optimize(dense2)
        optimizer.post_optimize()

    plt.show()


def test_network(X, y, layers, activations, loss):
    dense1 = layers[0]
    activation1 = activations[0]
    dense2 = layers[1]
    activation2 = activations[1]

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    loss_value = loss.forward(activation2.output, y)

    predictions = (activation2.output > 0.5) * 1
    accuracy = np.mean(predictions == y)

    print(f'accuracy: {accuracy:.2f}')
    print('loss:', loss_value.mean())


layers = [Dense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4), Dense(64, 1)]
activations = [ReLU(), Sigmoid()]
optimizer = Adam(learning_rate=0.05, decay=5e-5)
loss = BinaryCrossEntropy()

forward_network(X, y, layers, activations, loss, optimizer)

test_network(X_test, y_test, layers, activations, loss)
