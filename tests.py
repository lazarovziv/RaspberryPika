import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
from activation_functions import Softmax, ReLU
from layers import Dense
from loss import CategoricalCrossentropy
from optimizers import *
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X, y = spiral_data(samples=100, classes=3)

colors_y = {0: 'blue', 1: 'red', 2: 'green'}

dense1 = Dense(2, 64)
activation1 = ReLU()
dense2 = Dense(64, 3)
activation2 = Softmax()
loss = CategoricalCrossentropy()

# optimizer = SGD(decay=1e-3, momentum=0.9)
# optimizer = AdaGrad(decay=1e-4)
# optimizer = RMSProp(learning_rate=0.02, decay=1e-5, rho=0.999)
optimizer = Adam(learning_rate=0.02, decay=1e-5)

loss_graph_values = []
accuracy_graph_values = []

for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    loss_value = loss.forward(activation2.output, y)

    predictions = np.argmax(activation2.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    if not epoch % 500:
        print(f'epoch: {epoch}')
        print(f'loss: {loss_value.mean():.3f}')
        print(f'accuracy: {accuracy:.3f}')
        print(f'lr: {optimizer.current_learning_rate}')
        print('---------------')

        # plotting accuracy and loss values over time
        """
        loss_graph_values.append(loss_value.mean())
        accuracy_graph_values.append(accuracy)

        plt.plot(loss_graph_values, label='Loss', color='blue')
        plt.plot(accuracy_graph_values, label='Accuracy', color='red')
        plt.xlabel('Epoch x 100')

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.pause(0.05)
        """

    # backpropagation
    loss.backward(activation2.output, y)
    activation2.backward(loss.d_inputs)
    dense2.backward(activation2.d_inputs)
    activation1.backward(dense2.d_inputs)
    dense1.backward(activation1.d_inputs)

    # update learning rate
    optimizer.pre_optimize()
    # update weights and biases
    optimizer.optimize(dense1)
    optimizer.optimize(dense2)
    # increment iterations
    optimizer.post_optimize()

# plt.show()
