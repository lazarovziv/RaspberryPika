import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
from activation_functions import Softmax, ReLU
from layers import *
from loss import CategoricalCrossentropy, SoftmaxCategoricalCrossentropy
from optimizers import *
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X, y = spiral_data(samples=1000, classes=3)

colors_y = {0: 'blue', 1: 'red', 2: 'green'}

dense1 = Dense(2, 512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
activation1 = ReLU()
dropout1 = Dropout(0.1)
dense2 = Dense(512, 3)
loss_activation = SoftmaxCategoricalCrossentropy()

# optimizer = SGD(decay=1e-3, momentum=0.9)
# optimizer = AdaGrad(decay=1e-4)
# optimizer = RMSProp(learning_rate=0.02, decay=1e-5, rho=0.999)
optimizer = Adam(learning_rate=0.02, decay=5e-7)

loss_graph_values = []
accuracy_graph_values = []

EPOCHS = 10000

for epoch in range(EPOCHS+1):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dropout1.forward(activation1.output)
    dense2.forward(dropout1.output)
    data_loss = loss_activation.forward(dense2.output, y)
    regularization_loss = loss_activation.loss.regularization_loss(dense1) + loss_activation.loss.regularization_loss(dense2)
    loss = data_loss + regularization_loss

    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(f'epoch: {epoch}')
        print(f'loss: {loss.mean()}')
        print(f'data loss: {data_loss.mean()}')
        print(f'regularization loss: {regularization_loss}')
        print(f'accuracy: {accuracy:.2f}')
        print(f'lr: {optimizer.current_learning_rate}')
        print('---------------')

        # plotting accuracy and loss values over time
        loss_graph_values.append(data_loss.mean())
        accuracy_graph_values.append(accuracy)

        plt.plot(loss_graph_values, label='Loss', color='blue')
        plt.plot(accuracy_graph_values, label='Accuracy', color='red')
        plt.xlabel('Epoch x 100')

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.pause(0.05)

    # backpropagation (calculating gradients)
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.d_inputs)
    dropout1.backward(dense2.d_inputs)
    activation1.backward(dropout1.d_inputs)
    dense1.backward(activation1.d_inputs)

    # update learning rate
    optimizer.pre_optimize()
    # update weights and biases
    optimizer.optimize(dense1)
    optimizer.optimize(dense2)
    # increment iterations
    optimizer.post_optimize()

plt.show()


X_test, y_test = spiral_data(samples=100, classes=3)

dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y_test)

predictions_test = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions_test == y_test)

print(f'accuracy: {accuracy:.2f}')
print('loss:', loss.mean())