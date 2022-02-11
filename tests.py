import numpy as np
from activation_functions import Softmax, ReLU
from layers import Dense
from loss import CategoricalCrossentropy
from optimizers import SGD
import nnfs
from nnfs.datasets import spiral_data

X, y = spiral_data(samples=100, classes=3)

dense1 = Dense(2, 64)
activation1 = ReLU()
dense2 = Dense(64, 3)
activation2 = Softmax()
loss = CategoricalCrossentropy()

optimizer = SGD()

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
        print(f'loss: {loss_value.mean()}')
        print(f'accuracy: {accuracy}')

    # backpropagation
    loss.backward(activation2.output, y)
    activation2.backward(loss.d_inputs)
    dense2.backward(activation2.d_inputs)
    activation1.backward(dense2.d_inputs)
    dense1.backward(activation1.d_inputs)

    # update weights and biases
    optimizer.optimize(dense1)
    optimizer.optimize(dense2)