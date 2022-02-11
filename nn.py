import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data, vertical_data
from layers import Dense
from activation_functions import ReLU, Softmax
from loss import CategoricalCrossentropy

nnfs.init()

# X, y = spiral_data(samples=100, classes=3)

X, y = spiral_data(samples=100, classes=3)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
# plt.show()

dense1 = Dense(2, 3)
activation1 = ReLU()
dense2 = Dense(3, 3)
activation2 = Softmax()
loss = CategoricalCrossentropy()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
loss_value = loss.forward(activation2.output, y)

print(activation2.output[:5])
print(f'loss: {loss_value.mean()}')

predictions = np.argmax(activation2.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions == y)
print(f'accuracy: {accuracy}')

# backpropagation
loss.backward(activation2.output, y)
activation2.backward(loss.d_inputs)
dense2.backward(activation2.d_inputs)
activation1.backward(dense2.d_inputs)
dense1.backward(activation1.d_inputs)

print(dense1.d_weights)
print(dense1.d_biases)
print(dense2.d_weights)
print(dense2.d_biases)
