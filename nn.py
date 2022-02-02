import numpy as np
import matplotlib.pyplot as plt
import nnfs

nnfs.init()

from nnfs.datasets import spiral_data
from layers import Dense
from activation_functions import ReLU, Softmax
from loss import CategoricalCrossentropy, MSE

X, y = spiral_data(samples=100, classes=3)

print(X.shape)
print(y.shape)

dense1 = Dense(2, 3)
activation1 = ReLU()
dense2 = Dense(3, 3)
activation2 = Softmax()
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
print(activation2.output[:5])
loss = CategoricalCrossentropy()
loss_value = loss.calculate(activation2.output, y)
print('Categorical Crossentropy Loss:', loss_value)
mse_loss = MSE()
mse_loss_value = mse_loss.calculate(activation2.output, y)
print('MSE Loss:', loss_value)