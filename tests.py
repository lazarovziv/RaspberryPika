import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
from activation_functions import *
from layers import *
from loss import *
from models import *
from optimizers import *
from metrics import *
import nnfs
from nnfs.datasets import spiral_data, sine_data

nnfs.init()

#
# def forward_network(X, y, layers, activations, loss, optimizer, epochs=10000):
#     dense1 = layers[0]
#     activation1 = activations[0]
#     dense2 = layers[1]
#     activation2 = activations[1]
#     dense3 = layers[2]
#     activation3 = activations[2]
#
#     loss_graph_values = []
#     accuracy_graph_values = []
#
#     for epoch in range(epochs+1):
#         dense1.forward(X)
#         activation1.forward(dense1.output)
#         dense2.forward(activation1.output)
#         activation2.forward(dense2.output)
#         dense3.forward(activation2.output)
#         activation3.forward(dense3.output)
#         data_loss = loss.calculate(activation3.output, y)
#         regularization_loss = loss.regularization_loss(dense1) + loss.regularization_loss(dense2) + loss.regularization_loss(dense3)
#         loss_value = data_loss + regularization_loss
#
#         predictions = activation3.output
#         accuracy = np.mean(np.absolute(predictions - y) < accuracy_precision)
#
#         if not epoch % 100:
#             print(f'epoch: {epoch}')
#             print(f'loss: {loss_value.mean()}')
#             print(f'data loss: {data_loss.mean()}')
#             print(f'regularization loss: {regularization_loss}')
#             print(f'accuracy: {accuracy:.3f}')
#             print(f'lr: {optimizer.current_learning_rate}')
#             print('---------------')
#
#             # plotting accuracy and loss values over time
#             loss_graph_values.append(loss_value.mean())
#             accuracy_graph_values.append(accuracy)
#
#             plt.plot(loss_graph_values, label='Loss', color='blue')
#             plt.plot(accuracy_graph_values, label='Accuracy', color='red')
#             plt.xlabel('Epoch x 100')
#
#             handles, labels = plt.gca().get_legend_handles_labels()
#             by_label = dict(zip(labels, handles))
#             plt.legend(by_label.values(), by_label.keys())
#             plt.pause(0.05)
#
#         # backpropagation (calculating gradients)
#         loss.backward(activation3.output, y)
#         activation3.backward(loss.d_inputs)
#         dense3.backward(activation3.d_inputs)
#         activation2.backward(dense3.d_inputs)
#         dense2.backward(activation2.d_inputs)
#         activation1.backward(dense2.d_inputs)
#         dense1.backward(activation1.d_inputs)
#
#         optimizer.pre_optimize()
#         optimizer.optimize(dense1)
#         optimizer.optimize(dense2)
#         optimizer.optimize(dense3)
#         optimizer.post_optimize()
#
#     plt.show()


X, y = spiral_data(samples=1000, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

model = Model()

model.add(Dense(2, 512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(Dropout(0.1))
model.add(ReLU())
model.add(Dense(512, 3))
model.add(Softmax())

model.compile(
    loss=CategoricalCrossentropy(),
    optimizer=Adam(learning_rate=0.05, decay=5e-5),
    accuracy=CategoricalAccuracy()
)

model.fit(X, y, epochs=10000, print_every=100, validation_data=(X_test, y_test))
