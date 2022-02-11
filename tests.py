import numpy as np
from activation_functions import Softmax
from loss import CategoricalCrossentropy

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

class_targets = np.array([0, 1, 1])

softmax = Softmax()
loss = CategoricalCrossentropy()
softmax.output = softmax_outputs
# backpropagate with softmax output
loss.backward(softmax.output, class_targets)
# backpropagate the softmax function
softmax.backward(loss.d_inputs)
print(softmax.d_inputs)
