import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
from activation_functions import *
from layers import *
from loss import *
from models import *
from optimizers import *
from metrics import *

from zipfile import ZipFile
import os
import urllib
import urllib.request

import cv2


#
# URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
# FILE = 'fashion_mnist_images.zip'
# FOLDER = 'fashion_mnist_images'
#
# if not os.path.isfile(FILE):
#     print(f'Downloading {URL} as {FILE}...')
#     urllib.request.urlretrieve(URL, FILE)
#
# print('Unzipping images...')
# with ZipFile(FILE) as zip_images:
#     zip_images.extractall(FOLDER)
#
# print('Done!')

def load_mnist_dataset(dataset, path):
    labels = os.listdir(os.path.join(path, dataset))

    X = []
    y = []

    for label in labels:
        for file in os.listdir(os.path.join(path, dataset, label)):
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)

            X.append(image)
            y.append(label)

    return np.array(X), np.array(y).astype('uint8')


def create_data_mnist(path):
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)

    return X, y, X_test, y_test


classes = {0: 'T-Shirt',
           1: 'Trousers',
           2: 'Pullover',
           3: 'Dress',
           4: 'Coat',
           5: 'Sandal',
           6: 'Shirt',
           7: 'Sneaker',
           8: 'Bag',
           9: 'Ankle Boot'}

X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')

X = (X.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5

X = X.reshape(X.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# shuffle the data
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)

X = X[keys]
y = y[keys]

model = Model()

model.add(Dense(X.shape[1], 128))
model.add(ReLU())
model.add(Dense(128, 128))
model.add(ReLU())
model.add(Dense(128, 10))
model.add(Softmax())

model.compile(
    loss=CategoricalCrossentropy(),
    optimizer=Adam(decay=1e-3),
    accuracy=CategoricalAccuracy()
)

model.fit(X, y, validation_data=(X_test, y_test), epochs=10, batch_size=128, print_every=100)
