import numpy as np
from layers import Dense
from loss import CategoricalCrossentropy, MSE
from activation_functions import ReLU, Softmax
import cv2 as cv

from models import Model

eyes_cascade = cv.CascadeClassifier('/Users/zivlazarov/Projects/RaspberryPika/eyes_recognition_model.xml')
face_cascade = cv.CascadeClassifier('/Users/zivlazarov/Projects/RaspberryPika/faces_recognition_model.xml')

# getting camera
cap = cv.VideoCapture(0)

if not cap.isOpened():
	print("Cannot open camera")
	exit()

# while camera is displaying
while True:
	check, frame = cap.read()
	# frame is a numpy array
	# shape (720, 1280, 3)
	print(frame.shape)

	# cant capture the camera
	if not check:
		print("Can't receive frame. Exiting...")
		break

	img_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	# gray_flip = cv.flip(gray, 1)
	# 2 windows "mirrored"
	# combined_window = np.hstack([gray, gray_flip])

	# Getting corners around the face
	faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)
   	# drawing bounding box around face
	for (x, y, w, h) in faces:
   	    frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

	# dogs = dog_cascade.detectMultiScale(img_gray, 1.3, 5)
	# for (xx, yy, ww, hh) in dogs:
	# 	frame = cv.rectangle(frame, (xx, yy), (xx + ww, yy + hh), (255, 0, 0), 3)

	'''
   	# detecting eyes
	eyes = eyes_cascade.detectMultiScale(imgGray)
   	# drawing bounding box for eyes
	for (ex, ey, ew, eh) in eyes:
   	    frame = cv.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 3)
	'''

	cv.imshow('frame', frame)

	# listening for exit key which is 'q'
	if cv.waitKey(1) == ord('q'):
		break

cap.release()
cv.destroyAllWindows()

frame = frame.reshape(720*1280, 3)

model = Model()

input_layer = Dense(frame.shape[1], 64)
activation0 = ReLU()
input_layer.add_activation_func(activation0)
model.add(input_layer)

dense1 = Dense(64, 64)
activation1 = ReLU()
dense1.add_activation_func(activation1)
model.add(dense1)

dense2 = Dense(64, 2)
activation2 = Softmax()
dense2.add_activation_func(activation2)
model.add(dense2)

loss = MSE()
model.compile(None, loss, None)
model.train(frame, np.array([0,1]))