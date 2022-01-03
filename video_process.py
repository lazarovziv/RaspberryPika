import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential

import cv2 as cv

face_cascade = cv.CascadeClassifier(cv.data.harrcascade + 'haarcascade_frontalface_default.xml')

# getting camera
cap = cv.VideoCapture(0)

if not cap.isOpened():
	print("Cannot open camera")
	exit()

# while camera is displaying
while True:
	check, frame = cap.read()

	# cant capture the camera
	if not check:
		print("Can't receive frame. Exiting...")
		break

	gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	gray_flip = cv.flip(gray, 1)
	# 2 windows "mirrored"
	# combined_window = np.hstack([gray, gray_flip])

	faces = face_cascade.detectMultiScale(gray_flip, 1.3, 5)
	for (x, y, w, h) in faces:
		frame = cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

	cv.imshow('frame', frame)

	# listening for exit key which is 'q'
	if cv.waitKey(1) == ord('q'):
		break

	# add object detection scripts here

cap.release()
cv.destroyAllWindows()