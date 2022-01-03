import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential

import cv2 as cv

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

	cv.imshow('frame', gray_flip)

	# listening for exit key which is 'q'
	if cv.waitKey(1) == ord('q'):
		break

	# add object detection scripts here

cap.release()
cv.destroyAllWindows()