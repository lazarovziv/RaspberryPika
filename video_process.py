import numpy as np

import cv2 as cv

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