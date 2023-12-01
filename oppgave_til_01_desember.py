"""
Oppgave til fredag, gjenkjenne ansikter ansikter i en videostrøm!
Høres ut som at det same oss gjorde i det første kurset.

Sjå fila `Webcam_capture.py` for å komme i gang.
"""

import cv2 as cv
# import numpy as np
# import matplotlib.pyplot as plt

# cascPath = "haarcascade_frontalface_default.xml"
cascPath = "haarcascade_face.xml"
faceCascade = cv.CascadeClassifier(cascPath)

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

is_capturing, frame = cap.read()
while is_capturing:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=1,
        flags=cv.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if len(faces) > 0:
        print("Something in the picture")

    # Display the resulting frame
    cv.imshow('Capture', gray)
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
