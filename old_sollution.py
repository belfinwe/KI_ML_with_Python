import numpy as np
import cv2
import matplotlib.pyplot as plt


def type_of_script():
    try:
        ipy_str = str(type(get_ipython()))
        if 'zmqshell' in ipy_str:
            return 'jupyter'
        if 'terminal' in ipy_str:
            return 'ipython'
    except:
        return 'terminal'


runtime_env = type_of_script()


# Due to jupyters architecture we can not use cv2.imshow() so we create a helper method to utilize pyplot.
def show_image(image, title):
    if (runtime_env == "jupyter"):
        plt.imshow(image)
        plt.title(title)
        plt.show()
    else:
        cv2.imshow(title, image)
        cv2.waitKey(1)


cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

cap = cv2.VideoCapture(0)
# Change to a smaller framesize for speed
ret = cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
ret = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 200)

# Try to get the first frame
if cap.isOpened():
    is_capturing, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
else:
    is_capturing = False

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while is_capturing:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Our operations on the frame come here
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # First: Make it grayscale.

    faces = faceCascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=1,
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(gray_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if len(faces) > 0:
        # print("Something in the picture")
        if (runtime_env == "jupyter"):
            show_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), "Capture")
            pass
        else:
            # cv2.imshow does now work on my Mac?
            cv2.imshow('Capture', gray_image)
            if (cv2.waitKey(1) == ord("q")):
                break

    # Avoids a NotImplementedError caused by `plt.pause`
    try:
        plt.pause(0.05)
    except Exception:
        pass

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
