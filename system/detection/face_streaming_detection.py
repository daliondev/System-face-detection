import cv2
import numpy as np


streaming = cv2.VideoCapture(0)

faceClassif = cv2.CascadeClassifier(
    'system\detection\haarcascade_frontalface_default.xml')

while True:
    ret, frame = streaming.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Streaming', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

streaming.realease()
cv2.destroyAllWindows()
