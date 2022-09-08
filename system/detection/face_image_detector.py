import cv2
import numpy as np


def face_image_detection():
    faceClassif = cv2.CascadeClassifier(
        'system\detection\haarcascade_frontalface_default.xml')
    numbFaces = 0

    # Carga de las imagenes
    path = 'system\images\image1.jpg'
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Caracteristicas de la deteccion
    faces = faceClassif.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(30, 30),
                                         maxSize=(200, 200))

    # Dibujado de los cuadrados al rededor de las caras detectadas
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        numbFaces += 1

    # Conteo de los objetos
    text = f'Numero de rostros: {numbFaces}'
    cv2.putText(image, text, (20, 30), cv2.FONT_ITALIC,
                0.7, (0, 0, 0), 2)
    cv2.imshow('image', image)

    # Salida de la interfaz
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Test
if __name__ == '__main__':
    face_image_detection()
