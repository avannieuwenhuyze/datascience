#---------------------------------------------------
# QSTOM-IT
# Aurélien Vannieuwenhuyze
# 26/04/2020
#---------------------------------------------------


import cv2
import os
import numpy as np
import tensorflow as tf


# Detection de visages à l'aide du model Cafee Model Zoo
# http://caffe.berkeleyvision.org/model_zoo.html
prototxt_path = os.path.join('model_data/deploy.prototxt')
caffemodel_path = os.path.join('model_data/weights.caffemodel')
model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

#Chargement du modèle permettant de détecter le port du masque
modelMasque = tf.keras.models.load_model("QSTOMIT-MASQUE.model")

#Capture de la caméra (idCamera)
cap = cv2.VideoCapture(0)

while True:

    _, image = cap.read()

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    model.setInput(blob)
    detections = model.forward()

    h = image.shape[0]
    w = image.shape[1]

    for i in range(0, detections.shape[2]):
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        confidence = detections[0, 0, i, 2]

        # If confidence > 0.5, save it as a separate file
        if (confidence > 0.5):
            frame = image[startY:endY, startX:endX]

            #Appel du modèle appris pour la detection de masque
            capture = cv2.resize(frame, (224, 224))
            capture = capture.reshape((1, capture.shape[0], capture.shape[1], capture.shape[2]))
            predict = modelMasque.predict(capture)
            pasDeMasque = predict[0][0]
            avecMasque = predict[0][1]

            if (pasDeMasque > avecMasque):
                cv2.rectangle(image, (startX, startY), (endX, endY),(0, 0, 255), 2)
                cv2.putText(image, "PAS DE MASQUE", (startX, startY-10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            else:
                cv2.rectangle(image, (startX, startY), (endX, endY),(0, 255, 0), 2)
                cv2.putText(image, "OK", (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)


    # Affichage de l'image
    cv2.imshow('img', image)

    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

cap.release()