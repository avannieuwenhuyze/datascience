#---------------------------------------------------
# QSTOM-IT
# Aurélien Vannieuwenhuyze
# 26/04/2020
#---------------------------------------------------


#------ IMPORTS -----
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np


# Taux d'apprentissage
INIT_LR = 1e-3

#Nombre d'iterations
EPOCHS = 25

#Taille de batchs
BS = 8



#------ PREPARATION DES DONNEES -----

cheminTrain = "dataset/train/"
cheminValidation = "dataset/validation/";

nbImagesTrain = len([f for f in os.listdir(cheminTrain)if os.path.isfile(os.path.join(cheminTrain, f))])
print("  > Nombres d'images d'apprentissage : "+str(nbImagesTrain))


nbImagesValidation = len([f for f in os.listdir(cheminValidation)if os.path.isfile(os.path.join(cheminValidation, f))])
print("  > Nombres d'images de validation : "+str(nbImagesValidation))

data = []
labels = []

for file in os.listdir(cheminTrain):
    file_name, file_extension = os.path.splitext(file)
    type = file_name.split("_")[1]

    #Label de l'image
    labels.append(int(type))

    #Chargement de l'image
    image = cv2.imread(cheminTrain+file_name+file_extension)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    #Sauvegarde de l'image
    data.append(image)


# Conversion de image en tableau Numpy
# Egalisation des intensités de l'image
data = np.array(data) / 255.0
labels = np.array(labels)


# One Hot encoding des labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)


# Découpage des données en jeu d'apprentissage (80%) et  jeu de test (20%)
(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=0.20, stratify=labels, random_state=42)


#------ CHARGEMENT ET MODIFICATION DU MODELE -----

# Chargement du modele VGG16
baseModel = VGG16(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))

#Modification de la sortie de VGG
sortieVGG = baseModel.output
sortieVGG = Flatten(name="flatten")(sortieVGG)
sortieVGG = Dense(512, activation="relu")(sortieVGG)
sortieVGG = Dropout(0.5)(sortieVGG)
sortieVGG = Dense(2, activation="softmax")(sortieVGG)

model = Model(inputs=baseModel.input, outputs=sortieVGG)

#On n'oublie pas de geler les couches du réseau VGG car on ne souhaite pas qu'elles perdent d'informations
#lors de la phase de rétro-propagation dû à notre modification de la sortie de VGG.
for layer in baseModel.layers:
	layer.trainable = False


#Compilation du modèle
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])


# Generation de nouvelles images à l'aide de Keras lors de l'apprentissage
augmentation = ImageDataGenerator(
	rotation_range=15,
	fill_mode="nearest")


#Nombre d'iterations
EPOCHS = 1

#Taille des batchs
BS = 8

# Apprentissage
print("Apprentissage...")
H = model.fit_generator(
	augmentation.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# Sauvegarde du modèle
model.save("QSTOMIT-MASQUE.model", save_format="h5")