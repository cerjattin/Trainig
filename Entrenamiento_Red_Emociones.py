import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
import os

def cargar_datos(directorio):
    imagenes = []
    etiquetas = []
    etiqueta_a_numero = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6}

    for etiqueta in os.listdir(directorio):
        for imagen_nombre in os.listdir(os.path.join(directorio, etiqueta)):
            imagen = cv2.imread(os.path.join(directorio, etiqueta, imagen_nombre), 0)
            imagen = cv2.resize(imagen, (48, 48))
            imagenes.append(imagen)
            etiquetas.append(etiqueta_a_numero[etiqueta])  # Usamos el mapeo para convertir la etiqueta a un número
    return np.array(imagenes), np.array(etiquetas)


X_train, y_train = cargar_datos('C:/WorkSpace/fer2013/train')
X_test, y_test = cargar_datos('C:/WorkSpace/fer2013/test')

# Redimensionar y normalizar
X_train = np.expand_dims(X_train, -1) / 255.0
X_test = np.expand_dims(X_test, -1) / 255.0

y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)

modelo = Sequential()
modelo.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
modelo.add(MaxPooling2D(pool_size=(2, 2)))
modelo.add(Conv2D(64, (3, 3), activation='relu'))
modelo.add(MaxPooling2D(pool_size=(2, 2)))
modelo.add(Flatten())
modelo.add(Dense(128, activation='relu'))
modelo.add(Dense(7, activation='softmax'))

modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
modelo.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))

# Guardar el modelo
modelo.save('modelo_deteccion_sentimientos.h5')
