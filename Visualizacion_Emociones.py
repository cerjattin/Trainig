from tensorflow.keras.models import load_model

modelo = load_model('modelo_deteccion_sentimientos_dmd2.h5')

import cv2
import numpy as np

numero_a_etiqueta = {0: 'Molesto', 1: 'Disgusto', 2: 'Miedo', 3: 'Feliz', 4: 'Triste', 5: 'Sorpresa', 6: 'Neutral'}
numRusell_c1 = {0: 'Molesto', 1: 'Disgusto', 2: 'Miedo', 3: 'Feliz', 4: 'Triste', 5: 'Sorpresa', 6: 'Neutral'}
numRusell_c2 = {0: 'Molesto', 1: 'Disgusto', 2: 'Miedo', 3: 'Feliz', 4: 'Triste', 5: 'Sorpresa', 6: 'Neutral'}
numRusell_c3 = {0: 'Molesto', 1: 'Disgusto', 2: 'Miedo', 3: 'Feliz', 4: 'Triste', 5: 'Sorpresa', 6: 'Neutral'}
numRusell_c4 = {0: 'Molesto', 1: 'Disgusto', 2: 'Miedo', 3: 'Feliz', 4: 'Triste', 5: 'Sorpresa', 6: 'Neutral'}

cap = cv2.VideoCapture(0)  # 0 para la cámara principal

while True:
    ret, frame = cap.read()
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gris, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gris = gris[y:y+h, x:x+w]
        roi_redimensionado = cv2.resize(roi_gris, (48, 48))
        entrada_modelo = np.expand_dims(np.expand_dims(roi_redimensionado, -1), 0) / 255.0
        prediccion = modelo.predict(entrada_modelo)
        etiqueta_predicha = numero_a_etiqueta[np.argmax(prediccion)]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, etiqueta_predicha, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Deteccion de emociones', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
