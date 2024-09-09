import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns  # Importa Seaborn para la visualización
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def cargar_datos(directorio):
    imagenes = []
    etiquetas = []
    etiqueta_a_numero = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6}
    for etiqueta in os.listdir(directorio):
        for imagen_nombre in os.listdir(os.path.join(directorio, etiqueta)):
            imagen = cv2.imread(os.path.join(directorio, etiqueta, imagen_nombre), 0)
            imagen = cv2.resize(imagen, (48, 48))
            imagenes.append(imagen)
            etiquetas.append(etiqueta_a_numero[etiqueta])
    return np.array(imagenes), np.array(etiquetas)

# Carga de datos
X_train, y_train = cargar_datos('C:/WorkSpace/fer2013/train')
X_test, y_test = cargar_datos('C:/WorkSpace/fer2013/test')

# Preparación de datos
X_train = np.expand_dims(X_train, -1) / 255.0
X_test = np.expand_dims(X_test, -1) / 255.0
y_train = to_categorical(y_train, num_classes=7)
y_test_categorical = to_categorical(y_test, num_classes=7)

# Construcción y entrenamiento del modelo
modelo = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(7, activation='softmax')
])
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
modelo.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test_categorical))

# Guardar el modelo
modelo.save('modelo_deteccion_sentimientos2.h5')

# Evaluación del modelo
predictions = modelo.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test_categorical, axis=1)

# Matriz de Confusión y visualización
cm = confusion_matrix(true_classes, predicted_classes)
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
plot_confusion_matrix(cm, classes=class_names, title='Confusion Matrix')
plt.show()

# Curva ROC para cada clase
fpr, tpr, roc_auc = {}, {}, {}
for i in range(7):
    fpr[i], tpr[i], _ = roc_curve(y_test_categorical[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
plt.figure()
for i in range(7):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

