import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense # type: ignore
from sklearn.metrics import confusion_matrix, roc_curve, auc # type: ignore
import seaborn as sns # type: ignore
import itertools

# Función para calcular la Descomposición de Modos Dinámicos (DMD)
def dmd(X):
    X1 = X[:, :-1]  # Estado anterior (todas las columnas menos la última)
    X2 = X[:, 1:]   # Estado siguiente (todas las columnas menos la primera)
    
    # SVD de la matriz X1
    U, Sigma, Vh = np.linalg.svd(X1, full_matrices=False)
    r = min(10, U.shape[1])  # Limitar a 10 modos o el máximo disponible
    U_r = U[:, :r]
    Sigma_r = np.diag(Sigma[:r])
    V_r = Vh.conj().T[:, :r]

    # Aproximación del operador Koopman
    A_tilde = U_r.conj().T @ X2 @ V_r @ np.linalg.inv(Sigma_r)
    Lambda, W = np.linalg.eig(A_tilde)
    Phi = X2 @ V_r @ np.linalg.inv(Sigma_r) @ W

    return Phi.real

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
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
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
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
            imagenes.append(imagen.flatten())
            etiquetas.append(etiqueta_a_numero[etiqueta])
    return np.array(imagenes), np.array(etiquetas)

# Carga de datos

X_train, y_train = cargar_datos('C:/WorkSpace/fer2013/train')
X_test, y_test = cargar_datos('C:/WorkSpace/fer2013/test')

# Aplicar DMD para extraer características dinámicas
X_train_dmd = dmd(X_train.reshape(X_train.shape[0], -1))  # Asegúrate de que X_train es 2D
X_test_dmd = dmd(X_test.reshape(X_test.shape[0], -1))

# Preparación de datos
X_train = np.expand_dims(X_train, -1) / 255.0
X_test = np.expand_dims(X_test, -1) / 255.0
y_train = to_categorical(y_train, num_classes=7)
y_test_categorical = to_categorical(y_test, num_classes=7)

# Combinar características DMD con imágenes originales (aplanadas)
X_train_combined = np.concatenate([X_train.reshape(len(X_train), -1), X_train_dmd], axis=1)
X_test_combined = np.concatenate([X_test.reshape(len(X_test), -1), X_test_dmd], axis=1)

# Construcción y entrenamiento del modelo con características combinadas
modelo = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_combined.shape[1],)),
    Dense(64, activation='relu'),
    Dense(7, activation='softmax')
])
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
modelo.fit(X_train_combined, y_train, epochs=50, batch_size=64, validation_data=(X_test_combined, y_test_categorical))

# Guardar el modelo
modelo.save('modelo_deteccion_sentimientos_dmd.h5')

# Evaluación del modelo y visualización
predictions = modelo.predict(X_test_combined)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test_categorical, axis=1)

cm = confusion_matrix(true_classes, predicted_classes)
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
plot_confusion_matrix(cm, classes=class_names, title='Confusion Matrix')
plt.show()

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
