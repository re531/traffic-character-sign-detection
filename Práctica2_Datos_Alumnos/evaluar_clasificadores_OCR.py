# Asignatura de Visión Artificial (URJC). Script de evaluación.
# @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
# @date 2025


import argparse
import os
#import panel_det
import matplotlib.pyplot as plt
import cv2
import numpy as np
import sklearn


# Importar el clasificador que hemos creado
from lda_normal_bayes_classifier import LdaNormalBayesClassifier
from lda_normal_bayes_classifier import LdaNormalBayesClassifier
from clasificadores_alternativos import PcaKnnClassifier, LdaRandomForestClassifier # <-- NUEVO

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.get_cmap('Blues')):    
    '''
    Given a confusión matrix in cm (np.array) it plots it in a fancy way.
    '''
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, range(cm.shape[0]))
    plt.yticks(tick_marks, range(cm.shape[0]))
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    ax = plt.gca()
    width = cm.shape[1]
    height = cm.shape[0]

    for x in range(width):
        for y in range(height):
            ax.annotate(str(cm[y,x]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

def cargar_datos_ocr(ruta_directorio):
    """
    Lee las carpetas de entrenamiento/validación.
    Se asume que dentro de ruta_directorio hay subcarpetas con los nombres de las clases
    (e.g., 'A', 'B', 'C', ..., '0', '1').
    Devuelve un diccionario { 'A': [img1, img2...], 'B': [...] }
    """
    images_dict = {}
    print(f"Cargando datos desde {ruta_directorio}...")
    
    # Recorrer subdirectorios
    for root, dirs, files in os.walk(ruta_directorio):
        # El nombre del último directorio es la clase (el caracter)
        char_class = os.path.basename(root)
        
        # Si no es un caracter válido (ej. la propia carpeta raíz), saltar
        if len(char_class) != 1:
            continue
            
        imgs = []
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(root, file)
                img = cv2.imread(filepath)
                if img is not None:
                    imgs.append(img)
        
        if imgs:
            images_dict[char_class] = imgs
            
    print(f"  Cargadas {sum([len(v) for v in images_dict.values()])} imágenes de {len(images_dict)} clases.")
    return images_dict

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Trains and executes a given classifier for OCR over testing images')
    parser.add_argument(
        '--classifier', type=str, default="LDA_Bayes", help='Classifier string name')
    parser.add_argument(
        '--train_path', default="./train_ocr", help='Select the training data dir')
    parser.add_argument(
        '--validation_path', default="./test_ocr", help='Select the validation data dir')

    args = parser.parse_args()

    # 1) Cargar las imágenes de entrenamiento
    train_dict = cargar_datos_ocr(args.train_path)

    # 2) Cargar datos de validación
    validation_dict = cargar_datos_ocr(args.validation_path)

    # Inicializar el clasificador
    print(f"Inicializando clasificador {args.classifier}...")
    if args.classifier == "LDA_Bayes":
        ocr_model = LdaNormalBayesClassifier(ocr_char_size=(25, 25))
    elif args.classifier == "PCA_KNN":
        ocr_model = PcaKnnClassifier(ocr_char_size=(25, 25))
    elif args.classifier == "LDA_RF":
        ocr_model = LdaRandomForestClassifier(ocr_char_size=(25, 25))
    else:
        print("Clasificador no reconocido. Usando LDA_Bayes por defecto.")
        ocr_model = LdaNormalBayesClassifier(ocr_char_size=(25, 25))

    # 3) Entrenar clasificador
    print("Entrenando modelo...")
    ocr_model.train(train_dict)

    # 4) Ejecutar el clasificador sobre los datos de test/validación
    print("Evaluando validación...")
    
    # Usamos las funciones de la clase base OCRClassifier
    gt_labels = ocr_model.get_labels_dict(validation_dict)
    predicted_labels = ocr_model.predict_dict(validation_dict)

    # 5) Evaluar los resultados
    accuracy = sklearn.metrics.accuracy_score(gt_labels, predicted_labels)
    print(f"----------------------------------------")
    print(f"Accuracy = {accuracy*100:.2f}%")
    print(f"----------------------------------------")
    
    # Mostrar matriz de confusión (Opcional, si son muchas clases puede verse pequeña)
    # cm = sklearn.metrics.confusion_matrix(gt_labels, predicted_labels)
    # plot_confusion_matrix(cm, title=f"Confusion Matrix - {args.classifier}")
    # plt.show()