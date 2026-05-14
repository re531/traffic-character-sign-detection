# Asignatura de Visión Artificial (URJC). Script de evaluación.
# @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
# @date 2025

import argparse
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import sklearn
from collections import Counter

# Importar los clasificadores implementados
from lda_normal_bayes_classifier import LdaNormalBayesClassifier
from clasificadores_alternativos import PcaKnnClassifier, LdaRandomForestClassifier


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
            ax.annotate(str(cm[y, x]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')


def pad_to_square(img_gray):
    """
    Añade bordes negros para hacer la imagen cuadrada sin deformar el carácter.
    Esto ayuda antes de redimensionar a 25x25.
    """
    h, w = img_gray.shape

    if h == w:
        return img_gray

    elif h > w:
        pad_left = (h - w) // 2
        pad_right = h - w - pad_left
        return cv2.copyMakeBorder(
            img_gray, 0, 0, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=0
        )

    else:
        pad_top = (w - h) // 2
        pad_bottom = w - h - pad_top
        return cv2.copyMakeBorder(
            img_gray, pad_top, pad_bottom, 0, 0,
            cv2.BORDER_CONSTANT, value=0
        )


def cargar_datos_ocr(ruta_directorio, img_size=(25, 25)):
    """
    Lee las carpetas de entrenamiento/validación.

    """
    images_dict = {}
    archivos_procesados = 0

    print(f"Cargando datos desde {ruta_directorio}...")

    # Recorrer subdirectorios
    for root, dirs, files in os.walk(ruta_directorio):
        # El nombre del último directorio es la clase (el caracter)
        char_class = os.path.basename(root)

        # Si no es un caracter válido, saltar
        if len(char_class) != 1:
            continue

        imgs = []

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(root, file)

                # Leemos directamente en gris porque el clasificador trabaja con intensidad
                img_gray = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

                if img_gray is None:
                    continue

                archivos_procesados += 1

                # Normalizar a texto blanco sobre fondo negro.
                # Se estima el fondo mirando las esquinas de la imagen.
                bg_intensity = np.median([
                    img_gray[0, 0],
                    img_gray[0, -1],
                    img_gray[-1, 0],
                    img_gray[-1, -1]
                ])

                if bg_intensity > 127:
                    img_gray = 255 - img_gray

                # Umbralización mediante Otsu
                _, thresh = cv2.threshold(
                    img_gray, 0, 255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )

                # Buscar contorno principal caracter
                contornos, _ = cv2.findContours(
                    thresh,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                if contornos:
                    c_max = max(contornos, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(c_max)

                    if w > 0 and h > 0:
                        roi_gray = img_gray[y:y + h, x:x + w]
                    else:
                        roi_gray = img_gray
                else:
                    roi_gray = img_gray

                # Normalización y padding antes del resize
                roi_gray = cv2.normalize(roi_gray, None, 0, 255, cv2.NORM_MINMAX)
                roi_padded = pad_to_square(roi_gray)
                roi_resized = cv2.resize(
                    roi_padded,
                    img_size,
                    interpolation=cv2.INTER_AREA
                )

                imgs.append(roi_resized)

        if imgs:
            images_dict[char_class] = imgs

    print(f"  Cargadas {sum([len(v) for v in images_dict.values()])} imágenes de {len(images_dict)} clases.")
    print(f"  Archivos procesados con padding cuadrado: {archivos_procesados}")

    return images_dict


def mostrar_errores_frecuentes(gt_labels, predicted_labels, ocr_model, top_n=15):
    """
    Muestra los pares de caracteres que más se confunden.

        """
    errores = []

    for real, pred in zip(gt_labels, predicted_labels):
        if real != pred:
            real_char = ocr_model.label2char(real)
            pred_char = ocr_model.label2char(pred)
            errores.append((real_char, pred_char))

    contador = Counter(errores)

    print("\nErrores más frecuentes:")
    if len(contador) == 0:
        print("No se han encontrado errores.")
        return

    for (real_char, pred_char), num in contador.most_common(top_n):
        print(f"{real_char} - {pred_char}: {num} veces")


def evaluar_por_grupos(gt_labels, predicted_labels, ocr_model):
    """
    Calcula la accuracy separando dígitos, minúsculas y mayúsculas.
    """
    grupos = {
        "Dígitos": [],
        "Minúsculas": [],
        "Mayúsculas": []
    }

    for real, pred in zip(gt_labels, predicted_labels):
        char_real = ocr_model.label2char(real)

        if char_real.isdigit():
            grupos["Dígitos"].append((real, pred))
        elif char_real.islower():
            grupos["Minúsculas"].append((real, pred))
        elif char_real.isupper():
            grupos["Mayúsculas"].append((real, pred))

    print("\nAccuracy por grupos:")

    for nombre_grupo, valores in grupos.items():
        if len(valores) == 0:
            continue

        reales = [v[0] for v in valores]
        preds = [v[1] for v in valores]

        acc = sklearn.metrics.accuracy_score(reales, preds)
        print(f"{nombre_grupo}: {acc * 100:.2f}% ({len(valores)} muestras)")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Trains and executes a given classifier for OCR over testing images'
    )
    parser.add_argument(
        '--classifier', type=str, default="LDA_Bayes", help='Classifier string name'
    )
    parser.add_argument(
        '--train_path', default="./train_ocr", help='Select the training data dir'
    )
    parser.add_argument(
        '--validation_path', default="./test_ocr", help='Select the validation data dir'
    )
    parser.add_argument(
        '--show_confusion', action="store_true", help='Show confusion matrix'
    )

    args = parser.parse_args()

    # 1. Cargar las imágenes de entrenamiento
    train_dict = cargar_datos_ocr(args.train_path)

    # 2. Cargar datos de validación
    validation_dict = cargar_datos_ocr(args.validation_path)

    # Inicializar el clasificador
    print(f"Inicializando {args.classifier}")

    if args.classifier == "LDA_Bayes":
        ocr_model = LdaNormalBayesClassifier(ocr_char_size=(25, 25))
    elif args.classifier == "PCA_KNN":
        ocr_model = PcaKnnClassifier(ocr_char_size=(25, 25))
    elif args.classifier == "LDA_RF":
        ocr_model = LdaRandomForestClassifier(ocr_char_size=(25, 25))
    else:
        print("Clasificador no reconocido. Usando LDA_Bayes por defecto.")
        ocr_model = LdaNormalBayesClassifier(ocr_char_size=(25, 25))

    # 3. Entrenar clasificador
    print("Entrenando modelo")
    ocr_model.train(train_dict)

    # 4. Ejecutar el clasificador sobre los datos de test/validación
    print("Evaluando validación")

    # Usamos las funciones de la clase base OCRClassifier
    gt_labels = ocr_model.get_labels_dict(validation_dict)
    predicted_labels = ocr_model.predict_dict(validation_dict)

    # 5. Evaluar los resultados
    accuracy = sklearn.metrics.accuracy_score(gt_labels, predicted_labels)

    precision_macro = sklearn.metrics.precision_score(
        gt_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )

    recall_macro = sklearn.metrics.recall_score(
        gt_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )

    f1_macro = sklearn.metrics.f1_score(
        gt_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )

    precision_weighted = sklearn.metrics.precision_score(
        gt_labels,
        predicted_labels,
        average="weighted",
        zero_division=0
    )

    recall_weighted = sklearn.metrics.recall_score(
        gt_labels,
        predicted_labels,
        average="weighted",
        zero_division=0
    )

    f1_weighted = sklearn.metrics.f1_score(
        gt_labels,
        predicted_labels,
        average="weighted",
        zero_division=0
    )

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision macro: {precision_macro * 100:.2f}%")
    print(f"Recall macro: {recall_macro * 100:.2f}%")
    print(f"F1 macro: {f1_macro * 100:.2f}%")
    print(f"Precision weighted: {precision_weighted * 100:.2f}%")
    print(f"Recall weighted: {recall_weighted * 100:.2f}%")
    print(f"F1 weighted: {f1_weighted * 100:.2f}%")

    mostrar_errores_frecuentes(gt_labels, predicted_labels, ocr_model, top_n=15)

    evaluar_por_grupos(gt_labels, predicted_labels, ocr_model)

    # La matriz completa se deja como opción porque con 62 clases no siempre se ve bien.
    if args.show_confusion:
        cm = sklearn.metrics.confusion_matrix(gt_labels, predicted_labels)
        plt.figure(figsize=(20, 20))
        plot_confusion_matrix(cm, title=f'Confusion Matrix: {args.classifier}')
        plt.show()