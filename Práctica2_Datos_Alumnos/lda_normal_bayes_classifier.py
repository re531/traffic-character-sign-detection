# @brief LdaNormalBayesClassifier
# @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
# @date 2025

import cv2
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from ocr_classifier import OCRClassifier # Ojo: he quitado el '.' inicial para que importe bien si están en la misma carpeta

class LdaNormalBayesClassifier(OCRClassifier):
    """
    Classifier for Optical Character Recognition using LDA and the Bayes with Gaussian classfier.
    """

    def __init__(self, ocr_char_size=(25,25)):
        super().__init__(ocr_char_size)
        self.lda = LinearDiscriminantAnalysis()
        # Usamos el clasificador de Bayes de OpenCV
        self.classifier = cv2.ml.NormalBayesClassifier_create()

    def extraer_caracteristicas(self, img):
        """
        Dada una imagen de un caracter (ya recortado), la umbraliza,
        busca el contorno principal (el carácter), recorta ajustado al contorno,
        redimensiona y aplana a un vector fila de 625 elementos.
        """
        # Asegurarnos de que está en gris
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Umbralización (según el PDF: cv2.adaptiveThreshold)
        # Ojo: como las letras del dataset suelen ser blancas sobre negro o viceversa,
        # puede que necesitemos invertir. Asumimos texto blanco sobre fondo negro
        # Si el dataset es texto negro sobre fondo blanco, usamos THRESH_BINARY_INV
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Buscar contornos
        contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contornos:
            # Asumimos que el contorno más grande es la letra
            c_max = max(contornos, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c_max)
            roi = thresh[y:y+h, x:x+w]
        else:
            # Si falla, usamos la imagen umbralizada entera
            roi = thresh

        # Redimensionar al tamaño fijo (25x25)
        roi_resized = cv2.resize(roi, self.ocr_char_size)
        
        # Aplanar a vector de 1D de 625 columnas
        vector = roi_resized.flatten()
        return vector

    def train(self, images_dict):
        """.
        Given character images in a dictionary of list of char images of fixed size, 
        train the OCR classifier. The dictionary keys are the class of the list of images 
        (or corresponding char).
        """
        X_list = []
        y_list = []

        # Extraer características para cada imagen
        for char_class, list_of_imgs in images_dict.items():
            # Convertimos el caracter (ej: 'A') a una etiqueta numérica (ej: 11)
            label = self.char2label(char_class) 
            for img in list_of_imgs:
                vector = self.extraer_caracteristicas(img)
                X_list.append(vector)
                y_list.append(label)

        # Matriz C (características) y Vector E (etiquetas)
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32) # OpenCV necesita int32 para las etiquetas

        # 1. Reducción de Dimensionalidad (LDA)
        # sklearn usa float64 por defecto, le pasamos X directamente.
        # fit() encuentra la matriz de proyección.
        self.lda.fit(X, y)
        
        # transform() proyecta C en el espacio de dimensión menor (CR)
        X_reducido = self.lda.transform(X)

        # 2. Entrenar el clasificador
        # OpenCV's NormalBayesClassifier requiere float32 para los datos
        X_reducido_cv = np.array(X_reducido, dtype=np.float32)
        
        # Entrenamos el clasificador
        self.classifier.train(X_reducido_cv, cv2.ml.ROW_SAMPLE, y)

        return X, y

    def predict(self, img):
        """.
        Given a single image of a character already cropped classify it.
        """
        # Extraer características (redimensionar y aplanar)
        vector = self.extraer_caracteristicas(img)
        
        # El vector debe ser una matriz (fila, columnas) para el LDA
        vector_matrix = np.array([vector], dtype=np.float32)

        # Reducir dimensiones con el LDA entrenado
        vector_reducido = self.lda.transform(vector_matrix)
        vector_reducido_cv = np.array(vector_reducido, dtype=np.float32)

        # Predecir con Bayes
        _, results = self.classifier.predict(vector_reducido_cv)
        
        # results es una matriz, sacamos el valor escalar
        predicted_label = int(results[0][0])

        return predicted_label