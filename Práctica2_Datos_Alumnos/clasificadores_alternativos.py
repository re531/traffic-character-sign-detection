import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from ocr_classifier import OCRClassifier

class BasePixelClassifier(OCRClassifier):
    #Clase padre de la que heredan los otros clasificadores.
    def extraer_caracteristicas(self, img):
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contornos:
            c_max = max(contornos, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c_max)
            roi = thresh[y:y+h, x:x+w]
        else:
            roi = thresh
            
        roi_resized = cv2.resize(roi, self.ocr_char_size)
        return roi_resized.flatten()


#1: PCA + KNN
class PcaKnnClassifier(BasePixelClassifier):
    def __init__(self, ocr_char_size=(25, 25)):
        super().__init__(ocr_char_size)
        # Reducimos los 625 píxeles a las 50 características principales
        self.reductor = PCA(n_components=50) 
        # Clasificamos mirando los 3 vecinos más parecidos
        self.classifier = KNeighborsClassifier(n_neighbors=3)

    def train(self, images_dict):
        X_list, y_list = [], []
        for char_class, list_of_imgs in images_dict.items():
            label = self.char2label(char_class)
            for img in list_of_imgs:
                X_list.append(self.extraer_caracteristicas(img))
                y_list.append(label)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)

        # Entrenamos la reducción PCA
        X_reducido = self.reductor.fit_transform(X) 
        self.classifier.fit(X_reducido, y)
        return X, y

    def predict(self, img):
        vector = self.extraer_caracteristicas(img)
        vector_matrix = np.array([vector], dtype=np.float32)
        vector_reducido = self.reductor.transform(vector_matrix)
        return int(self.classifier.predict(vector_reducido)[0])

#2: LDA + RANDOM FOREST
class LdaRandomForestClassifier(BasePixelClassifier):
    def __init__(self, ocr_char_size=(25, 25)):
        super().__init__(ocr_char_size)
        self.reductor = LinearDiscriminantAnalysis()
        # En vez de un árbol, usamos 100 para que vote la mayoría
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, images_dict):
        X_list, y_list = [], []
        for char_class, list_of_imgs in images_dict.items():
            label = self.char2label(char_class)
            for img in list_of_imgs:
                X_list.append(self.extraer_caracteristicas(img))
                y_list.append(label)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)

        X_reducido = self.reductor.fit_transform(X, y)
        self.classifier.fit(X_reducido, y)
        return X, y

    def predict(self, img):
        vector = self.extraer_caracteristicas(img)
        vector_matrix = np.array([vector], dtype=np.float32)
        vector_reducido = self.reductor.transform(vector_matrix)
        return int(self.classifier.predict(vector_reducido)[0])