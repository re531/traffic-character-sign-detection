import cv2
import numpy as np


# =============================================================================
# Funciones auxiliares para trabajar con cajas
# =============================================================================

def calcular_iou_y_contencion(box1, box2):
    """
    Calcula dos medidas de solapamiento entre cajas:
    - IoU: intersección entre unión.
    - Contención: proporción de la caja menor que queda dentro de la otra.
    """
    x_izq = max(box1[0], box2[0])
    y_arriba = max(box1[1], box2[1])
    x_der = min(box1[2], box2[2])
    y_abajo = min(box1[3], box2[3])

    if x_der <= x_izq or y_abajo <= y_arriba:
        return 0.0, 0.0

    area_interseccion = (x_der - x_izq) * (y_abajo - y_arriba)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    if area_box1 <= 0 or area_box2 <= 0:
        return 0.0, 0.0

    area_union = area_box1 + area_box2 - area_interseccion
    if area_union <= 0:
        return 0.0, 0.0

    iou = area_interseccion / float(area_union)

    area_minima = min(area_box1, area_box2)
    contencion = area_interseccion / float(area_minima)

    return iou, contencion


def eliminar_repetidos_nms(detecciones, umbral_iou=0.2, umbral_contencion=0.6):
    """
    Quita detecciones repetidas de un mismo panel.
    Para decidir cuál conservar, se usa el score de la detección y se favorecen un poco
    las cajas más grandes, porque suelen cubrir mejor el panel completo.
    """
    if len(detecciones) == 0:
        return []

    # Se usa una puntuación auxiliar para ordenar las detecciones antes del filtrado.
    for det in detecciones:
        ancho = det['box'][2] - det['box'][0]
        alto = det['box'][3] - det['box'][1]
        area = ancho * alto
        det['score_nms'] = det['score'] * (1.0 + (area / 100000.0))

    detecciones = sorted(detecciones, key=lambda x: x['score_nms'], reverse=True)
    detecciones_finales = []

    while len(detecciones) > 0:
        mejor_det = detecciones.pop(0)
        detecciones_finales.append(mejor_det)

        detecciones_restantes = []
        for det in detecciones:
            iou, contencion = calcular_iou_y_contencion(mejor_det['box'], det['box'])

            # La detección se conserva solo si no se solapa demasiado con otra ya aceptada.
            if iou < umbral_iou and contencion < umbral_contencion:
                detecciones_restantes.append(det)

        detecciones = detecciones_restantes

    return detecciones_finales


# =============================================================================
# Detector de paneles
# =============================================================================

class DetectorPaneles:
    def __init__(self):
        """Inicializa los parámetros principales del detector."""
        self.mser = cv2.MSER_create(delta=5, min_area=600, max_area=90000)
        self.tamano_base = (80, 40)

        # Rango HSV utilizado para detectar azul saturado en los paneles.
        self.lower_blue = np.array([100, 130, 40])
        self.upper_blue = np.array([140, 255, 255])

    def buscar_rectangulos_grandes(self, imagen):
        """
        Detector complementario basado en Canny y contornos.
        Se utiliza para recuperar paneles grandes o borrosos que MSER no detecta bien.
        """
        detecciones_extra = []
        alto_img, ancho_img = imagen.shape[:2]

        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        bordes = cv2.Canny(blur, 30, 100)
        contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contornos:
            area = cv2.contourArea(cnt)
            if area > 4000:
                x, y, w, h = cv2.boundingRect(cnt)
                extent = area / float(w * h)

                if extent > 0.60 and x > 5 and y > 5:
                    relacion_aspecto = w / float(h)

                    if 0.8 < relacion_aspecto < 3.2 and y < alto_img * 0.6:
                        roi = imagen[y:y + h, x:x + w]
                        roi_resized = cv2.resize(roi, self.tamano_base)
                        hsv = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2HSV)

                        # Rango algo más permisivo para no perder paneles degradados
                        lower_blue_relajado = np.array([90, 40, 40])
                        mask = cv2.inRange(hsv, lower_blue_relajado, self.upper_blue)
                        score = cv2.countNonZero(mask) / float(self.tamano_base[0] * self.tamano_base[1])

                        if score > 0.20:
                            detecciones_extra.append({
                                'box': [x, y, x + w, y + h],
                                'score': score
                            })

        return detecciones_extra

    def buscar_paneles_niebla(self, imagen):
        """
        Detector complementario para escenas con niebla o baja saturación.
        Usa un rango HSV más permisivo y operaciones morfológicas para unir regiones.
        """
        detecciones_niebla = []
        alto_img, ancho_img = imagen.shape[:2]

        # Se analiza solo la mitad superior, donde normalmente aparecen los paneles.
        roi_superior = imagen[0:int(alto_img * 0.5), 0:ancho_img]

        lower_blue_fog = np.array([95, 30, 50])
        upper_blue_fog = np.array([135, 110, 220])

        hsv = cv2.cvtColor(roi_superior, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_blue_fog, upper_blue_fog)

        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contornos:
            area = cv2.contourArea(cnt)
            if 1500 < area < 40000:
                x, y, w, h = cv2.boundingRect(cnt)
                extent = area / float(w * h)
                relacion_aspecto = w / float(h)

                if extent > 0.65 and 0.8 < relacion_aspecto < 3.2:
                    roi_mask = mask[y:y + h, x:x + w]
                    densidad = cv2.countNonZero(roi_mask) / float(w * h)

                    if densidad > 0.50:
                        detecciones_niebla.append({
                            'box': [x, y, x + w, y + h],
                            'score': densidad
                        })

        return detecciones_niebla

    def detectar(self, imagen):
        """Ejecuta el detector completo sobre una imagen."""
        alto_img, ancho_img = imagen.shape[:2]

        # Preprocesado en escala de grises antes de aplicar MSER
        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        regiones, _ = self.mser.detectRegions(gray_blur)
        detecciones_iniciales = []

        # Detección principal mediante MSER
        for region in regiones:
            x, y, w, h = cv2.boundingRect(region)

            # Filtros básicos de posición y compacidad
            if y < alto_img * 0.05 or y > alto_img * 0.65:
                continue
            if (len(region) / float(w * h)) < 0.45:
                continue

            relacion_aspecto = w / float(h)
            if 0.8 < relacion_aspecto < 3.2:
                # Se amplía la caja para incluir mejor el borde del panel
                mw, mh = int(w * 0.08), int(h * 0.08)
                x1, y1 = max(0, x - mw), max(0, y - mh)
                x2, y2 = min(ancho_img, x + w + mw), min(alto_img, y + h + mh)

                roi = imagen[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                roi_resized = cv2.resize(roi, self.tamano_base)
                hsv = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
                score = cv2.countNonZero(mask) / float(self.tamano_base[0] * self.tamano_base[1])

                if score > 0.40:
                    detecciones_iniciales.append({
                        'box': [x1, y1, x2, y2],
                        'score': score
                    })

        # Se combinan las detecciones del método principal y de los métodos complementarios
        detecciones_totales = (
            detecciones_iniciales
            + self.buscar_rectangulos_grandes(imagen)
            + self.buscar_paneles_niebla(imagen)
        )

        # Validación final mediante densidad de bordes dentro de la caja
        detecciones_validadas = []
        for det in detecciones_totales:
            x1, y1, x2, y2 = det['box']
            roi = imagen[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            bordes = cv2.Canny(roi_gray, 30, 100)
            densidad_bordes = cv2.countNonZero(bordes) / float(roi.shape[0] * roi.shape[1])

            if densidad_bordes > 0.012:
                detecciones_validadas.append(det)

        return eliminar_repetidos_nms(
            detecciones_validadas,
            umbral_iou=0.2,
            umbral_contencion=0.6
        )
