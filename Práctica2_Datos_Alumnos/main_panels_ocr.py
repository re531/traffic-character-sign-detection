import argparse
import os
import cv2
import numpy as np
import random
from collections import defaultdict
from lda_normal_bayes_classifier import LdaNormalBayesClassifier

def pad_to_square(img_gray):
    h, w = img_gray.shape
    if h == w:
        return img_gray
    elif h > w:
        pad_left = (h - w) // 2
        pad_right = h - w - pad_left
        return cv2.copyMakeBorder(img_gray, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    else:
        pad_top = (w - h) // 2
        pad_bottom = w - h - pad_top
        return cv2.copyMakeBorder(img_gray, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=0)

def extract_roi_with_margin(img, x, y, w, h, margin_ratio=0.08):
    margin = int(h * margin_ratio) + 1
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(img.shape[1], x + w + margin)
    y2 = min(img.shape[0], y + h + margin)
    return img[y1:y2, x1:x2]

def cargar_datos(ruta, img_size=(25, 25)):
    images_dict = defaultdict(list)
    archivos_encontrados = 0
    for root, _, archivos in os.walk(ruta):
        for archivo in archivos:
            if archivo.endswith(".png"):
                filepath = os.path.join(root, archivo)
                char_class = os.path.basename(root)
                img_gray = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                
                if img_gray is not None:
                    bg_intensity = np.median([img_gray[0,0], img_gray[0,-1], img_gray[-1,0], img_gray[-1,-1]])
                    if bg_intensity > 127:
                        img_gray = 255 - img_gray
                        
                    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        c = max(cnts, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(c)
                        if w > 0 and h > 0:
                            roi_gray = extract_roi_with_margin(img_gray, x, y, w, h)
                            roi_gray = cv2.normalize(roi_gray, None, 0, 255, cv2.NORM_MINMAX)
                            roi_padded = pad_to_square(roi_gray)
                            roi_25x25 = cv2.resize(roi_padded, img_size, interpolation=cv2.INTER_AREA)
                            images_dict[char_class].append(roi_25x25)
                            archivos_encontrados += 1
    return dict(images_dict)

def detectar_mascara_panel(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([80, 40, 40])
    upper_blue = np.array([150, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(mask)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(mask_filled, [c], -1, 255, thickness=cv2.FILLED)
        erosion_px = max(5, int(min(img.shape[:2]) * 0.020))
        k_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (erosion_px, erosion_px))
        mask_filled = cv2.erode(mask_filled, k_erode)
        return mask_filled
    return mask

def nms_caracteres(caracteres, iou_thresh=0.20):
    if not caracteres: return []
    caracteres = sorted(caracteres, key=lambda c: c['w'] * c['h'], reverse=True)
    keep = []
    for c1 in caracteres:
        suprimido = False
        area1 = c1['w'] * c1['h']
        for c2 in keep:
            area2 = c2['w'] * c2['h']
            x1 = max(c1['x'], c2['x'])
            y1 = max(c1['y'], c2['y'])
            x2 = min(c1['x'] + c1['w'], c2['x'] + c2['w'])
            y2 = min(c1['y'] + c1['h'], c2['y'] + c2['h'])

            inter = max(0, x2 - x1) * max(0, y2 - y1)
            if inter > 0:
                iou = inter / (area1 + area2 - inter + 1e-6)
                cont1 = inter / (area1 + 1e-6)
                cont2 = inter / (area2 + 1e-6)
                if iou > iou_thresh or cont1 > 0.60 or cont2 > 0.60:
                    suprimido = True
                    break
        if not suprimido:
            keep.append(c1)
    return keep

def leer_panel(img, clf, debug=True):
    img_h, img_w = img.shape[:2]
    img_area = img_h * img_w

    mask_panel = detectar_mascara_panel(img)
    tiene_panel = cv2.countNonZero(mask_panel) > (img_area * 0.05)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    
    b, g, r = cv2.split(img)
    r_eq = clahe.apply(r)

    min_h_ref = max(8, int(img_h * 0.015))
    max_h_ref = int(img_h * 0.50)
    min_area_ref = max(15, img_area * 0.00008)
    max_area_ref = img_area * 0.10

    def contar_validos(thresh_img):
        cnts, _ = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        validos = 0
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if h == 0: continue
            asp = w / float(h)
            
            area_contorno = cv2.contourArea(c)
            extent = area_contorno / float(w * h)
            
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0: continue
            solidity = area_contorno / float(hull_area)
            
            if (min_area_ref < w*h < max_area_ref) and (0.08 < asp < 1.4) and (min_h_ref < h < max_h_ref) and (extent > 0.28):
                validos += 1
        return validos

    candidatos = []
    _, t_otsu_r = cv2.threshold(r_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidatos.append((contar_validos(t_otsu_r), t_otsu_r, "Otsu_Canal_Rojo"))

    t_adp_1 = cv2.adaptiveThreshold(gray_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, -5)
    candidatos.append((contar_validos(t_adp_1), t_adp_1, "Adaptativo_Gris_Suave"))

    t_adp_2 = cv2.adaptiveThreshold(gray_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, -11)
    candidatos.append((contar_validos(t_adp_2), t_adp_2, "Adaptativo_Gris_Fuerte"))

    candidatos.sort(key=lambda x: x[0], reverse=True)
    mejor_n, mejor_thresh, nombre_metodo = candidatos[0]

    if debug:
        cv2.imshow("1. Mascara Binaria", mejor_thresh)

    contours, _ = cv2.findContours(mejor_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    caracteres_raw = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h == 0: continue
        
        aspect_ratio = w / float(h)
        cx, cy = int(x + w/2), int(y + h/2)

        if tiene_panel and mask_panel[cy, cx] == 0: continue
        if not (min_area_ref < w*h < max_area_ref): continue
        if not (0.08 < aspect_ratio < 1.4): continue
        if not (min_h_ref < h < max_h_ref): continue

        area_contorno = cv2.contourArea(c)
        extent = area_contorno / float(w * h)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0: continue
        solidity = area_contorno / float(hull_area)
        
        if extent < 0.28 or solidity < 0.38: 
            continue 

        roi_gray = extract_roi_with_margin(gray, x, y, w, h)
        if roi_gray.size == 0 or roi_gray.std() < 10: continue

        roi_gray = cv2.normalize(roi_gray, None, 0, 255, cv2.NORM_MINMAX)
        roi_padded = pad_to_square(roi_gray)
        roi_25x25 = cv2.resize(roi_padded, (25, 25), interpolation=cv2.INTER_AREA)

        caracteres_raw.append({
            'x': x, 'y': y, 'w': w, 'h': h, 
            'roi_clasificador': roi_25x25,
            'center_x': cx, 'center_y': cy
        })

    if not caracteres_raw: return ""

    alturas = sorted([c['h'] for c in caracteres_raw])
    if len(alturas) > 6:
        trim = int(len(alturas) * 0.15)
        mediana_h = np.median(alturas[trim:-trim])
    else:
        mediana_h = np.median(alturas)

    caracteres = [c for c in caracteres_raw if 0.5 * mediana_h < c['h'] < 1.6 * mediana_h]
    caracteres = nms_caracteres(caracteres)

    if not caracteres: return ""

    lineas = []
    chars_restantes = caracteres.copy()
    distancia_max = max(5.0, mediana_h * 0.40)

    while len(chars_restantes) >= 2:
        mejor_inliers = []
        for _ in range(500):
            p1, p2 = random.sample(chars_restantes, 2)
            dx = p2['center_x'] - p1['center_x']
            if abs(dx) < 1: continue
            m = (p2['center_y'] - p1['center_y']) / dx
            if abs(m) > 0.35: continue
            
            c_recta = p1['center_y'] - m * p1['center_x']
            inliers = [c for c in chars_restantes if abs(m * c['center_x'] - c['center_y'] + c_recta) / np.sqrt(m**2 + 1) < distancia_max]

            if len(inliers) > len(mejor_inliers):
                mejor_inliers = inliers

        if len(mejor_inliers) < 2: break
        lineas.append(mejor_inliers)
        chars_restantes = [c for c in chars_restantes if c not in mejor_inliers]

    lineas.extend([[c] for c in chars_restantes])

    lineas_limpias = []
    for linea in lineas:
        linea.sort(key=lambda c: c['center_x'])
        
        if len(linea) == 1:
            continue
            
        linea_filtrada = []
        for i, char in enumerate(linea):
            dist_izq = (char['center_x'] - linea[i-1]['center_x']) if i > 0 else float('inf')
            dist_der = (linea[i+1]['center_x'] - char['center_x']) if i < len(linea)-1 else float('inf')
            
            dist_minima = min(dist_izq, dist_der)
            
            if dist_minima < char['h'] * 4.0:
                linea_filtrada.append(char)
                
        if len(linea_filtrada) >= 2:
            lineas_limpias.append(linea_filtrada)
            
    lineas = lineas_limpias

    lineas.sort(key=lambda l: np.mean([c['center_y'] for c in l]))

    texto_final = ""
    img_debug = img.copy()

    for i, linea in enumerate(lineas):
        texto_linea = ""
        for char in linea:
            pred = clf.predict(char['roi_clasificador'])
            letra = clf.label2char(pred)
            texto_linea += letra

            if debug:
                x, y, w, h = char['x'], char['y'], char['w'], char['h']
                cv2.rectangle(img_debug, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img_debug, letra, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        texto_final += texto_linea + ("+" if i < len(lineas) - 1 else "")

    if debug:
        cv2.imshow("2. Cajas OCR", img_debug)
        print(f" -> [DEBUG] Texto final OCR: {texto_final}")
        cv2.waitKey(0)

    return texto_final

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector',   type=str, default="LdaNormalBayes")
    parser.add_argument('--train_path', default="./train_ocr")
    parser.add_argument('--test_path',  default="./IMÁGENES_PANELES/test_ocr_panels")
    args = parser.parse_args()

    print("Entrenando el OCR...")
    train_dict = cargar_datos(args.train_path, img_size=(25, 25))
    clf = LdaNormalBayesClassifier()
    clf.train(train_dict)

    fichero_salida = os.path.join(args.test_path, "resultado.txt")
    print("\nProcesando paneles... ¡PULSA ESPACIO PARA PASAR DE IMAGEN!")
    
    with open(fichero_salida, "w", encoding="utf-8") as f_out:
        for root, _, archivos in sorted(os.walk(args.test_path)):
            for archivo in sorted(archivos):
                if archivo.lower().endswith((".png", ".jpg", ".jpeg")):
                    filepath = os.path.join(root, archivo)
                    img_array = np.fromfile(filepath, np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                    if img is not None:
                        print(f"\n--- Analizando: {archivo} ---")
                        texto_reconocido = leer_panel(img, clf, debug=True)
                        h, w = img.shape[:2]
                        f_out.write(f"{archivo};0;0;{w};{h};PANEL;1.0;{texto_reconocido}\n")

    cv2.destroyAllWindows()
    print(f"\n¡Resultados guardados exitosamente en {fichero_salida}!")