import argparse
import os
import cv2
import numpy as np
import random
from collections import defaultdict

# Importamos nuestro detector de la Práctica 1
from detector_paneles import DetectorPaneles

# Importamos nuestro clasificador de la Práctica 2
from lda_normal_bayes_classifier import LdaNormalBayesClassifier

# Motor de OCR 

def pad_to_square(img_gray):
    # Al hacer el resize a 25x25 las letras delgadas (como la 'I' o la 'l') para que  
    # no se achaten ni se deformen, le metemos bordes negros hasta que sean un cuadrado.
    h, w = img_gray.shape
    if h == w: return img_gray
    elif h > w:
        pad_left = (h - w) // 2
        pad_right = h - w - pad_left
        return cv2.copyMakeBorder(img_gray, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    else:
        pad_top = (w - h) // 2
        pad_bottom = w - h - pad_top
        return cv2.copyMakeBorder(img_gray, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=0)

def extract_roi_with_margin(img, x, y, w, h, margin_ratio=0.08):
    # Le damos margen a la caja de la letra para no cortar los bordes difuminados
    margin = int(h * margin_ratio) + 1
    x1, y1 = max(0, x - margin), max(0, y - margin)
    x2, y2 = min(img.shape[1], x + w + margin), min(img.shape[0], y + h + margin)
    return img[y1:y2, x1:x2]

def cargar_datos(ruta, img_size=(25, 25)):
    # Cargamos las letras de la carpeta train_ocr para enseñar al modelo
    images_dict = defaultdict(list)
    for root, _, archivos in os.walk(ruta):
        for archivo in archivos:
            if archivo.endswith(".png"):
                filepath = os.path.join(root, archivo)
                char_class = os.path.basename(root)
                img_gray = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                
                if img_gray is not None:
                    # Si el fondo es blanco y la letra negra, lo invertimos (queremos letra blanca)
                    bg_intensity = np.median([img_gray[0,0], img_gray[0,-1], img_gray[-1,0], img_gray[-1,-1]])
                    if bg_intensity > 127: img_gray = 255 - img_gray
                        
                    # Binarizamos y pillamos el contorno de la letra
                    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        c = max(cnts, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(c)
                        if w > 0 and h > 0:
                            # Recortamos, cuadramos con padding y guardamos
                            roi_gray = extract_roi_with_margin(img_gray, x, y, w, h)
                            roi_gray = cv2.normalize(roi_gray, None, 0, 255, cv2.NORM_MINMAX)
                            roi_padded = pad_to_square(roi_gray)
                            roi_25x25 = cv2.resize(roi_padded, img_size, interpolation=cv2.INTER_AREA)
                            images_dict[char_class].append(roi_25x25)
    return dict(images_dict)

def detectar_mascara_panel(img):
    # Sacamos una máscara azul para saber dónde está el panel y no buscar letras fuera del mismo
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([80, 40, 40]), np.array([150, 255, 255]))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(mask)
    if cnts:
        # Rellenamos el panel entero para no tener huecos y lo erosionamos un poco
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(mask_filled, [c], -1, 255, thickness=cv2.FILLED)
        erosion_px = max(5, int(min(img.shape[:2]) * 0.020))
        k_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (erosion_px, erosion_px))
        return cv2.erode(mask_filled, k_erode)
    return mask

def nms_caracteres(caracteres, iou_thresh=0.20):
    # Supresión de no máximos: si hay dos cajas detectando la misma letra, nos quedamos con la mejor
    if not caracteres: return []
    caracteres = sorted(caracteres, key=lambda c: c['w'] * c['h'], reverse=True)
    keep = []
    for c1 in caracteres:
        suprimido = False
        area1 = c1['w'] * c1['h']
        for c2 in keep:
            area2 = c2['w'] * c2['h']
            inter = max(0, min(c1['x']+c1['w'], c2['x']+c2['w']) - max(c1['x'], c2['x'])) * \
                    max(0, min(c1['y']+c1['h'], c2['y']+c2['h']) - max(c1['y'], c2['y']))
            if inter > 0:
                iou = inter / (area1 + area2 - inter + 1e-6)
                if iou > iou_thresh or (inter/(area1+1e-6)) > 0.60 or (inter/(area2+1e-6)) > 0.60:
                    suprimido = True
                    break
        if not suprimido: keep.append(c1)
    return keep

def procesar_panel(img_panel, clf):
    
    # Función del OCR. Coge la foto del panel recortado y devuelve el texto que lee.

    img_h, img_w = img_panel.shape[:2]
    img_area = img_h * img_w
    mask_panel = detectar_mascara_panel(img_panel)
    tiene_panel = cv2.countNonZero(mask_panel) > (img_area * 0.05)

    # Ecualizamos para mejorar el contraste de las letras (ideal si la foto está oscura o borrosa)
    gray = cv2.cvtColor(img_panel, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    r_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(cv2.split(img_panel)[2])

    min_h_ref, max_h_ref = max(8, int(img_h * 0.015)), int(img_h * 0.50)
    min_area_ref, max_area_ref = max(15, img_area * 0.00008), img_area * 0.10

    def contar_validos(thresh_img):
        # Cuenta cuántos contornos parecen letras de verdad usando proporciones y área
        cnts, _ = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        validos = 0
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if h == 0: continue
            area_c = cv2.contourArea(c)
            hull_area = cv2.contourArea(cv2.convexHull(c))
            if hull_area == 0: continue
            # Filtros de forma (que no sea una línea ni un cuadro muy grande) y solidez
            if (min_area_ref < w*h < max_area_ref) and (0.08 < w/float(h) < 1.4) and (min_h_ref < h < max_h_ref):
                if (area_c / float(w * h)) > 0.28: validos += 1
        return validos

    # Probamos varios umbrales y nos quedamos con el que saque más "letras"
    candidatos = []
    _, t_otsu_r = cv2.threshold(r_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidatos.append((contar_validos(t_otsu_r), t_otsu_r))
    t_adp_1 = cv2.adaptiveThreshold(gray_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, -5)
    candidatos.append((contar_validos(t_adp_1), t_adp_1))
    candidatos.sort(key=lambda x: x[0], reverse=True)
    mejor_thresh = candidatos[0][1]

    #Sacamos las cajas definitivas con el mejor umbral
    contours, _ = cv2.findContours(mejor_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    caracteres_raw = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h == 0: continue
        cx, cy = int(x + w/2), int(y + h/2)

        #Descartamos detecciones erróneas: fuera del panel azul o proporciones raras
        if tiene_panel and mask_panel[cy, cx] == 0: continue
        if not (min_area_ref < w*h < max_area_ref) or not (0.08 < w/float(h) < 1.4) or not (min_h_ref < h < max_h_ref): continue

        #Descartamos flechas y marcos basándonos en si son formas con huecos que no tendrían las letras a detectar
        area_c = cv2.contourArea(c)
        hull_area = cv2.contourArea(cv2.convexHull(c))
        if hull_area == 0 or (area_c / float(w * h)) < 0.28 or (area_c / float(hull_area)) < 0.38: continue 

        roi_gray = extract_roi_with_margin(gray, x, y, w, h)
        if roi_gray.size == 0 or roi_gray.std() < 10: continue

        roi_gray = cv2.normalize(roi_gray, None, 0, 255, cv2.NORM_MINMAX)
        roi_25x25 = cv2.resize(pad_to_square(roi_gray), (25, 25), interpolation=cv2.INTER_AREA)

        caracteres_raw.append({'x': x, 'y': y, 'w': w, 'h': h, 'roi': roi_25x25, 'cx': cx, 'cy': cy})

    if not caracteres_raw: return "", []

    #Eliminamos cajas muy grandes que se nos hayan colado comparando con la mediana de altura
    mediana_h = np.median([c['h'] for c in caracteres_raw])
    caracteres = nms_caracteres([c for c in caracteres_raw if 0.5 * mediana_h < c['h'] < 1.6 * mediana_h])
    if not caracteres: return "", []

    #RANSAC: trazamos rectas iterativamente para agrupar las letras por líneas
    lineas, chars_restantes = [], caracteres.copy()
    distancia_max = max(5.0, mediana_h * 0.40)

    while len(chars_restantes) >= 2:
        mejor_inliers = []
        for _ in range(500): #500 intentos de RANSAC
            p1, p2 = random.sample(chars_restantes, 2)
            dx = p2['cx'] - p1['cx']
            if abs(dx) < 1: continue
            m = (p2['cy'] - p1['cy']) / dx
            if abs(m) > 0.35: continue #Las líneas del texto no suelen estar muy torcidas
            
            c_recta = p1['cy'] - m * p1['cx']
            inliers = [c for c in chars_restantes if abs(m * c['cx'] - c['cy'] + c_recta) / np.sqrt(m**2 + 1) < distancia_max]
            if len(inliers) > len(mejor_inliers): mejor_inliers = inliers

        if len(mejor_inliers) < 2: break
        lineas.append(mejor_inliers)
        chars_restantes = [c for c in chars_restantes if c not in mejor_inliers]

    lineas.extend([[c] for c in chars_restantes])

    #Si una "letra" está muy lejos de cualquier otra (muy aislada),
    #suele ser una flecha o ruido que coló RANSAC y la borramos
    lineas_limpias = []
    for linea in lineas:
        linea.sort(key=lambda c: c['cx'])
        if len(linea) == 1: continue 
        linea_filtrada = []
        for i, char in enumerate(linea):
            dist_izq = (char['cx'] - linea[i-1]['cx']) if i > 0 else float('inf')
            dist_der = (linea[i+1]['cx'] - char['cx']) if i < len(linea)-1 else float('inf')
            if min(dist_izq, dist_der) < char['h'] * 4.0:
                linea_filtrada.append(char)
        if len(linea_filtrada) >= 2: lineas_limpias.append(linea_filtrada)

    lineas_limpias.sort(key=lambda l: np.mean([c['cy'] for c in l]))

    # Pasamos las imágenes ya limpias y ordenadas por el clasificador
    texto_final = ""
    for i, linea in enumerate(lineas_limpias):
        for char in linea:
            char['letra'] = clf.label2char(clf.predict(char['roi']))
            texto_final += char['letra']
        if i < len(lineas_limpias) - 1: texto_final += "+" # Salto de línea

    return texto_final, lineas_limpias

#Modulo principal
if __name__ == "__main__":
    #Cogemos los argumentos por consola
    parser = argparse.ArgumentParser(description="Sistema Integrado: Detector + OCR (Ejercicio 4)")
    parser.add_argument('--test_path', default="test_detection", help='Directorio de imágenes enteras')
    parser.add_argument('--train_ocr_path', default="train_ocr", help='Directorio de entrenamiento del OCR')
    parser.add_argument('--visualize_ocr', action='store_true', help='Activar visualización paso a paso')
    args = parser.parse_args()

    # 1.Inicializamos detector
    print("Inicializando detector de paneles")
    detector = DetectorPaneles() 

    # 2.Inicializar y entrenar OCR
    print("Entrenando modelo OCR")
    train_dict = cargar_datos(args.train_ocr_path)
    ocr_model = LdaNormalBayesClassifier()
    ocr_model.train(train_dict)

    print(f"Cargando imágenes de autopista desde: {args.test_path}")
    if os.path.exists(args.test_path):
        imagenes_test = [f for f in sorted(os.listdir(args.test_path)) if f.lower().endswith(('.png', '.jpg'))]
    else:
        print(f"Error: la ruta '{args.test_path}' no existe."); exit()

    os.makedirs("resultado_imgs", exist_ok=True)
    with open("resultado.txt", "w") as f_txt:
        print("\nProcesando lectura de paneles")

        # Bucle principal sobre todas las fotos
        for nombre_img in imagenes_test:
            img = cv2.imread(os.path.join(args.test_path, nombre_img))
            if img is None: continue

            # Detectamos paneles azules
            detecciones_paneles = detector.detectar(img)

            for det in detecciones_paneles:
                x1, y1, x2, y2 = det['box']
                score = det.get('score', 1.0)
                
                # Recortamos el panel asegurando no salirnos del tamaño de la imagen
                x1_c, y1_c = max(0, int(x1)), max(0, int(y1))
                x2_c, y2_c = min(img.shape[1], int(x2)), min(img.shape[0], int(y2))
                panel_crop = img[y1_c:y2_c, x1_c:x2_c]
                
                if panel_crop.size == 0: continue

                # Pasamos el panel recortado al OCR
                texto_panel, lineas_chars = procesar_panel(panel_crop, ocr_model)

                # Guardamos en el txt
                f_txt.write(f"{nombre_img};{x1};{y1};{x2};{y2};1;{score:.3f};{texto_panel}\n")

                # Pintamos todo si el flag está activado
                if args.visualize_ocr:
                    cv2.rectangle(img, (x1_c, y1_c), (x2_c, y2_c), (0, 0, 255), 2)
                    for linea in lineas_chars:
                        if len(linea) > 1:
                            for i in range(len(linea)-1):
                                # Hay que sumar x1_c e y1_c para dibujar en las coordenadas de la imagen
                                pt1 = (int(linea[i]['cx'] + x1_c), int(linea[i]['cy'] + y1_c))
                                pt2 = (int(linea[i+1]['cx'] + x1_c), int(linea[i+1]['cy'] + y1_c))
                                cv2.line(img, pt1, pt2, (255, 0, 0), 2)
                        
                        for d in linea:
                            lx, ly, lw, lh = int(d['x'] + x1_c), int(d['y'] + y1_c), int(d['w']), int(d['h'])
                            cv2.rectangle(img, (lx, ly), (lx+lw, ly+lh), (0, 255, 0), 1)
                            cv2.putText(img, d['letra'], (lx, ly - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imwrite(os.path.join("resultado_imgs", nombre_img), img)
            print(f" -> {nombre_img}: {len(detecciones_paneles)} paneles procesados.")

            if args.visualize_ocr:
                cv2.imshow("Sistema Integrado", img)
                if cv2.waitKey(0) == 27: args.visualize_ocr = False # Esc para salir del bucle de ver imgs

    if args.visualize_ocr: cv2.destroyAllWindows()
    print("\nProceso completado")