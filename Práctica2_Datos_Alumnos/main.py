import argparse
import os
import cv2
import numpy as np
import random
from collections import defaultdict

# Importamos tu detector de la Práctica 1
from detector_paneles import DetectorPaneles
# Importamos el clasificador ganador de la Práctica 2
from clasificadores_alternativos import LdaRandomForestClassifier

def agrupar_en_lineas(detecciones):
    """Agrupa caracteres en líneas según su posición vertical (coordenada Y)."""
    if not detecciones: return []
    # Ordenamos por Y para procesar de arriba a abajo
    detecciones.sort(key=lambda d: d['y'])
    lineas = []
    while detecciones:
        base = detecciones.pop(0)
        linea_actual = [base]
        restantes = []
        for d in detecciones:
            # Tolerancia del 70% de la altura para agrupar en el mismo renglón
            if abs(d['centro_y'] - base['centro_y']) < (base['h'] * 0.7):
                linea_actual.append(d)
            else:
                restantes.append(d)
        # Ordenamos la línea de izquierda a derecha
        linea_actual.sort(key=lambda d: d['x'])
        lineas.append(linea_actual)
        detecciones = restantes
    return lineas

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integración Final: Detector + OCR (Ejercicio 4)")
    parser.add_argument('--test_path', default="test_detection", help='Fotos de la carretera')
    parser.add_argument('--train_ocr_path', default="train_ocr", help='Datos para entrenar el OCR')
    parser.add_argument('--visualize_ocr', action='store_true', help='Activar visor paso a paso')
    args = parser.parse_args()

    # 1. INICIALIZACIÓN
    print(">>> Cargando Detector (MSER)...")
    detector = DetectorPaneles() 

    print(">>> Entrenando OCR (Random Forest)...")
    images_dict = defaultdict(list)
    for root, _, archivos in os.walk(args.train_ocr_path):
        char_class = os.path.basename(root)
        if len(char_class) == 1:
            for f in archivos:
                if f.endswith(".png"):
                    img_char = cv2.imread(os.path.join(root, f))
                    if img_char is not None: images_dict[char_class].append(img_char)

    ocr_model = LdaRandomForestClassifier(ocr_char_size=(25, 25))
    ocr_model.train(dict(images_dict))

    # 2. PROCESAMIENTO DE IMÁGENES
    if not os.path.exists(args.test_path):
        print(f"Error: No existe la carpeta {args.test_path}"); exit()

    imagenes_test = [f for f in sorted(os.listdir(args.test_path)) if f.lower().endswith(('.png', '.jpg'))]
    os.makedirs("resultado_imgs", exist_ok=True)

    with open("resultado.txt", "w") as f_out:
        for nombre_img in imagenes_test:
            ruta_completa = os.path.join(args.test_path, nombre_img)
            img_orig = cv2.imread(ruta_completa)
            if img_orig is None: continue

            # Paso A: Buscar paneles (Práctica 1)
            detecciones_paneles = detector.detectar(img_orig)
            img_visual = img_orig.copy()

            for det in detecciones_paneles:
                x1, y1, x2, y2 = [int(v) for v in det['box']]
                score = det.get('score', 1.0)

                # Paso B: Recortar panel (con seguridad de bordes)
                h_o, w_o = img_orig.shape[:2]
                rx1, ry1 = max(0, x1), max(0, y1)
                rx2, ry2 = min(w_o, x2), min(h_o, y2)
                panel_crop = img_orig[ry1:ry2, rx1:rx2]
                
                if panel_crop.size == 0: continue

                # Paso C: OCR del panel (Inversión y Detección de letras)
                gray = cv2.cvtColor(panel_crop, cv2.COLOR_BGR2GRAY)
                # Invertimos: el azul oscuro pasa a ser claro para que el OCR lea bien
                gray_inv = cv2.bitwise_not(gray) 
                
                thresh = cv2.adaptiveThreshold(gray_inv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY_INV, 11, 2)
                
                cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                letras_panel = []
                h_p, w_p = panel_crop.shape[:2]

                for c in cnts:
                    px, py, pw, ph = cv2.boundingRect(c)
                    # Filtros de ruido
                    if px < 4 or py < 4 or (px+pw) > (w_p-4) or (py+ph) > (h_p-4): continue
                    if ph >= 10 and 30 < (pw*ph) < (h_p*w_p*0.1) and 0.2 < (ph/float(pw)) < 5.0:
                        
                        # Padding para que el modelo no se asuste con recortes pegados
                        pad = 5
                        y_p1, y_p2 = max(0, py-pad), min(h_p, py+ph+pad)
                        x_p1, x_p2 = max(0, px-pad), min(w_p, px+pw+pad)
                        
                        roi_para_clasificar = gray_inv[y_p1:y_p2, x_p1:x_p2]
                        
                        char_pred = ocr_model.label2char(ocr_model.predict(roi_para_clasificar))
                        
                        letras_panel.append({
                            'x': px, 'y': py, 'w': pw, 'h': ph, 
                            'centro_x': px + pw//2, 'centro_y': py + ph/2.0, 
                            'txt': char_pred
                        })

                # Paso D: Agrupar y Generar String
                lineas = agrupar_en_lineas(letras_panel)
                texto_final = "+".join(["".join([l['txt'] for l in lin]) for lin in lineas])

                # Guardar en resultado.txt (Formato Ejercicio 4)
                f_out.write(f"{nombre_img};{x1};{y1};{x2};{y2};1;{score:.3f};{texto_final}\n")

                # Paso E: Visualización (Requisitos del PDF)
                if args.visualize_ocr:
                    # Rectángulo del panel (Rojo)
                    cv2.rectangle(img_visual, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    for lin in lineas:
                        # Líneas azules uniendo caracteres del mismo renglón
                        if len(lin) > 1:
                            for idx in range(len(lin)-1):
                                p1 = (int(lin[idx]['centro_x'] + x1), int(lin[idx]['centro_y'] + y1))
                                p2 = (int(lin[idx+1]['centro_x'] + x1), int(lin[idx+1]['centro_y'] + y1))
                                cv2.line(img_visual, p1, p2, (255, 0, 0), 2)
                        
                        # Cajas verdes y predicción roja
                        for letra in lin:
                            lx, ly = letra['x'] + x1, letra['y'] + y1
                            cv2.rectangle(img_visual, (lx, ly), (lx+letra['w'], ly+letra['h']), (0, 255, 0), 1)
                            cv2.putText(img_visual, letra['txt'], (lx, ly-5), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.imwrite(os.path.join("resultado_imgs", nombre_img), img_visual)
            print(f" -> {nombre_img} procesada.")

            if args.visualize_ocr:
                cv2.imshow("Ejercicio 4: Sistema Completo", img_visual)
                if cv2.waitKey(0) == 27: args.visualize_ocr = False

    cv2.destroyAllWindows()
    print("\n>>> ¡Práctica completada! Revisa resultado.txt y la carpeta resultado_imgs.")