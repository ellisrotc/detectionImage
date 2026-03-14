import argparse
import os
from typing import List, Tuple

import cv2
import numpy as np
import pytesseract


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detecta numeros en la columna derecha de una captura usando OpenCV + Tesseract"
    )
    parser.add_argument(
        "--img",
        required=True,
        help="Ruta de la imagen (screenshot) del programa Raynen",
    )
    parser.add_argument(
        "--tesseract",
        default=None,
        help="Ruta opcional al ejecutable de tesseract.exe si no esta en PATH",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Muestra ventanas con los pasos intermedios",
    )
    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        help="Recorte manual: coordenadas x y ancho alto. Evita la ventana selectROI",
    )
    return parser.parse_args()


def select_roi(img: np.ndarray, coords: Tuple[int, int, int, int] | None) -> np.ndarray:
    if coords:
        x, y, w, h = coords
    else:
        try:
            roi = cv2.selectROI("Selecciona la columna", img, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow("Selecciona la columna")
            x, y, w, h = map(int, roi)
        except cv2.error as e:
            raise RuntimeError(
                "selectROI no esta disponible en esta build de OpenCV. Usa --roi X Y W H para recortar manualmente."
            ) from e
    if w == 0 or h == 0:
        raise ValueError("No se selecciono ninguna region")
    return img[y : y + h, x : x + w]


def preprocess(roi: np.ndarray) -> np.ndarray:
    # Convertimos a escala de grises para detectar tanto rojo como negro
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # El fondo es gris claro (>200 aprox), el texto rojo y negro es mas oscuro (<200)
    # THRESH_BINARY convierte lo oscuro a 0 (negro) y lo claro a 255 (blanco)
    # Tesseract prefiere texto negro sobre fondo blanco
    _, bin_img = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    return bin_img


def extract_digits(bin_img: np.ndarray) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    # Redimensionamos x3 para separar los digitos pegados como el "11"
    resized = cv2.resize(bin_img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    
    # Aplicamos Blur para suavizar los bordes del escalado y binarizamos de nuevo
    blur = cv2.GaussianBlur(resized, (5, 5), 0)
    _, final_bin = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
    
    # Añadimos un borde blanco para que Tesseract trabaje mejor
    padded = cv2.copyMakeBorder(final_bin, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
    
    # Configuracion: psm 6 asume un bloque uniforme de texto (columna de numeros)
    custom_config = r"--psm 6 -c tessedit_char_whitelist=0123456789"
    
    # Obtenemos los datos detallados (incluyendo coordenadas)
    data = pytesseract.image_to_data(padded, config=custom_config, output_type=pytesseract.Output.DICT)
    
    results = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if text:
            # Revertimos coordenadas por el padding y el escalado x3 para dibujar sobre el roi original
            x = (data["left"][i] - 30) // 3
            y = (data["top"][i] - 30) // 3
            w = data["width"][i] // 3
            h = data["height"][i] // 3
            
            if w > 0 and h > 0:
                results.append((text, (x, y, w, h)))
            
    return results



def main():
    args = parse_args()
    if args.tesseract:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract

    if not os.path.exists(args.img):
        raise FileNotFoundError(args.img)

    img = cv2.imread(args.img)
    if img is None:
        raise ValueError("No pude leer la imagen")

    roi = select_roi(img, tuple(args.roi) if args.roi else None)
    bin_img = preprocess(roi)
    digits = extract_digits(bin_img)

    overlay = roi.copy()
    for text, (x, y, w, h) in digits:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(
            overlay,
            text,
            (x, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    ordered_digits = [d for d, _ in digits]
    print("Numeros detectados (arriba a abajo):", ordered_digits)

    cv2.imshow("ROI", roi)
    cv2.imshow("Binario", bin_img)
    cv2.imshow("Detecciones", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
