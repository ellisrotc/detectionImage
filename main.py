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
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    bin_img = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 10
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)
    return bin_img


def extract_digits(bin_img: np.ndarray) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h < 10 or h > 80 or w < 5 or w > 80:
            continue
        boxes.append((y, x, w, h))
    boxes.sort()  # arriba -> abajo

    results = []
    for y, x, w, h in boxes:
        digit_roi = bin_img[y : y + h, x : x + w]
        digit_roi = cv2.copyMakeBorder(digit_roi, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
        text = pytesseract.image_to_string(
            digit_roi, config="--psm 10 -c tessedit_char_whitelist=0123456789"
        ).strip()
        if text:
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
