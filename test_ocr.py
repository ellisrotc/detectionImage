import cv2
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
img = cv2.imread(r"C:\Users\Nell Modal\Desktop\py\images\rayen.png")

# Ajuste fino de la ROI para asegurarnos de que el "1" inicial entra
roi = img[130:800, 105:145]

# Usamos escala de grises y un threshold fijo en lugar de adaptativo u Otsu
# para asegurarnos de que la anti-aliasing de la fuente no borre el 11
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

# Todo lo que sea mas oscuro que 200 (fondo gris es ~230) sera 0 (negro), resto 255 (blanco)
_, bin_img = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# Redimensionamos x3 para separar los digitos pegados como el "11"
resized = cv2.resize(bin_img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

# Aplicar un poco de Blur para suavizar el escalado x3 y luego umbral otra vez para bordes limpios
blur = cv2.GaussianBlur(resized, (5, 5), 0)
_, final_bin = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)

padded = cv2.copyMakeBorder(final_bin, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)

custom_config = r"--psm 6 -c tessedit_char_whitelist=0123456789"
data = pytesseract.image_to_data(padded, config=custom_config, output_type=pytesseract.Output.DICT)

results = []
for i in range(len(data["text"])):
    text = data["text"][i].strip()
    if text:
        results.append(text)

print("Resultados umbral fijo x3:", results)
