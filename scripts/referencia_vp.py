"""Construye la imagen de referencia (rejilla de recortes) y las cajas de ejemplo para YOLOE."""
from pathlib import Path
import cv2, numpy as np

def construir(cfg, celda=256, columnas=4):
    work = Path(cfg["raiz"]) / "work" / "imagenes"
    ej = [(n, *e) for n, lst in cfg["ejemplos"].items() for e in lst]
    filas = (len(ej) + columnas - 1) // columnas
    ref = np.zeros((filas * celda, columnas * celda, 3), np.uint8)
    cajas, clases, cache = [], [], {}
    for i, (n, c, x, y, s) in enumerate(ej):
        img = cache.setdefault(n, cv2.imread(str(work / n)))
        x0, y0 = int(x - celda / 2), int(y - celda / 2)
        oy, ox = (i // columnas) * celda, (i % columnas) * celda
        ref[oy:oy + celda, ox:ox + celda] = img[y0:y0 + celda, x0:x0 + celda]
        h = s / 2
        cajas.append([ox + celda / 2 - h, oy + celda / 2 - h, ox + celda / 2 + h, oy + celda / 2 + h])
        clases.append(c)
    return ref, np.array(cajas, float), np.array(clases)
