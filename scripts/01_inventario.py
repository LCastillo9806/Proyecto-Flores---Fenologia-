"""Paso 1: inventario de fotos, asignación de partición por grupo y copia normalizada.

Uso: python -I 01_inventario.py config.yaml
Salida: work/imagenes/<grupo>__<foto>.jpg (ancho = ancho_ref) y work/inventario.csv
"""
import csv, os, sys
from pathlib import Path

import yaml
from PIL import Image, ImageOps

cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
raiz, raw, ancho = Path(cfg["raiz"]), Path(cfg["raw"]), cfg["ancho_ref"]
out = raiz / "work" / "imagenes"
out.mkdir(parents=True, exist_ok=True)

def particion(grupo):
    for k in ("test", "val"):
        if grupo in cfg["particion"][k]:
            return k
    return "train"

filas = []
for d in sorted(p for p in raw.iterdir() if p.is_dir()):
    fotos = sorted({f.resolve() for f in d.iterdir() if f.suffix.lower() in (".jpg", ".jpeg")})
    for f in fotos:
        nombre = f"{d.name}__{f.stem}.jpg"
        destino = out / nombre
        im = Image.open(f)
        w0, h0 = im.size
        escala = ancho / w0
        if not destino.exists():
            im = ImageOps.exif_transpose(im).convert("RGB")
            if escala != 1:
                im = im.resize((ancho, round(h0 * escala)), Image.LANCZOS)
            im.save(destino, quality=92)
        partes = d.name.split("_")
        filas.append(dict(imagen=nombre, grupo=d.name, fecha="/".join(partes[:3]),
                          bloque=partes[3], variedad="vainilla" if "vainilla" in d.name.lower() else "freedom",
                          ancho_orig=w0, alto_orig=h0, escala=round(escala, 4), particion=particion(d.name)))
    print(f"{d.name}: {len(fotos)} fotos")

with open(raiz / "work" / "inventario.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(filas[0]))
    w.writeheader(); w.writerows(filas)

from collections import Counter
print("Total:", len(filas), Counter(f["particion"] for f in filas))
