"""Paso 3: construir los 3 datasets binarios (A, B, C) en mosaicos a partir de la anotación de 4 estados.

Fuente de etiquetas (en este orden): work/etiquetas/ (corregidas a mano) o work/preanot/ (automáticas).
Uso: python -I 03_datasets.py config.yaml [--fuente etiquetas|preanot]
Salida: datasets/<modelo>/{images,labels}/{train,val,test} + data.yaml
"""
import argparse, csv, random, shutil
from pathlib import Path

import cv2
import yaml

ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--fuente", default=None)
ap.add_argument("--semilla", action="store_true", help="un solo dataset de 4 estados para el modelo semilla")
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
raiz = Path(cfg["raiz"]); work = raiz / "work"
fuente = a.fuente or ("etiquetas" if any((work / "etiquetas").glob("*.txt")) else "preanot")
print("Fuente de etiquetas:", fuente)
EST = cfg["estados"]; T = cfg["tile"]; S = int(T * (1 - cfg["solape"]))
FRAC_VACIOS = 0.15     # proporción de mosaicos sin objetos que se conservan (fondo)
random.seed(0)

inv = list(csv.DictReader(open(work / "inventario.csv", encoding="utf-8")))


def leer(img_name, W, H):
    p = work / fuente / (Path(img_name).stem + ".txt")
    if not p.exists():
        return None          # foto sin etiquetas de esta fuente: no se usa
    cajas = []
    if True:
        for ln in p.read_text().split("\n"):
            if ln.strip():
                c, cx, cy, w, h = ln.split()
                cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
                cajas.append((int(c), cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return cajas


def posiciones(L):
    ps = list(range(0, max(L - T, 0) + 1, S))
    if ps[-1] + T < L: ps.append(L - T)
    return ps


modelos = {"S_semilla_4estados": {e: [e] for e in EST}} if a.semilla else cfg["modelos"]
for nombre_m, clases in modelos.items():
    dst = raiz / "datasets" / nombre_m
    shutil.rmtree(dst, ignore_errors=True)
    mapa = {EST.index(e): i for i, (_, ests) in enumerate(clases.items()) for e in ests}
    n = {}
    for fila in inv:
        sp = fila["particion"]
        if not (work / fuente / (Path(fila["imagen"]).stem + ".txt")).exists():
            continue
        img = cv2.imread(str(work / "imagenes" / fila["imagen"])); H, W = img.shape[:2]
        cajas = leer(fila["imagen"], W, H)
        if cajas is None:
            continue
        (dst / "images" / sp).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / sp).mkdir(parents=True, exist_ok=True)
        for y in posiciones(H):
            for x in posiciones(W):
                lineas = []
                for c, x1, y1, x2, y2 in cajas:
                    if c not in mapa:
                        continue
                    ix1, iy1, ix2, iy2 = max(x1, x), max(y1, y), min(x2, x + T), min(y2, y + T)
                    if ix2 <= ix1 or iy2 <= iy1:
                        continue
                    # conservar si se ve al menos el 50 % del objeto
                    if (ix2 - ix1) * (iy2 - iy1) < 0.5 * (x2 - x1) * (y2 - y1):
                        continue
                    lineas.append(f"{mapa[c]} {((ix1+ix2)/2-x)/T:.6f} {((iy1+iy2)/2-y)/T:.6f} "
                                  f"{(ix2-ix1)/T:.6f} {(iy2-iy1)/T:.6f}")
                if not lineas and random.random() > FRAC_VACIOS:
                    continue
                base = f"{Path(fila['imagen']).stem}_{x}_{y}"
                cv2.imwrite(str(dst / "images" / sp / f"{base}.jpg"), img[y:y + T, x:x + T],
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                (dst / "labels" / sp / f"{base}.txt").write_text("\n".join(lineas))
                n.setdefault(sp, [0, [0] * len(clases)])
                n[sp][0] += 1
                for ln in lineas:
                    n[sp][1][int(ln.split()[0])] += 1
    yaml.safe_dump({"path": str(dst).replace("\\", "/"), "train": "images/train", "val": "images/val",
                    "test": "images/test", "names": dict(enumerate(clases))},
                   open(dst / "data.yaml", "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
    print(nombre_m, {k: f"{v[0]} mosaicos, objetos por clase {v[1]}" for k, v in n.items()})
