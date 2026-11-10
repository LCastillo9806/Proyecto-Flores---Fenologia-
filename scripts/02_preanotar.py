"""Paso 2: preanotación automática de botones/flores con 4 estados (PROPUESTA para corregir a mano).

Modo 'cero' (sin modelo propio todavía): candidatos de dos fuentes, fusionados
  1) YOLOE-26 con indicaciones visuales (12 recortes de ejemplo de fotos de train, ver config 'ejemplos').
  2) Manchas compactas de color de pétalo (rojo Freedom; blanco Vainilla con filtro más estricto).
  El estado se asigna por reglas de color y tamaño.
Modo 'semilla' (--modelo ruta.pt): usa un YOLO26 de 4 clases ya entrenado con lo corregido.

Uso: python -I 02_preanotar.py config.yaml [--solo N] [--modelo best.pt] [--particion train]
Salida: work/preanot/<imagen>.txt (YOLO: clase cx cy w h) y work/preanot_vis/ para revisar.
"""
import argparse, csv, sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torchvision.ops import nms

sys.path.insert(0, str(Path(__file__).parent))
from referencia_vp import construir  # noqa: E402
from balance_blancos import balancear  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("config"); ap.add_argument("--solo", type=int, default=0)
ap.add_argument("--modelo", default=None); ap.add_argument("--particion", nargs="*")
ap.add_argument("--conf", type=float, default=None)
ap.add_argument("--variedad", default=None, help="procesar solo freedom o vainilla")
ap.add_argument("--imagenes", nargs="*", help="procesar solo estas fotos")
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
raiz = Path(cfg["raiz"]); work = raiz / "work"
EST = cfg["estados"]                       # garbanzo, rayando_color, punto_corte, abierta
T, S = cfg["tile"], int(cfg["tile"] * (1 - cfg["solape"]))

# Tamaños en píxeles a ancho_ref = 8064 (calibrados mirando las fotos)
LADO_MIN, LADO_MAX = 25, 260               # caja de botón/flor plausible
LADO_ABIERTA = 150                         # caja con color mayor a esto -> flor abierta
FRAC_RAYANDO, FRAC_COLOR = 0.05, 0.30      # fracción de píxeles de pétalo en la caja
FRAC_BOTON = 0.12                          # fracción mínima de "color de botón" en el centro de la caja

if a.modelo:
    from ultralytics import YOLO
    modelo = YOLO(a.modelo); conf = a.conf or 0.2
else:
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
    modelo = YOLOE(str(raiz / "pesos" / "yoloe-26l-seg.pt")); conf = a.conf or 0.03
    REF, REF_CAJAS, REF_CLS = construir(cfg)


def mascara_petalo(hsv, variedad, amplia=False):
    """amplia=True: todo el rojo (para buscar manchas, que luego se filtran por matiz mediano).
    amplia=False: solo carmesí, para medir pétalo dentro de cajas que propone YOLOE sin que cuente el
    follaje nuevo rojo anaranjado."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    if variedad == "freedom":
        if amplia:
            return ((h <= 8) | (h >= 160)) & (s >= 110) & (v >= 45)
        return (h >= 158) & (h <= 178) & (s >= 110) & (v >= 45)
    return (s <= 45) & (v >= 200)                                # blanco / crema


def mascara_boton(hsv, variedad):
    """Pétalo o cáliz verde-amarillento claro (los botones son más claros y amarillos que las hojas)."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return mascara_petalo(hsv, variedad) | ((h >= 22) & (h <= 38) & (s >= 70) & (v >= 150))


def anillo(hsv, x, y, w, hh, px):
    """Fracción de verde y de rojo anaranjado en un anillo alrededor de la mancha (px = máscara de la mancha)."""
    L = max(w, hh); x0, y0 = max(x - L // 2, 0), max(y - L // 2, 0)
    reg = hsv[y0:y + hh + L // 2, x0:x + w + L // 2].astype(int)
    fuera = np.ones(reg.shape[:2], bool); fuera[y - y0:y - y0 + hh, x - x0:x - x0 + w] &= ~px
    H, S, V = reg[..., 0][fuera], reg[..., 1][fuera], reg[..., 2][fuera]
    return ((H >= 25) & (H <= 90) & (S >= 40)).mean(), ((H <= 14) & (S >= 90) & (V >= 60)).mean()


def candidatos_color(img, variedad):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = mascara_petalo(hsv, variedad, amplia=True).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    _, lab, st, _ = cv2.connectedComponentsWithStats(m)
    area_min = 450 if variedad == "freedom" else 1500           # los brillos de hoja imitan pétalos blancos
    out = []
    for i, (x, y, w, h, area) in enumerate(st[1:], start=1):
        lado = max(w, h)
        # forma compacta (descarta tutores, cuerdas y plástico, que son alargados)
        if lado < 30 or lado > LADO_MAX or area < area_min or min(w, h) / lado < 0.45 or area / (w * h) < 0.4:
            continue
        if variedad == "freedom":
            # Tras el balance de blancos, el pétalo es carmesí-vino (matiz mediano <= 3) y el follaje nuevo
            # rojo anaranjado (> 3): en 48 manchas de train conserva 10/10 botones y quita ~70 % de brotes.
            # Además, un pétalo caído está rodeado de tierra (poco verde) y una punta de brote, de hoja naranja.
            px = lab[y:y + h, x:x + w] == i
            hh = hsv[y:y + h, x:x + w, 0][px].astype(int)
            if np.median(np.where(hh >= 90, hh - 180, hh)) > 3:
                continue
            verde, naranja = anillo(hsv, x, y, w, h, px)
            if verde < 0.20 or naranja >= 0.25:
                continue
        p = 0.15 * lado                                          # ampliar para incluir los sépalos
        out.append([x - p, y - p, x + w + p, y + h + p, 0.5, 1])   # fuente 1 = mancha de pétalo
    return out


def mosaicos(W, H):
    xs = list(range(0, max(W - T, 0) + 1, S)); ys = list(range(0, max(H - T, 0) + 1, S))
    if xs[-1] + T < W: xs.append(W - T)
    if ys[-1] + T < H: ys.append(H - T)
    return [(x, y) for y in ys for x in xs]


def candidatos_modelo(img):
    H, W = img.shape[:2]; tiles = mosaicos(W, H); out = []
    for i in range(0, len(tiles), 8):
        lote = tiles[i:i + 8]; recs = [img[y:y + T, x:x + T] for x, y in lote]
        if a.modelo:
            res = modelo.predict(recs, imgsz=T, conf=conf, verbose=False)
        else:
            res = modelo.predict(recs, refer_image=REF, visual_prompts=dict(bboxes=REF_CAJAS, cls=REF_CLS),
                                 predictor=YOLOEVPSegPredictor, imgsz=T, conf=conf, iou=0.5, verbose=False)
        for (x, y), r in zip(lote, res):
            for b, c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                out.append([b[0] + x, b[1] + y, b[2] + x, b[3] + y, float(c), int(k)])
    return out


def recorte(img, c, frac=1.0):
    x1, y1, x2, y2 = c[:4]; cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * frac, (y2 - y1) * frac
    return img[max(int(cy - h / 2), 0):int(cy + h / 2), max(int(cx - w / 2), 0):int(cx + w / 2)]


def estado(img, c, variedad):
    rec = recorte(img, c)
    if rec.size == 0:
        return None
    # si la caja nació de una mancha de pétalo ya filtrada, se mide con la máscara amplia
    frac = mascara_petalo(cv2.cvtColor(rec, cv2.COLOR_BGR2HSV), variedad, amplia=bool(c[5])).mean()
    lado = max(c[2] - c[0], c[3] - c[1])
    if frac < FRAC_RAYANDO:
        return 0                                                 # garbanzo (verde)
    if frac < FRAC_COLOR:
        return 1                                                 # rayando color
    return 3 if lado >= LADO_ABIERTA else 2                      # abierta / punto de corte


def quitar_contenidas(t):
    """Elimina cajas casi contenidas en otra de mayor puntaje (botón detectado entero y por partes)."""
    t = t[torch.argsort(t[:, 4], descending=True)]; keep = []
    area = (t[:, 2] - t[:, 0]) * (t[:, 3] - t[:, 1])
    for i in range(len(t)):
        ok = True
        for j in keep:
            ix = (torch.minimum(t[i, 2], t[j, 2]) - torch.maximum(t[i, 0], t[j, 0])).clamp(min=0)
            iy = (torch.minimum(t[i, 3], t[j, 3]) - torch.maximum(t[i, 1], t[j, 1])).clamp(min=0)
            if ix * iy > 0.6 * min(area[i], area[j]):
                ok = False; break
        if ok: keep.append(i)
    return t[keep]


def procesar(img, variedad):
    H, W = img.shape[:2]
    cands = candidatos_modelo(img)
    if a.modelo:      # el modelo semilla ya predice el estado
        t = torch.tensor(cands, dtype=torch.float32).reshape(-1, 6)
        t = quitar_contenidas(t[nms(t[:, :4], t[:, 4], 0.5)])
        return [(int(c[5]), *c[:4]) for c in t.tolist()]
    img = balancear(img)              # el color se analiza con balance de blancos (dominantes entre vuelos)
    buenos = []
    for c in cands:
        lado = max(c[2] - c[0], c[3] - c[1]); corto = min(c[2] - c[0], c[3] - c[1])
        if not (LADO_MIN <= lado <= LADO_MAX):
            continue
        centro = recorte(img, c, 0.5)
        if not centro.size:
            continue
        hsv_c = cv2.cvtColor(centro, cv2.COLOR_BGR2HSV)
        if mascara_boton(hsv_c, variedad).mean() < FRAC_BOTON:
            continue
        # caja sin pétalo = candidato a garbanzo: debe ser pequeña, casi cuadrada y con algo de confianza
        # (las cajas grandes o alargadas sin pétalo son hojas amarillentas)
        if mascara_petalo(hsv_c, variedad).mean() < FRAC_RAYANDO and (lado > 90 or corto / lado < 0.6 or c[4] < 0.05):
            continue
        buenos.append(c[:5] + [0])
    cands = buenos + candidatos_color(img, variedad)
    if not cands:
        return []
    t = torch.tensor(cands, dtype=torch.float32).reshape(-1, 6)
    t = quitar_contenidas(t[nms(t[:, :4], t[:, 4], 0.4)])
    cajas = []
    for c in t.tolist():
        e = estado(img, c, variedad)
        if e is not None:
            cajas.append((e, max(c[0], 0), max(c[1], 0), min(c[2], W), min(c[3], H)))
    return cajas


inv = list(csv.DictReader(open(work / "inventario.csv", encoding="utf-8")))
if a.imagenes:
    inv = [f for f in inv if f["imagen"] in a.imagenes]
if a.variedad:
    inv = [f for f in inv if f["variedad"] == a.variedad]
if a.particion:
    inv = [f for f in inv if f["particion"] in a.particion]
if a.solo:
    inv = inv[::max(1, len(inv) // a.solo)][:a.solo]
(work / "preanot").mkdir(exist_ok=True); (work / "preanot_vis").mkdir(exist_ok=True)
COL = [(0, 200, 0), (0, 200, 255), (0, 0, 255), (255, 0, 255)]
tot = np.zeros(4, int)
for fila in inv:
    img = cv2.imread(str(work / "imagenes" / fila["imagen"])); H, W = img.shape[:2]
    cajas = procesar(img, fila["variedad"])
    vis = img.copy()
    with open(work / "preanot" / (Path(fila["imagen"]).stem + ".txt"), "w") as fh:
        for e, x1, y1, x2, y2 in cajas:
            fh.write(f"{e} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}\n")
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), COL[e], 4)
            tot[e] += 1
    cv2.imwrite(str(work / "preanot_vis" / fila["imagen"]), cv2.resize(vis, (W // 2, H // 2)),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(fila["imagen"], np.bincount([c[0] for c in cajas], minlength=4), flush=True)
print("Total por estado:", dict(zip(EST, tot.tolist())))
