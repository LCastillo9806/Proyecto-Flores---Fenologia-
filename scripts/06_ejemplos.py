"""Paso 6: ejemplos de predicción sobre fotos COMPLETAS de test con SAHI (mosaicos + fusión).

Por cada modelo y foto: imagen reducida con las cajas predichas y un acercamiento a la zona con más detecciones.
Uso: python -I 06_ejemplos.py config.yaml [--n 3]
Salida: runs/ejemplos/<modelo>__<foto>_completa.jpg y _zoom.jpg
"""
import argparse, csv
from pathlib import Path

import cv2
import numpy as np
import yaml
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction



def main():
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    raiz = Path(cfg["raiz"]); work = raiz / "work"; out = raiz / "runs" / "ejemplos"; out.mkdir(parents=True, exist_ok=True)
    test = [f for f in csv.DictReader(open(work / "inventario.csv", encoding="utf-8")) if f["particion"] == "test"]
    # mezclar variedades
    fotos = [f for f in test if f["variedad"] == "freedom"][:(a.n + 1) // 2] + [f for f in test if f["variedad"] == "vainilla"][:a.n // 2]
    COL = [(214, 120, 42), (52, 104, 235)]      # BGR: azul (clase 0), naranja (clase 1)

    for nombre, clases in cfg["modelos"].items():
        pesos = raiz / "runs" / nombre / "weights" / "best.pt"
        if not pesos.exists():
            continue
        det = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=str(pesos),
                                                 confidence_threshold=0.25, device="cuda:0")
        for f in fotos:
            img = cv2.imread(str(work / "imagenes" / f["imagen"]))
            res = get_sliced_prediction(img[..., ::-1].copy(), det, slice_height=cfg["tile"], slice_width=cfg["tile"],
                                        overlap_height_ratio=cfg["solape"], overlap_width_ratio=cfg["solape"], verbose=0)
            centros = []
            for o in res.object_prediction_list:
                x1, y1, x2, y2 = map(int, o.bbox.to_xyxy()); k = o.category.id
                cv2.rectangle(img, (x1, y1), (x2, y2), COL[k], 5)
                cv2.putText(img, f"{o.score.value:.2f}", (x1, y1 - 6), 0, 1.0, COL[k], 2)
                centros.append(((x1 + x2) / 2, (y1 + y2) / 2))
            H, W = img.shape[:2]
            cuenta = np.bincount([o.category.id for o in res.object_prediction_list], minlength=len(clases))
            leyenda = "  ".join(f"{c}: {n}" for c, n in zip(clases, cuenta))
            peq = cv2.resize(img, (W // 4, H // 4))
            cv2.rectangle(peq, (0, 0), (W // 4, 44), (255, 255, 255), -1)
            cv2.putText(peq, leyenda, (10, 32), 0, 1.0, (30, 30, 30), 2)
            base = f"{nombre}__{Path(f['imagen']).stem}"
            cv2.imwrite(str(out / f"{base}_completa.jpg"), peq, [cv2.IMWRITE_JPEG_QUALITY, 88])
            # acercamiento 1600x1200 en la zona más densa
            if centros:
                c = np.array(centros); d = ((c[:, None] - c[None]) ** 2).sum(-1) < 800 ** 2
                cx, cy = c[d.sum(1).argmax()]
            else:
                cx, cy = W / 2, H / 2
            x0 = int(np.clip(cx - 800, 0, W - 1600)); y0 = int(np.clip(cy - 600, 0, H - 1200))
            cv2.imwrite(str(out / f"{base}_zoom.jpg"), img[y0:y0 + 1200, x0:x0 + 1600], [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(base, leyenda)


if __name__ == "__main__":
    main()
