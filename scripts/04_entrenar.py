"""Paso 4: entrenar los 3 modelos YOLO26 con los mismos hiperparámetros.

Uso: python -I 04_entrenar.py config.yaml [--modelos A_rayando_vs_corte ...] [--epochs N]
Salida: runs/<modelo>/weights/best.pt
"""
import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO



def main():
    ap = argparse.ArgumentParser(); ap.add_argument("config")
    ap.add_argument("--modelos", nargs="*"); ap.add_argument("--epochs", type=int)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    raiz = Path(cfg["raiz"]); e = cfg["entrenamiento"]

    for nombre in a.modelos or cfg["modelos"]:
        m = YOLO(str(raiz / "pesos" / e["modelo"]))
        m.train(data=str(raiz / "datasets" / nombre / "data.yaml"), imgsz=e["imgsz"],
                epochs=a.epochs or e["epochs"], patience=e["patience"], batch=e["batch"], seed=e["seed"],
                deterministic=True, project=str(raiz / "runs"), name=nombre, exist_ok=True,
                workers=4, cos_lr=True, close_mosaic=15,
                # vista cenital: los botones no tienen orientación fija
                fliplr=0.5, flipud=0.5, degrees=10, scale=0.3, hsv_h=0.01, hsv_s=0.5, hsv_v=0.3,
                plots=True)


if __name__ == "__main__":
    main()
