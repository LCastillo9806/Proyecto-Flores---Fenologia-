"""Matriz de confusión con conteos y métricas derivadas (precisión, recall, F1, exactitud) para un modelo."""
import sys
from pathlib import Path
import numpy as np
from ultralytics import YOLO

def main():
    pesos, data, split = sys.argv[1], sys.argv[2], sys.argv[3]
    r = YOLO(pesos).val(data=data, split=split, imgsz=1024, batch=8, plots=True, verbose=False, conf=0.25,
                           project="runs/matrices_conf025", name=Path(pesos).parents[1].name, exist_ok=True)
    M = r.confusion_matrix.matrix.astype(int)          # filas = predicho, columnas = real; último = fondo
    nombres = list(r.names.values()) + ["fondo"]
    print("\nMatriz (filas = predicho, columnas = real), umbral de confianza 0.25")
    print(" " * 16 + "".join(f"{n:>16}" for n in nombres))
    for i, n in enumerate(nombres):
        print(f"{n:>16}" + "".join(f"{v:>16}" for v in M[i]))
    k = len(nombres) - 1
    for c in range(k):
        tp = M[c, c]; fp = M[c].sum() - tp; fn = M[:, c].sum() - tp
        p = tp / (tp + fp) if tp + fp else 0; rc = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * p * rc / (p + rc) if p + rc else 0
        print(f"{nombres[c]:>16}: VP={tp} FP={fp} FN={fn}  precisión={p:.3f} recall={rc:.3f} F1={f1:.3f}")
    det = M[:k, :k]
    print(f"\nExactitud de CLASIFICACIÓN entre botones detectados (sin fondo): {np.trace(det)/det.sum():.3f}")
    print(f"Exactitud GLOBAL incluyendo omisiones y falsos positivos: {np.trace(det)/M.sum():.3f}")

if __name__ == "__main__":
    main()
