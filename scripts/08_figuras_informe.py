"""Paso 8: figuras comparativas para el informe (PNG) y copia de las gráficas de Ultralytics.

Uso: python -I 08_figuras_informe.py config.yaml <carpeta_destino>
"""
import shutil, sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
raiz = Path(cfg["raiz"]); RUNS = raiz / "runs"; dst = Path(sys.argv[2]); dst.mkdir(parents=True, exist_ok=True)
MODELOS = list(cfg["modelos"])
ETQ = {"A_rayando_vs_corte": "A · rayando vs corte", "B_garbanzo_vs_rayando": "B · garbanzo vs rayando",
       "C_cerrado_vs_abierta": "C · cerrado vs abierta"}
# paleta categórica validada (azul, naranja, verde agua); la clase 0/1 de cada modelo usa azul/naranja
COLOR = dict(zip(MODELOS, ["#2a78d6", "#eb6834", "#1baf7a"])); CLASE = ["#2a78d6", "#eb6834"]
TXT, TXT2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 140, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.edgecolor": "#8a8984",
                     "axes.labelcolor": TXT2, "xtick.color": TXT2, "ytick.color": TXT2, "text.color": TXT,
                     "font.size": 10, "figure.facecolor": "white", "axes.facecolor": "white"})


def guardar(fig, nombre):
    fig.tight_layout(); fig.savefig(dst / nombre, bbox_inches="tight"); plt.close(fig); print("figura", nombre)


comp = pd.read_csv(RUNS / "comparacion.csv").set_index("modelo")

# 1. métricas globales en test
met = ["P", "R", "mAP50", "mAP50_95"]; nom = ["Precisión", "Recall", "mAP50", "mAP50-95"]
fig, ax = plt.subplots(figsize=(8.5, 3.8)); w = 0.26
for i, m in enumerate(MODELOS):
    xs = [j + (i - 1) * w for j in range(len(met))]
    b = ax.bar(xs, comp.loc[m, met], w - 0.03, color=COLOR[m], label=ETQ[m])
    ax.bar_label(b, fmt="%.2f", fontsize=8, color=TXT2, padding=2)
ax.set_xticks(range(len(met)), nom); ax.set_ylim(0, 1); ax.set_ylabel("valor en test"); ax.grid(axis="x", visible=False)
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
guardar(fig, "comparacion_metricas_test.png")

# 2. AP50 por clase
fig, axs = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
for ax, m in zip(axs, MODELOS):
    clases = list(cfg["modelos"][m]); vals = [comp.loc[m, f"AP50_{c}"] for c in clases]
    b = ax.bar(range(2), vals, 0.6, color=CLASE)
    ax.bar_label(b, fmt="%.2f", fontsize=9, color=TXT2, padding=2)
    ax.set_xticks(range(2), [c.replace("_", " ") for c in clases]); ax.set_title(ETQ[m], fontsize=10, loc="left")
    ax.set_ylim(0, 1); ax.grid(axis="x", visible=False)
axs[0].set_ylabel("AP50 en test")
guardar(fig, "ap50_por_clase.png")

# 3. curvas de entrenamiento
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.4))
for m in MODELOS:
    r = pd.read_csv(RUNS / m / "results.csv"); r.columns = r.columns.str.strip()
    for ax, col, tit in zip(axs, ["val/box_loss", "val/cls_loss", "metrics/mAP50-95(B)"],
                            ["pérdida de caja (val)", "pérdida de clase (val)", "mAP50-95 (val)"]):
        ax.plot(r["epoch"], r[col], color=COLOR[m], lw=2, label=ETQ[m]); ax.set_title(tit, fontsize=10, loc="left")
        ax.set_xlabel("época")
axs[2].legend(frameon=False, fontsize=8)
guardar(fig, "curvas_entrenamiento.png")

# 4. conteo por foto
cont = pd.read_csv(RUNS / "conteos_test.csv")
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.8))
for ax, m in zip(axs, MODELOS):
    d = cont[cont.modelo == m]; tope = 1
    for k, c in enumerate(cfg["modelos"][m]):
        ax.scatter(d[f"ref_{c}"], d[f"pred_{c}"], s=36, color=CLASE[k], edgecolor="white", linewidth=1.5,
                   label=c.replace("_", " "))
        tope = max(tope, d[f"ref_{c}"].max(), d[f"pred_{c}"].max())
    ax.plot([0, tope], [0, tope], color="#8a8984", lw=1, ls="--")
    ax.set_title(ETQ[m], fontsize=10, loc="left"); ax.set_xlabel("conteo de referencia por foto")
    ax.legend(frameon=False, fontsize=8)
axs[0].set_ylabel("conteo predicho por foto")
guardar(fig, "conteo_por_foto_test.png")

# 5. copia de gráficas de Ultralytics (test y entrenamiento) y ejemplos reducidos
for m in MODELOS:
    d = dst / m; d.mkdir(exist_ok=True)
    for n in ["confusion_matrix_normalized.png", "confusion_matrix.png", "BoxPR_curve.png", "BoxF1_curve.png",
              "BoxP_curve.png", "BoxR_curve.png", "val_batch0_labels.jpg", "val_batch0_pred.jpg"]:
        p = RUNS / f"{m}_test" / n
        if p.exists(): shutil.copy(p, d / f"test_{n}")
    for n in ["results.png", "labels.jpg"]:
        p = RUNS / m / n
        if p.exists(): shutil.copy(p, d / f"entrenamiento_{n}")
    for p in sorted((RUNS / "ejemplos").glob(f"{m}__*.jpg")):
        im = cv2.imread(str(p)); h, w = im.shape[:2]; f = min(1.0, 1400 / w)
        cv2.imwrite(str(d / ("ejemplo_" + p.name.split("__", 1)[1])), cv2.resize(im, (int(w * f), int(h * f))),
                    [cv2.IMWRITE_JPEG_QUALITY, 82])
print("listo")
