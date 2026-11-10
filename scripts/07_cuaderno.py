"""Paso 7: genera y ejecuta resultados.ipynb (tabla comparativa, curvas, matrices, PR, ejemplos).

Uso: python -I 07_cuaderno.py config.yaml [--preliminar]
"""
import argparse
from pathlib import Path

import nbformat as nbf
import yaml
from nbconvert.preprocessors import ExecutePreprocessor

ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--preliminar", action="store_true")
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
raiz = Path(cfg["raiz"])

aviso = ("> **⚠ RESULTADOS PRELIMINARES.** Las etiquetas de train, val y test son la preanotación automática "
         "SIN corregir. Las métricas miden cuánto se parece el modelo a esas reglas automáticas, no a la realidad. "
         "No reportar. Sirven para verificar que el pipeline completo funciona.\n") if a.preliminar else ""

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
celdas = [
    md(f"# Estados fenológicos de rosa en fotos de dron — comparación de 3 modelos YOLO26\n\n{aviso}\n"
       "| Modelo | Clase 0 | Clase 1 |\n|---|---|---|\n"
       "| **A** | rayando color | punto de corte |\n| **B** | garbanzo | rayando color |\n"
       "| **C** | botón cerrado (garbanzo + rayando + corte) | flor abierta |\n\n"
       "Los tres se entrenan con **YOLO26s**, mosaicos de 1024 px, mismos hiperparámetros y semilla. "
       "La partición es por bloque y fecha: test = BL18 Freedom + BL19 Vainilla, val = BL22 + BL32."),
    code(f"""import pandas as pd, matplotlib.pyplot as plt, yaml
from pathlib import Path
from IPython.display import Image, display, Markdown
RAIZ = Path(r"{raiz}"); RUNS = RAIZ / "runs"
cfg = yaml.safe_load(open(RAIZ / "scripts" / "config.yaml", encoding="utf-8"))
MODELOS = list(cfg["modelos"])
# paleta categórica validada (azul, naranja, verde agua): A, B, C
COLOR = dict(zip(MODELOS, ["#2a78d6", "#eb6834", "#1baf7a"]))
plt.rcParams.update({{"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "font.size": 10}})"""),
    md("## 1. Datos\nMosaicos y objetos por partición y clase en cada dataset."),
    code("""filas = []
for m in MODELOS:
    names = yaml.safe_load(open(RAIZ / "datasets" / m / "data.yaml", encoding="utf-8"))["names"]
    for sp in ["train", "val", "test"]:
        labs = list((RAIZ / "datasets" / m / "labels" / sp).glob("*.txt"))
        cnt = {n: 0 for n in names.values()}
        for p in labs:
            for ln in p.read_text().split("\\n"):
                if ln.strip(): cnt[names[int(ln.split()[0])]] += 1
        filas.append(dict(modelo=m, particion=sp, mosaicos=len(labs), **{f"obj_{k}": v for k, v in cnt.items()}))
pd.DataFrame(filas).fillna("").set_index(["modelo", "particion"])"""),
    md("## 2. Comparación en test\n**P** precisión, **R** exhaustividad (recall), **mAP50** y **mAP50-95** por mosaico. "
       "**MAE_conteo**: error absoluto medio del número de botones por foto completa (inferencia SAHI), "
       "que es lo que importa para proyectar flor."),
    code("""comp = pd.read_csv(RUNS / "comparacion.csv").set_index("modelo")
display(comp.T.round(3))
met = ["P", "R", "mAP50", "mAP50_95"]
fig, ax = plt.subplots(figsize=(8, 3.6))
w = 0.26
for i, m in enumerate(comp.index):
    xs = [j + (i - 1) * w for j in range(len(met))]
    bars = ax.bar(xs, comp.loc[m, met], w - 0.02, color=COLOR[m], label=m)
    ax.bar_label(bars, fmt="%.2f", fontsize=8, color="#52514e", padding=2)
ax.set_xticks(range(len(met)), met); ax.set_ylim(0, 1.08); ax.set_ylabel("valor en test")
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15)); ax.grid(axis="x", visible=False)
plt.tight_layout(); plt.show()"""),
    md("### AP50 por clase (test)"),
    code("""for m in comp.index:
    ap = comp.loc[m, [c for c in comp.columns if c.startswith("AP50_")]].dropna()
    print(m, {k.replace("AP50_", ""): round(v, 3) for k, v in ap.items()})"""),
    md("## 3. Curvas de entrenamiento\nPérdida de caja y de clasificación en validación, y mAP50-95 de validación por época. "
       "Una pérdida de val que sube mientras la de train baja indica sobreajuste."),
    code("""fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
for m in MODELOS:
    p = RUNS / m / "results.csv"
    if not p.exists(): continue
    r = pd.read_csv(p); r.columns = r.columns.str.strip()
    for ax, col, tit in zip(axs, ["val/box_loss", "val/cls_loss", "metrics/mAP50-95(B)"],
                            ["pérdida de caja (val)", "pérdida de clase (val)", "mAP50-95 (val)"]):
        ax.plot(r["epoch"], r[col], color=COLOR[m], lw=2, label=m)
        ax.set_title(tit, fontsize=10, loc="left"); ax.set_xlabel("época")
axs[2].legend(frameon=False)
plt.tight_layout(); plt.show()"""),
    md("## 4. Matrices de confusión (test, normalizadas)\nLa fila/columna *background* son objetos no detectados o detecciones sobrantes."),
    code("""for m in MODELOS:
    p = RUNS / f"{m}_test" / "confusion_matrix_normalized.png"
    if p.exists(): display(Markdown(f"**{m}**")); display(Image(filename=str(p), width=520))"""),
    md("## 5. Curvas precisión–exhaustividad (test)"),
    code("""for m in MODELOS:
    p = RUNS / f"{m}_test" / "BoxPR_curve.png"
    if p.exists(): display(Markdown(f"**{m}**")); display(Image(filename=str(p), width=560))"""),
    md("## 6. Conteo por foto completa (test)\nCada punto es una foto; la diagonal es conteo perfecto."),
    code("""cont = pd.read_csv(RUNS / "conteos_test.csv")
fig, axs = plt.subplots(1, len(MODELOS), figsize=(12, 3.8))
for ax, m in zip(axs, MODELOS):
    d = cont[cont.modelo == m]
    clases = [c[4:] for c in d.columns if c.startswith("ref_") and d[c].notna().any()]
    tope = 1
    for k, c in enumerate(clases):
        ax.scatter(d[f"ref_{c}"], d[f"pred_{c}"], s=40, color=["#2a78d6", "#eb6834"][k], label=c,
                   edgecolor="white", linewidth=1.5)
        tope = max(tope, d[f"ref_{c}"].max(), d[f"pred_{c}"].max())
    ax.plot([0, tope], [0, tope], color="#8a8984", lw=1, ls="--")
    ax.set_title(m, fontsize=10, loc="left"); ax.set_xlabel("conteo de referencia"); ax.set_ylabel("conteo predicho")
    ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.show()"""),
    md("## 7. Ejemplos de predicción en fotos completas de test (SAHI)\nAzul = clase 0, naranja = clase 1 del modelo."),
    code("""for m in MODELOS:
    display(Markdown(f"### {m}"))
    for p in sorted((RUNS / "ejemplos").glob(f"{m}__*_completa.jpg")):
        display(Markdown(f"`{p.stem.split('__')[1]}`"))
        display(Image(filename=str(p), width=900))
        z = p.with_name(p.name.replace("_completa", "_zoom"))
        if z.exists(): display(Image(filename=str(z), width=900))"""),
    md("## 8. Lotes de validación: etiqueta vs predicción (Ultralytics)"),
    code("""for m in MODELOS:
    for n in ["val_batch0_labels.jpg", "val_batch0_pred.jpg"]:
        p = RUNS / m / n
        if p.exists(): display(Markdown(f"**{m} — {n}**")); display(Image(filename=str(p), width=800))"""),
]
nb = nbf.v4.new_notebook(); nb.cells = celdas
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3 (rosas .venv)", "language": "python"}
ExecutePreprocessor(timeout=900, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(raiz)}})
dst = raiz / "resultados.ipynb"; nbf.write(nb, open(dst, "w", encoding="utf-8")); print("Escrito", dst)
