"""Paso 5: evaluar y comparar los 3 modelos en test.

  a) métricas por mosaico (P, R, mAP50, mAP50-95, por clase) con model.val(split='test')
  b) conteo por foto completa con SAHI (lo que importa para proyectar flor): error absoluto medio
     del conteo de cada clase frente a la anotación.
Uso: python -I 05_evaluar.py config.yaml [--fuente etiquetas|preanot]
Salida: runs/comparacion.csv, runs/conteos_test.csv
"""
import argparse, csv
from pathlib import Path

import numpy as np
import yaml
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO



def main():
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--fuente", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    raiz = Path(cfg["raiz"]); work = raiz / "work"; EST = cfg["estados"]
    fuente = a.fuente or ("etiquetas" if any((work / "etiquetas").glob("*.txt")) else "preanot")
    inv = [f for f in csv.DictReader(open(work / "inventario.csv", encoding="utf-8")) if f["particion"] == "test"]

    tabla, conteos = [], []
    for nombre, clases in cfg["modelos"].items():
        pesos = raiz / "runs" / nombre / "weights" / "best.pt"
        if not pesos.exists():
            print("Falta", pesos); continue
        r = YOLO(str(pesos)).val(data=str(raiz / "datasets" / nombre / "data.yaml"), split="test",
                                 imgsz=cfg["entrenamiento"]["imgsz"], batch=8, project=str(raiz / "runs"),
                                 name=f"{nombre}_test", exist_ok=True, verbose=False)
        fila = dict(modelo=nombre, P=r.box.mp, R=r.box.mr, mAP50=r.box.map50, mAP50_95=r.box.map)
        for i, c in enumerate(clases):
            fila[f"AP50_{c}"] = r.box.ap50[i] if i < len(r.box.ap50) else float("nan")

        # conteo por foto completa
        mapa = {EST.index(e): i for i, (_, ests) in enumerate(clases.items()) for e in ests}
        det = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=str(pesos),
                                                 confidence_threshold=0.25, device="cuda:0")
        errores = []
        for f in inv:
            ref = np.zeros(len(clases), int)
            p = work / fuente / (Path(f["imagen"]).stem + ".txt")
            for ln in (p.read_text().split("\n") if p.exists() else []):
                if ln.strip() and int(ln.split()[0]) in mapa:
                    ref[mapa[int(ln.split()[0])]] += 1
            res = get_sliced_prediction(str(work / "imagenes" / f["imagen"]), det, slice_height=cfg["tile"],
                                        slice_width=cfg["tile"], overlap_height_ratio=cfg["solape"],
                                        overlap_width_ratio=cfg["solape"], verbose=0)
            pred = np.bincount([o.category.id for o in res.object_prediction_list], minlength=len(clases))
            errores.append(np.abs(pred - ref))
            conteos.append(dict(modelo=nombre, imagen=f["imagen"],
                                **{f"ref_{c}": ref[i] for i, c in enumerate(clases)},
                                **{f"pred_{c}": pred[i] for i, c in enumerate(clases)}))
        errores = np.array(errores)
        for i, c in enumerate(clases):
            fila[f"MAE_conteo_{c}"] = errores[:, i].mean()
        tabla.append(fila)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in fila.items()})

    for nombre_csv, filas in (("comparacion.csv", tabla), ("conteos_test.csv", conteos)):
        campos = sorted({k for f in filas for k in f}, key=lambda k: (k != "modelo", k != "imagen", k))
        with open(raiz / "runs" / nombre_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=campos); w.writeheader(); w.writerows(filas)
    print("Fuente de las etiquetas de referencia:", fuente,
          "(si es 'preanot', las métricas NO son válidas para reportar)")


if __name__ == "__main__":
    main()
