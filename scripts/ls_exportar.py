"""Descarga las anotaciones corregidas de Label Studio a work/etiquetas/<imagen>.txt (YOLO, 4 estados).

Solo se escriben las fotos con al menos una anotación enviada (Submit), no las que solo tienen predicción.
Uso: python -I ls_exportar.py config.yaml --proyecto ID
"""
import argparse, json, os, urllib.request
from collections import Counter
from pathlib import Path

import yaml

ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--proyecto", type=int, default=2)
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
work = Path(cfg["raiz"]) / "work"; EST = cfg["estados"]
URL, TOKEN = "http://localhost:8080", os.environ["LS_TOKEN"]
req = urllib.request.Request(f"{URL}/api/projects/{a.proyecto}/export?exportType=JSON",
                             headers={"Authorization": f"Token {TOKEN}"})
tareas = json.load(urllib.request.urlopen(req))
dst = work / "etiquetas"; dst.mkdir(exist_ok=True)
n, cnt = 0, Counter()
for t in tareas:
    anots = [x for x in t.get("annotations", []) if not x.get("was_cancelled")]
    if not anots:
        continue
    ultima = max(anots, key=lambda x: x["updated_at"])
    lineas = []
    for r in ultima["result"]:
        if r.get("type") != "rectanglelabels":
            continue
        v = r["value"]; e = v["rectanglelabels"][0]
        x, y, w, h = v["x"] / 100, v["y"] / 100, v["width"] / 100, v["height"] / 100
        lineas.append(f"{EST.index(e)} {x + w / 2:.6f} {y + h / 2:.6f} {w:.6f} {h:.6f}"); cnt[e] += 1
    (dst / (Path(t["data"]["imagen"]).stem + ".txt")).write_text("\n".join(lineas))
    n += 1
print(f"{n} fotos corregidas exportadas a {dst}. Objetos: {dict(cnt)}")
