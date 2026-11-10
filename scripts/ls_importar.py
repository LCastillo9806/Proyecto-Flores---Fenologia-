"""Crea el proyecto de Label Studio y carga las fotos con la preanotación como predicción.

Orden de las tareas: semilla (train) -> val -> test -> resto de train.
Uso: python -I ls_importar.py config.yaml [--lote semilla|val|test|resto|todo] [--proyecto ID]
"""
import argparse, csv, json, os, urllib.request
from pathlib import Path

import yaml

ap = argparse.ArgumentParser(); ap.add_argument("config")
ap.add_argument("--lote", nargs="*", default=["semilla", "val", "test"])
ap.add_argument("--proyecto", type=int); ap.add_argument("--fuente", default="preanot")
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
work = Path(cfg["raiz"]) / "work"; EST = cfg["estados"]
URL, TOKEN = "http://localhost:8080", os.environ["LS_TOKEN"]
COLORES = ["#00c800", "#ffc800", "#ff0000", "#ff00ff"]


def api(metodo, ruta, datos=None):
    req = urllib.request.Request(URL + ruta, method=metodo, data=None if datos is None else json.dumps(datos).encode(),
                                 headers={"Authorization": f"Token {TOKEN}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        txt = r.read()
        return json.loads(txt) if txt else None


inv = list(csv.DictReader(open(work / "inventario.csv", encoding="utf-8")))
semilla = set(cfg["semilla"])
lotes = {"semilla": [f for f in inv if f["imagen"] in semilla],
         "val": [f for f in inv if f["particion"] == "val"],
         "test": [f for f in inv if f["particion"] == "test"],
         "resto": [f for f in inv if f["particion"] == "train" and f["imagen"] not in semilla]}
filas = [f for l in (["semilla", "val", "test", "resto"] if "todo" in a.lote else a.lote) for f in lotes[l]]

if a.proyecto:
    pid = a.proyecto
else:
    etiquetas = "\n".join(f'    <Label value="{e}" background="{c}"/>' for e, c in zip(EST, COLORES))
    p = api("POST", "/api/projects", {
        "title": "Rosas - estados fenológicos (dron)",
        "description": "Corrija las cajas: garbanzo (verde cerrado), rayando_color, punto_corte, abierta.",
        "label_config": f"""<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false" maxWidth="100%"/>
  <RectangleLabels name="label" toName="image" strokeWidth="2">
{etiquetas}
  </RectangleLabels>
</View>"""})
    pid = p["id"]
    api("POST", "/api/storages/localfiles", {"project": pid, "title": "imagenes",
                                             "path": str(work / "imagenes"), "use_blob_urls": False})
    print("Proyecto creado:", pid)

tareas = []
for f in filas:
    W, H = cfg["ancho_ref"], round(int(f["alto_orig"]) * float(f["escala"]))
    res = []
    p = work / a.fuente / (Path(f["imagen"]).stem + ".txt")
    for i, ln in enumerate(p.read_text().split("\n") if p.exists() else []):
        if not ln.strip():
            continue
        c, cx, cy, w, h = ln.split(); cx, cy, w, h = map(float, (cx, cy, w, h))
        res.append({"id": f"p{i}", "from_name": "label", "to_name": "image", "type": "rectanglelabels",
                    "original_width": W, "original_height": H, "image_rotation": 0,
                    "value": {"x": 100 * (cx - w / 2), "y": 100 * (cy - h / 2), "width": 100 * w,
                              "height": 100 * h, "rotation": 0, "rectanglelabels": [EST[int(c)]]}})
    tareas.append({"data": {"image": f"/data/local-files/?d=imagenes/{f['imagen']}", "imagen": f["imagen"],
                            "particion": f["particion"], "variedad": f["variedad"]},
                   "predictions": [{"model_version": a.fuente, "result": res}]})
api("POST", f"/api/projects/{pid}/import", tareas)
print(f"{len(tareas)} tareas importadas en el proyecto {pid}: {URL}/projects/{pid}")
