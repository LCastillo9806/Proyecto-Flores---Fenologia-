# Informe técnico: detección de estados fenológicos de rosa en fotos de dron con YOLO26

**Proyecto:** Proyección de flor en cultivo de rosa (variedades Freedom y Vainilla)
**Fecha de corte del informe:** 10 de noviembre de 2026
**Estado:** fase preliminar. Pipeline completo funcionando. Tres modelos entrenados con etiquetas automáticas. Corrección manual de etiquetas en curso.

> ⚠️ **Advertencia sobre las métricas.** Todas las métricas de este informe se midieron contra **etiquetas generadas automáticamente (preanotación v1) que todavía no corrigió una persona**. Indican que el pipeline funciona y permiten comparar los tres modelos entre sí, pero **no miden el desempeño real ni deben reportarse en una publicación**. Además se descubrió que la preanotación v1 marcaba muchos brotes de hoja rojizos como botones (ver §5.3). Las métricas válidas saldrán cuando se entrene con las etiquetas corregidas.

---

## Contenido

1. [Resumen ejecutivo](#1-resumen-ejecutivo)
2. [Objetivo y contexto agronómico](#2-objetivo-y-contexto-agronómico)
3. [Revisión de antecedentes](#3-revisión-de-antecedentes)
4. [Datos](#4-datos)
5. [Metodología paso a paso](#5-metodología-paso-a-paso)
6. [Resultados comparativos](#6-resultados-comparativos)
7. [Resultados detallados por modelo](#7-resultados-detallados-por-modelo)
8. [Discusión y limitaciones](#8-discusión-y-limitaciones)
9. [Decisión: qué modelo conservar](#9-decisión-qué-modelo-conservar)
10. [Lo que falta y plan para mejorar el modelo](#10-lo-que-falta-y-plan-para-mejorar-el-modelo)
11. [Cómo reproducir](#11-cómo-reproducir)
12. [Referencias](#12-referencias)

---

## 1. Resumen ejecutivo

- **Tarea.** Localizar cada botón de rosa en fotos cenitales de dron (DJI, 8064×6048 px) y clasificar su estado fenológico. No es una clasificación de imagen completa: cada foto tiene entre 50 y 200 botones de 30–130 px. Por eso se trató como **detección de objetos pequeños**, con mosaicos de 1024 px e inferencia SAHI.
- **Tres modelos binarios comparados**, todos con YOLO26s, los mismos hiperparámetros y la misma partición:

  | Modelo | Clase 0 | Clase 1 |
  |---|---|---|
  | **A** | rayando color | punto de corte |
  | **B** | garbanzo | rayando color |
  | **C** | botón cerrado (garbanzo + rayando + corte) | flor abierta |

- **Etiquetas.** No había anotaciones. Se construyó una **preanotación automática** con YOLOE-26 (indicaciones visuales) más reglas de color, y se anota **una sola vez con 4 estados**; los tres datasets se derivan de esa anotación.
- **Resultados preliminares en test** (22 fotos de bloques nunca vistos, 25 épocas, etiquetas automáticas):

  | Modelo | Precisión | Recall | mAP50 | mAP50-95 | F1 (conf 0,25) | Exactitud de clasificación* |
  |---|---|---|---|---|---|---|
  | A | 0,550 | 0,628 | 0,588 | 0,433 | 0,51 / 0,59 | 85,3 % |
  | B | 0,567 | 0,588 | 0,606 | 0,496 | 0,63 / 0,50 | 98,0 % |
  | C | 0,619 | 0,687 | **0,705** | **0,579** | 0,60 / 0,65 | 98,1 % |

  \*Entre los botones detectados, porcentaje con el estado correcto. Los F1 se dan por clase (clase 0 / clase 1).

- **Hallazgo crítico.** Alrededor del **75 %** de las cajas "rojas" de la preanotación v1 eran **brotes de hoja rojizos**, no botones, y los modelos aprendieron ese error. Se corrigió en la **preanotación v3**, con balance de blancos por foto, filtro de matiz por mancha, filtro de entorno y filtro de forma. La v3 ya está cargada en Label Studio para su corrección manual.
- **Recomendación.** Conservar **A** como modelo principal (horizonte de cosecha de 0–7 días), conservar **B** como secundario (2–3 semanas) y **descartar C**. Lo más urgente es **corregir las etiquetas**.

---

## 2. Objetivo y contexto agronómico

La **proyección de flor** consiste en estimar cuántos tallos se cosecharán en las próximas semanas. Se hace contando botones por estado fenológico, porque cada estado corresponde a un tiempo aproximado hasta el corte. En rosa de corte en Colombia se usa la secuencia:

**arroz** (≈0,4 cm) → **arveja** (0,5–0,7 cm) → **garbanzo** (0,8–1,2 cm) → **rayando color** → **punto de corte** → (flor abierta = pasada de corte)

El tiempo entre estados depende de la variedad y de los grados-día acumulados. Por ejemplo, en Facatativá se reportaron entre 77 y 99 días hasta "rayando color" según la variedad.

**Objetivo del proyecto:** automatizar el conteo por estado a partir de fotos de dron, para reemplazar o complementar el conteo manual por cama.

Estados usados en este proyecto (4 clases de anotación):

| Estado | Criterio visual desde el dron | Horizonte aproximado a corte |
|---|---|---|
| `garbanzo` | botón cerrado, verde; los sépalos lo cubren | 2–3 semanas |
| `rayando_color` | los sépalos se separan y se ve una franja de pétalo de color | ~1 semana |
| `punto_corte` | el color del pétalo se ve completo; los pétalos están cerrados o apenas se separan | 0–2 días |
| `abierta` | flor abierta, pasada de corte | (pérdida) |

---

## 3. Revisión de antecedentes

| Trabajo | Qué hace | Relevancia |
|---|---|---|
| Rose-Mamba-YOLO (Frontiers in Plant Science, 2025) | YOLOv11 con módulos Mamba para detectar rosas en invernadero desde UAV. mAP50 87,5 %, P 90,4 %, R 83,1 %. | Es lo más cercano. Advierte que los **botones tempranos pequeños** son el caso difícil. |
| Conteo de botones de rosa Explorer (AgriEngineering, 2024) | UAV + YOLOv5 + DeepSORT; mAP 94,1 %; R² = 0,998 frente al conteo manual. | Muestra que el **conteo** desde dron es viable. Es en campo abierto y solo distingue abierto/cerrado. |
| Estadios fenológicos de rosa de corte con YOLOv8 (Agronomía Colombiana, UNAL, 2025) | Detección de estadios en invernadero desde plataforma terrestre. | Rinde menos en estadios tempranos (botones de menos del 0,3 % del área de la imagen). |
| Fenología de rosa en Suesca (Agronomía Colombiana) | Define arroz, arveja, garbanzo, rayar color y corte, con diámetros. | Base de los criterios de clase. |
| Grados-día en rosa (U. de Cundinamarca; U. de La Salle) | Tiempo térmico hasta rayando color o corte. | Para convertir conteos por estado en proyección semanal. |
| SAHI con YOLO26 (Ultralytics) | Inferencia por mosaicos en imágenes grandes. | Es la técnica usada aquí para fotos de 8064 px. |

**Vacío identificado:** no se encontró un trabajo que combine **dron + YOLO + varios estados fenológicos + proyección** en rosa de invernadero. Este proyecto apunta a ese vacío.

---

## 4. Datos

### 4.1 Inventario

110 fotos JPG de un dron DJI (cámara FC3682), tomadas en vuelos cenitales sobre camas de invernadero.

| Grupo (fecha_bloque_variedad) | Variedad | Fotos | Resolución original | Partición |
|---|---|---|---|---|
| 28_08_26_BL22 | Vainilla (blanca) | 3 | 4000×2250 | val |
| 28_08_26_BL32 | Freedom (roja) | 7 | 4000×2250 | val |
| 28_08_26_BL64 | Vainilla | 21 | 4000×2250 | train |
| 02_09_26_BL17 | Vainilla | 7 | 8064×6048 | train |
| 02_09_26_BL18 | Freedom | 14 | 8064×6048 | **test** |
| 02_09_26_BL19 | Vainilla | 8 | 8064×6048 | **test** |
| 02_09_26_BL50prmv | Freedom | 8 | 8064×6048 | train |
| 09_09_26_BL50PM | Freedom | 20 | 8064×6048 | train |
| 09_09_26_BL52PM | Freedom | 22 | 8064×6048 | train |
| **Total** | | **110** | | train 78 · val 10 · test 22 |

- La carpeta `13_10_26_BLOQUE50` venía **vacía**.
- `28_08_26_BL64_vainilla_DNG` contiene 20 RAW que duplican los JPG del mismo bloque; no se usaron.

### 4.2 Decisiones sobre los datos

- **Partición por bloque y fecha, no por foto.** Las fotos consecutivas de un mismo bloque se solapan. Partir por foto metería el mismo botón en train y en test (fuga de datos). Test = BL18 Freedom + BL19 Vainilla; val = BL22 + BL32.
- **Normalización de escala.** Las fotos de agosto (4000×2250) y las de septiembre (8064×6048) cubren el mismo campo visual, así que un botón mide la mitad de píxeles en las de agosto. Todas se llevaron a 8064 px de ancho para que un botón mida lo mismo. Contra: las de agosto quedan ampliadas ×2 y algo borrosas.
- **Mosaicos.** Las fotos se cortan en mosaicos de 1024×1024 con 25 % de solape. Un objeto se conserva en un mosaico si se ve al menos el 50 % de él. Se guarda además un 15 % de mosaicos sin objetos, como ejemplos de fondo.

### 4.3 Tamaño de los datasets preliminares (etiquetas v1)

| Dataset | Partición | Mosaicos | Objetos clase 0 | Objetos clase 1 |
|---|---|---|---|---|
| A rayando/corte | train | 4 160 | 2 561 | 4 306 |
| | val | 464 | 408 | 758 |
| | test | 1 226 | 800 | 1 099 |
| B garbanzo/rayando | train | 5 019 | 8 503 | 2 561 |
| | val | 484 | 940 | 408 |
| | test | 1 690 | 4 234 | 800 |
| C cerrado/abierta | train | 5 668 | 15 370 | 709 |
| | val | 565 | 2 106 | 41 |
| | test | 1 782 | 6 133 | 158 |

El modelo C está muy desbalanceado: 1 flor abierta por cada 22 botones cerrados. Su val tiene solo 41 flores abiertas.

---

## 5. Metodología paso a paso

### 5.1 Pipeline

```
fotos dron ─► 01_inventario ─► 02_preanotar ─► Label Studio (corrección humana) ─► 03_datasets (A, B, C)
                 (escala, partición)   (YOLOE + color)        ls_importar / ls_exportar        (mosaicos)
                                                                                                │
   resultados.ipynb ◄─ 07_cuaderno ◄─ 06_ejemplos (SAHI) ◄─ 05_evaluar (test + conteo) ◄─ 04_entrenar (YOLO26s)
```

| Paso | Script | Qué hace |
|---|---|---|
| 1 | `01_inventario.py` | Lista fotos, extrae fecha, bloque y variedad, asigna la partición por grupo y normaliza a 8064 px. |
| 2 | `02_preanotar.py` | Propone cajas con 4 estados (YOLOE-26 + reglas de color). Con `--modelo` usa un YOLO26 ya entrenado (modelo semilla). |
| 3 | `ls_importar.py` / `ls_exportar.py` | Carga fotos y predicciones en Label Studio y descarga las correcciones a `work/etiquetas/`. |
| 4 | `03_datasets.py` | Deriva los 3 datasets binarios en mosaicos; con `--semilla`, un dataset de 4 estados. |
| 5 | `04_entrenar.py` | Entrena YOLO26s con hiperparámetros idénticos para los tres modelos. |
| 6 | `05_evaluar.py` | Calcula métricas en test por mosaico y el **conteo por foto completa** con SAHI. |
| 7 | `06_ejemplos.py` | Genera predicciones dibujadas sobre fotos completas de test. |
| 8 | `07_cuaderno.py`, `08_figuras_informe.py` | Producen el cuaderno de resultados y las figuras de este informe. |
| — | `matriz_metricas.py` | Matriz de confusión con umbral 0,25 y F1/exactitud derivados. |

### 5.2 Configuración de entrenamiento

| Parámetro | Valor | Motivo |
|---|---|---|
| Modelo base | `yolo26s.pt` (COCO) | Buen equilibrio entre tamaño y precisión para 12 GB de VRAM. |
| Tamaño de entrada | 1024 px | Igual al mosaico, sin reescalar los botones. |
| Épocas | **25** (preliminar); 150 previstas | Prueba rápida del pipeline. |
| Lote, semilla | 8, 0 (determinista) | Comparación justa entre modelos. |
| Aumentos | volteo H y V 0,5, rotación ±10°, escala 0,3, HSV (h 0,01, s 0,5, v 0,3), mosaico (se apaga en las últimas 15 épocas) | En vista cenital el botón no tiene orientación fija. |
| Optimización | LR coseno | |
| Inferencia en foto completa | SAHI, mosaico 1024, solape 0,25, confianza 0,25 | |
| Hardware | RTX 3060 12 GB, PyTorch 2.11 + CUDA 12.8, Ultralytics 8.4.193 | |

### 5.3 Preanotación automática: iteraciones y lo que se aprendió

La preanotación se rehízo cuatro veces. Documentar por qué falló cada versión es parte del resultado.

| Versión | Método | Resultado |
|---|---|---|
| v0 | YOLOE-26 con texto ("rose bud", "flower bud"…) | ❌ Casi no detecta nada a esta escala. |
| v0b | Manchas de color sin filtros | ❌ Cientos de cajitas sobre reflejos, suelo y brillo de hojas (548 "punto de corte" falsos en una foto de Vainilla). |
| **v1** | YOLOE-26 con **12 ejemplos visuales** + manchas de color con tamaño mínimo + supresión de cajas contenidas | ⚠️ Usable, y es con la que se entrenaron los modelos preliminares. Después se descubrió que ~75 % de las manchas "rojas" eran **brotes de hoja rojizos**. |
| v2 | Solo matiz carmesí + entorno verde | ❌ Elimina los brotes, pero **pierde botones** en el vuelo BL18, que tiene dominante cálida, y dispara los falsos "garbanzo" (~120 por foto). |
| **v3** (actual) | **Balance de blancos por foto** con el follaje como referencia + mancha roja con matiz mediano ≤ 3 + filtro de entorno + "garbanzo" solo si la caja es pequeña y casi cuadrada | ✅ Sin brotes marcados, recupera los botones vino de BL18 y los garbanzo falsos bajan de ~120 a ~12 por foto. **Cargada en Label Studio.** |

| Total de cajas | garbanzo | rayando | punto de corte | abierta |
|---|---|---|---|---|
| v1 | 6 777 | 1 989 | 3 245 | 472 |
| v3 | 2 232 | 910 | 1 481 | 433 |

**Ejemplos visuales de YOLOE (v1):** 12 recortes de fotos de train. Las coordenadas están en `scripts/config.yaml`.

![Ejemplos para YOLOE](docs/figuras/preanotacion/0_ejemplos_yoloe.jpg)

**Diagnóstico de la v1:** las 48 manchas rojas de dos fotos de train. Solo ~10 son botones (0, 8, 15, 18, 19, 22, 27, 33, 38, 42); el resto son brotes rojizos.

![Manchas rojas v1](docs/figuras/preanotacion/1_manchas_rojas_antes.jpg)

**Con la regla de matiz carmesí:** ~30 de 40 son botones. Los falsos restantes son puntas de brote y pétalos caídos en el suelo.

![Manchas carmesí](docs/figuras/preanotacion/2_manchas_carmesi.jpg)

**Por qué falló la v2:** en el vuelo BL18 el balance de blancos es cálido (matiz del follaje 47 frente a 54 en los otros vuelos). El botón real se ve vino anaranjado y la regla fija lo descartaba.

<img src="docs/figuras/preanotacion/6_boton_vino_BL18.jpg" width="260">

| Vuelo | Matiz mediano del follaje | B,G,R del follaje |
|---|---|---|
| 02_09_26_BL18 | **47,3** | 58, 118, 84 |
| 02_09_26_BL50prmv | 54,7 | 70, 117, 83 |
| 09_09_26_BL50PM | 54,0 | 67, 113, 74 |
| 09_09_26_BL52PM | 54,0 | 65, 113, 76 |
| 28_08_26_BL32 | 50,3 | 81, 131, 102 |

**v2 frente a v3 en la misma zona de BL18.** En la v2, las cajas verdes caen sobre hojas amarillentas y el botón rojo se pierde. En la v3 el botón vino aparece detectado y desaparece el ruido.

| v2 | v3 |
|---|---|
| ![v2](docs/figuras/preanotacion/3_v2_ruido_garbanzo.jpg) | ![v3](docs/figuras/preanotacion/4_v3_BL18.jpg) |

### 5.4 Incidentes técnicos resueltos

| Problema | Causa | Solución |
|---|---|---|
| El PyTorch del sistema no usaba la GPU | Era la versión solo-CPU | Entorno aislado `.venv` con PyTorch 2.11 + CUDA 12.8. |
| El entrenamiento fallaba al arrancar (`RuntimeError: ... bootstrapping phase`) | En Windows, los *workers* del DataLoader necesitan `if __name__ == "__main__"` | Todos los scripts envueltos en `main()`. |
| El modelo C se cayó en la época 9 (`MemoryError`) | Se lanzó una validación en paralelo al entrenamiento y se agotó la RAM | Reanudado desde `last.pt` con `workers=2`, sin tareas en paralelo. |
| Label Studio rechazaba el token | `label-studio --version` **arranca un servidor**: quedó un segundo servidor en el puerto 8080 con otra base de datos | Se apagó el servidor sobrante y se arrancó el del proyecto en 8080. |
| Matriz de confusión vacía con umbral 0,25 | En Ultralytics 8.4 la matriz solo se llena con `plots=True` | `matriz_metricas.py` usa `plots=True`. |

---

## 6. Resultados comparativos

> Recordatorio: test = 22 fotos (BL18 Freedom + BL19 Vainilla), etiquetas automáticas v1 y 25 épocas.

### 6.1 Métricas globales en test

| Modelo | Precisión | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| A · rayando vs corte | 0,550 | 0,628 | 0,588 | 0,433 |
| B · garbanzo vs rayando | 0,567 | 0,588 | 0,606 | 0,496 |
| C · cerrado vs abierta | **0,619** | **0,687** | **0,705** | **0,579** |

![Comparación de métricas en test](docs/figuras/comparacion_metricas_test.png)

**Glosario de métricas:**
- **Precisión:** de las cajas que marca el modelo, qué fracción es correcta.
- **Recall (exhaustividad):** de los botones reales, qué fracción encuentra.
- **mAP50:** precisión media, contando como acierto una caja que se solapa ≥50 % (IoU) con la real.
- **mAP50-95:** lo mismo, promediado con exigencias de solape de 50 % a 95 %. Mide además qué tan bien ajustada está la caja.
- **F1:** media armónica de precisión y recall.

### 6.2 AP50 por clase

| Modelo | Clase 0 | AP50 | Clase 1 | AP50 |
|---|---|---|---|---|
| A | rayando color | 0,518 | punto de corte | 0,659 |
| B | garbanzo | 0,671 | rayando color | 0,541 |
| C | botón cerrado | 0,651 | flor abierta | 0,758 |

![AP50 por clase](docs/figuras/ap50_por_clase.png)

"Rayando color" es la clase más difícil en los dos modelos donde aparece (0,52 y 0,54).

### 6.3 Matrices de confusión con umbral de confianza 0,25 (test, conteos)

Las filas son lo predicho y las columnas lo real. *Fondo* = no detectado (fila) o detección sin objeto real (columna).

**Modelo A**

| Predicho ↓ / Real → | rayando | corte | fondo |
|---|---|---|---|
| rayando | **417** | 113 | 306 |
| corte | 86 | **741** | 571 |
| fondo (omitido) | 297 | 245 | — |

**Modelo B**

| Predicho ↓ / Real → | garbanzo | rayando | fondo |
|---|---|---|---|
| garbanzo | **2 913** | 47 | 2 035 |
| rayando | 23 | **471** | 589 |
| fondo (omitido) | 1 298 | 282 | — |

**Modelo C**

| Predicho ↓ / Real → | cerrado | abierta | fondo |
|---|---|---|---|
| cerrado | **3 569** | 4 | 2 142 |
| abierta | 66 | **141** | 70 |
| fondo (omitido) | 2 498 | 13 | — |

### 6.4 Métricas derivadas de la matriz (umbral 0,25)

| Modelo | Clase | Precisión | Recall | **F1** |
|---|---|---|---|---|
| A | rayando color | 0,499 | 0,521 | **0,510** |
| A | punto de corte | 0,530 | 0,674 | **0,594** |
| B | garbanzo | 0,583 | 0,688 | **0,631** |
| B | rayando color | 0,435 | 0,589 | **0,500** |
| C | botón cerrado | 0,624 | 0,582 | **0,602** |
| C | flor abierta | 0,509 | 0,892 | **0,648** |

| Modelo | Exactitud de **clasificación** (entre botones detectados) | Exactitud global (incluye omisiones y falsos) |
|---|---|---|
| A | **85,3 %** | 41,7 % |
| B | **98,0 %** | 44,2 % |
| C | **98,1 %** | 43,6 % |

**Cómo leer la exactitud.** En detección de objetos la *accuracy* clásica no está bien definida, porque no se pueden contar los "verdaderos negativos": el fondo es infinito. Por eso se dan dos versiones:
- La **exactitud de clasificación**: cuando el modelo encuentra un botón, ¿acierta su estado? Aquí A acierta 85 % y B y C ~98 %.
- La **exactitud global**: incluye también lo que no encontró y lo que inventó.

La diferencia entre las dos muestra que **el problema principal no es confundir estados sino encontrar o no los botones**, es decir, las omisiones y los falsos positivos.

> Nota: los mosaicos de test se solapan un 25 %, así que un botón en el borde puede contarse en dos mosaicos. Las cifras absolutas de la matriz están infladas en esa proporción; los porcentajes no se ven afectados.

### 6.5 Validación al final del entrenamiento (época 25)

| Modelo | Precisión | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| A | 0,488 | 0,493 | 0,446 | 0,305 |
| B | 0,429 | 0,543 | 0,427 | 0,333 |
| C | 0,544 | 0,461 | 0,477 | 0,404 |

Val sale más bajo que test porque son solo 10 fotos de agosto, de 4000 px ampliadas ×2 y más borrosas.

### 6.6 Curvas de entrenamiento

![Curvas de entrenamiento](docs/figuras/curvas_entrenamiento.png)

- Las pérdidas de val bajan de forma sostenida en los tres modelos: **no hay sobreajuste**.
- El mAP seguía subiendo en la época 25: los modelos **quedaron cortos de entrenamiento**.

### 6.7 Conteo por foto completa (lo que importa para proyectar)

Se predijo con SAHI sobre cada foto completa de test y se comparó con el conteo de la referencia.

| Modelo | Clase | Media de referencia por foto | Media predicha | Error absoluto medio | Error relativo | Correlación |
|---|---|---|---|---|---|---|
| A | punto de corte | 26,0 | 30,8 | 6,1 | 24 % | **0,87** |
| A | rayando color | 20,0 | 20,0 | 4,2 | 21 % | 0,74 |
| B | garbanzo | 95,9 | 74,2 | 25,3 | 26 % | **0,87** |
| B | rayando color | 20,0 | 23,7 | 5,0 | 25 % | 0,77 |
| C | botón cerrado | 141,8 | 102,3 | 43,6 | 31 % | ❌ 0,09 |
| C | flor abierta | 4,0 | 4,0 | 0,9 | 22 % | **0,93** |

![Conteo por foto](docs/figuras/conteo_por_foto_test.png)

- **A** y **B** siguen bien la tendencia (correlación 0,87 en su clase principal).
- B **subcuenta** garbanzo cuando hay muchos botones.
- En **C**, el conteo de botones cerrados **se satura en unos 100 por foto** aunque la referencia llegue a 220, y no correlaciona (0,09). Ese conteo no sirve.

---

## 7. Resultados detallados por modelo

Cada carpeta `docs/figuras/<modelo>/` contiene: matrices de confusión (umbral 0,001 de Ultralytics y umbral 0,25), curvas PR, F1, P y R en test, curvas de entrenamiento, distribución de etiquetas, lotes de validación etiqueta/predicción y ejemplos en fotos completas.

### 7.1 Modelo A: rayando color vs punto de corte

| Matriz normalizada (conf 0,25) | Curva precisión–recall |
|---|---|
| ![](docs/figuras/A_rayando_vs_corte/test_matriz_conf025_normalizada.png) | ![](docs/figuras/A_rayando_vs_corte/test_BoxPR_curve.png) |

| Curva F1–confianza | Curva recall–confianza |
|---|---|
| ![](docs/figuras/A_rayando_vs_corte/test_BoxF1_curve.png) | ![](docs/figuras/A_rayando_vs_corte/test_BoxR_curve.png) |

![Entrenamiento A](docs/figuras/A_rayando_vs_corte/entrenamiento_results.png)

**Ejemplo en foto completa de test (Freedom, BL18).** Azul = rayando, naranja = punto de corte. Se ven **falsos positivos sobre brotes rojizos**, heredados de la preanotación v1.

![Ejemplo A](docs/figuras/A_rayando_vs_corte/ejemplo_02_09_26_BL18_freedom_JPEG__DJI_0424_zoom.jpg)

**Lectura.** 67 % de los cortes y 52 % de los rayando se detectan con el estado correcto. Rayando y corte se confunden entre sí en un 10–11 %, que es esperable porque son estados vecinos. La mayor pérdida está en las omisiones: 37 % de los rayando no se detectan.

### 7.2 Modelo B: garbanzo vs rayando color

| Matriz normalizada (conf 0,25) | Curva precisión–recall |
|---|---|
| ![](docs/figuras/B_garbanzo_vs_rayando/test_matriz_conf025_normalizada.png) | ![](docs/figuras/B_garbanzo_vs_rayando/test_BoxPR_curve.png) |

| Curva F1–confianza | Curva recall–confianza |
|---|---|
| ![](docs/figuras/B_garbanzo_vs_rayando/test_BoxF1_curve.png) | ![](docs/figuras/B_garbanzo_vs_rayando/test_BoxR_curve.png) |

![Entrenamiento B](docs/figuras/B_garbanzo_vs_rayando/entrenamiento_results.png)

![Ejemplo B](docs/figuras/B_garbanzo_vs_rayando/ejemplo_02_09_26_BL19_vainilla_JPEG__DJI_0507_zoom.jpg)

**Lectura.** Casi no confunde garbanzo con rayando (98 % de exactitud de clasificación). Su problema es **encontrar** los garbanzo: son el botón más pequeño y verde sobre verde, y 31 % se omite.

### 7.3 Modelo C: botón cerrado vs flor abierta

| Matriz normalizada (conf 0,25) | Curva precisión–recall |
|---|---|
| ![](docs/figuras/C_cerrado_vs_abierta/test_matriz_conf025_normalizada.png) | ![](docs/figuras/C_cerrado_vs_abierta/test_BoxPR_curve.png) |

| Curva F1–confianza | Curva recall–confianza |
|---|---|
| ![](docs/figuras/C_cerrado_vs_abierta/test_BoxF1_curve.png) | ![](docs/figuras/C_cerrado_vs_abierta/test_BoxR_curve.png) |

![Entrenamiento C](docs/figuras/C_cerrado_vs_abierta/entrenamiento_results.png)

**Ejemplo (Vainilla, BL19).** Detecta bien los botones crema (azul) y las flores abiertas (naranja). También marca algunos brotes rojizos como "botón cerrado".

![Ejemplo C](docs/figuras/C_cerrado_vs_abierta/ejemplo_02_09_26_BL19_vainilla_JPEG__DJI_0507_zoom.jpg)

**Lectura.** Encuentra el 89 % de las flores abiertas y casi no las confunde con botones. Pero sus métricas son altas porque la tarea es fácil (una flor abierta es grande y muy distinta), no porque resuelva el problema agronómico.

---

## 8. Discusión y limitaciones

1. **Las etiquetas son el cuello de botella.** Los modelos imitan la preanotación v1, que tenía errores sistemáticos: brotes rojizos marcados como botones y la frontera entre rayando y corte decidida por una regla de color, no por un experto. Con etiquetas así, una métrica alta puede significar que el modelo copia bien un error.
2. **Pocas épocas.** Los tres modelos seguían mejorando en la época 25.
3. **Validación pequeña y heterogénea.** Son 10 fotos de agosto, ampliadas ×2. La métrica de val tiene mucha varianza.
4. **Desbalance en C** (1:22) y muy pocas flores abiertas en val (41).
5. **Dominantes de color entre vuelos.** La luz y el balance de blancos cambian entre fechas, y afectan tanto a la preanotación como al modelo.
6. **Objetos muy pequeños.** El garbanzo mide unos 35–75 px a 8064 px de ancho. Volar más bajo ayudaría más que cualquier cambio de arquitectura.
7. **Conteo con mosaicos solapados.** El conteo por foto usa SAHI, que fusiona las detecciones duplicadas. Las matrices por mosaico, en cambio, cuentan dos veces los objetos de borde (§6.4).

---

## 9. Decisión: qué modelo conservar

| Modelo | Utilidad para proyectar cosecha | Desempeño preliminar | Decisión |
|---|---|---|---|
| **A** rayando vs corte | **Alta.** Corte = cosecha inmediata; rayando ≈ 1 semana. | mAP50 0,59; correlación del conteo de corte 0,87. | ✅ **Principal** |
| **B** garbanzo vs rayando | Media-alta: proyección a 2–3 semanas. | mAP50 0,61; subcuenta garbanzo. | ✅ **Secundario**; mejora si se vuela más bajo. |
| **C** cerrado vs abierta | **Baja.** La flor abierta ya pasó el corte (es pérdida). "Cerrado" mezcla todos los estados. | Mejor mAP (0,70) por ser la tarea más fácil; el conteo de cerrados no sirve (correlación 0,09). | ❌ **Descartar** |

**A largo plazo:** un único sistema de 4 estados (detector de botón + clasificador de estado) entrega los conteos de A y de B a la vez (ver §10).

---

## 10. Lo que falta y plan para mejorar el modelo

### 10.1 Estado actual

- [x] Revisión de antecedentes
- [x] Inventario, normalización y partición por bloque
- [x] Preanotación automática (v1 → v3)
- [x] Pipeline completo: datasets, entrenamiento, evaluación, SAHI, cuaderno e informe
- [x] Tres modelos preliminares entrenados y comparados
- [x] Label Studio montado con las 110 fotos y la preanotación v3
- [ ] **Corrección manual: 10 fotos semilla + 10 de val + 22 de test** ← siguiente paso
- [ ] Modelo semilla de 4 estados → re-preanotar las 68 fotos de train restantes → revisarlas
- [ ] Entrenamiento final (150 épocas) y métricas válidas
- [ ] Conversión de conteos por estado a proyección semanal (grados-día)

### 10.2 Mejoras, ordenadas por impacto

| # | Mejora | Impacto esperado | Esfuerzo | Detalle |
|---|---|---|---|---|
| 1 | **Corregir las etiquetas** | ★★★★★ | Medio (humano) | Quitar los brotes y las hojas marcados; definir **por escrito** el criterio de cada estado con un agrónomo; aplicarlo igual en todas las fotos. |
| 2 | **Etiquetado asistido por modelo** | ★★★★ | Bajo | Entrenar con la semilla corregida, preanotar el resto con ese modelo y revisar. Corregir así es mucho más rápido. |
| 3 | **Esquema en dos etapas** | ★★★★ | Medio | (a) Detector de una sola clase "botón"; (b) clasificador YOLO26-cls del estado sobre un recorte ampliado. Separa "encontrar" de "clasificar", que son los dos problemas observados (§6.4). |
| 4 | **Segmentar cada botón (SAM → YOLO26-seg)** | ★★★ | Medio | SAM convierte cada caja en máscara sin dibujar a mano. Permite medir el **diámetro en cm** con la altura de vuelo (EXIF), que es el criterio agronómico de arroz, arveja y garbanzo; el **% de pétalo visible** (rayando frente a corte); y filtrar por forma (botón redondo frente a hoja alargada). |
| 5 | **Más épocas y un modelo más grande** | ★★★ | Bajo | 150 épocas; probar YOLO26m. |
| 6 | **Aumento de datos dirigido** | ★★ | Bajo-medio | *Copy-paste* de botones (requiere las máscaras de SAM), rotaciones de 90°, brillo y desenfoque, **sobremuestreo** de las clases escasas. No se recomiendan imágenes sintéticas generativas: introducen sesgo y son difíciles de defender en una publicación. |
| 7 | **Normalizar el color entre vuelos** | ★★ | Bajo | Aplicar el balance de blancos por follaje (`balance_blancos.py`) también en el entrenamiento y la inferencia. |
| 8 | **Máscara de cama de cultivo** | ★ | Bajo | Descartar las detecciones sobre el camino (hojas secas). La mayor parte del ruido está dentro del follaje, así que su aporte es menor. |
| 9 | **Más y mejores datos de campo** | ★★★★ | Alto (campo) | Volar **más bajo** y a altura y hora constantes; recuperar las fotos del 13/10 (carpeta vacía); más bloques para test. |
| 10 | **Umbral de operación** | ★ | Bajo | Usar el umbral que maximiza F1 en val (≈0,15 en A), no 0,25 fijo. |

### 10.3 Protocolo de corrección en Label Studio

1. Arrancar con `scripts/iniciar_labelstudio.bat`, después de definir las variables `LS_TOKEN` y `LS_PASSWORD`, y abrir `http://localhost:8080`.
2. Orden: **semilla** (10) → **val** (10) → **test** (22). La columna `particion` lo indica.
3. Por foto: zoom; borrar las cajas falsas (Supr); cambiar el estado si está mal; dibujar los botones que faltan (teclas 1–4 para elegir estado); **Submit**.
4. Exportar con `python scripts/ls_exportar.py scripts/config.yaml --proyecto 2` → `work/etiquetas/`.

---

## 11. Cómo reproducir

```bash
# 1. Entorno (Windows, GPU NVIDIA)
python -m venv .venv
.venv\Scripts\pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\pip install -r requirements.txt

# 2. Ajustar rutas en scripts/config.yaml (raiz, raw) y ejecutar
python -I scripts/01_inventario.py  scripts/config.yaml
python -I scripts/02_preanotar.py   scripts/config.yaml            # preanotación v3
python -I scripts/03_datasets.py    scripts/config.yaml            # usa work/etiquetas si existen
python -I scripts/04_entrenar.py    scripts/config.yaml            # --epochs 25 para prueba rápida
python -I scripts/05_evaluar.py     scripts/config.yaml
python -I scripts/06_ejemplos.py    scripts/config.yaml --n 4
python -I scripts/07_cuaderno.py    scripts/config.yaml            # genera resultados.ipynb
python -I scripts/08_figuras_informe.py scripts/config.yaml docs/figuras

# Etiquetado asistido
python -I scripts/03_datasets.py scripts/config.yaml --semilla --fuente etiquetas
python -I scripts/04_entrenar.py scripts/config.yaml --modelos S_semilla_4estados
python -I scripts/02_preanotar.py scripts/config.yaml --modelo runs/S_semilla_4estados/weights/best.pt --particion train
```

Las fotos originales, los mosaicos, los pesos (`.pt`) y la base de datos de Label Studio **no se versionan** por tamaño (ver `.gitignore`).

### Archivos de resultados en este repositorio

| Archivo | Contenido |
|---|---|
| `resultados/comparacion.csv` | Métricas en test por modelo y clase, y error de conteo. |
| `resultados/conteos_test.csv` | Conteo de referencia y predicho por foto y por clase. |
| `resultados/matrices_test_conf025.txt` | Matrices de confusión con umbral 0,25, F1 y exactitud. |
| `resultados/entrenamiento_<modelo>.csv` | Métricas por época. |
| `resultados/inventario.csv` | Foto, grupo, variedad, resolución y partición. |
| `resultados/datasets_preliminares.txt` | Mosaicos y objetos por partición. |

---

## 12. Referencias

1. Rose-Mamba-YOLO: an enhanced framework for efficient and accurate greenhouse rose monitoring. *Frontiers in Plant Science*, 2025. https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1607582/full
2. Rose bud counting with UAV, YOLOv5 and DeepSORT. *AgriEngineering*, 2024. https://doaj.org/article/fff829c724274102be476c76a13d2c4a
3. Detección de estadios fenológicos en rosa de corte con YOLOv8. *Agronomía Colombiana*, UNAL, 2025. https://revistas.unal.edu.co/index.php/agrocol/article/view/122164
4. Fenología de rosa en Suesca (Cundinamarca). *Agronomía Colombiana*. https://revistas.unal.edu.co/index.php/agrocol/article/download/20028/21160/66906
5. Grados-día hasta rayando color en variedades de rosa, Facatativá. Universidad de Cundinamarca. https://repositorio.ucundinamarca.edu.co/handle/20.500.12558/4572?show=full
6. Grados-día en rosa. Universidad de La Salle. https://ciencia.lasalle.edu.co/server/api/core/bitstreams/e09008ff-04ae-410d-a273-9f127e825d03/content
7. Punto de corte comercial en rosa. *Revista Chapingo Serie Horticultura* (SciELO México). https://www.scielo.org.mx/pdf/rcsh/v20n3/v20n3a5.pdf
8. Ultralytics: SAHI tiled inference con YOLO26. https://docs.ultralytics.com/es/guides/sahi-tiled-inference
9. Ultralytics: detección desde dron y objetos pequeños con YOLO26. https://www.ultralytics.com/blog/drone-detection-dataset-yolo
