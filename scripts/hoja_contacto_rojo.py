"""Diagnóstico: hoja de contacto de manchas rojas (candidatos Freedom) con sus rasgos de color y forma."""
import sys, cv2, numpy as np
out = sys.argv[1]; fotos = sys.argv[2:]
celdas, filas = [], []
for f in fotos:
    img = cv2.imread(f); hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1], hsv[..., 2]
    m = (((h <= 8) | (h >= 165)) & (s >= 110) & (v >= 45)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    for i in range(1, n):
        x, y, w, hh, area = st[i]
        if max(w, hh) < 30 or area < 450: continue
        px = lab[y:y+hh, x:x+w] == i
        H = h[y:y+hh, x:x+w][px]; H = np.where(H >= 90, H - 180, H)
        bgr = img[y:y+hh, x:x+w][px].astype(int)
        cnt, _ = cv2.findContours(px.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnt, key=cv2.contourArea); sol = cv2.contourArea(c) / max(cv2.contourArea(cv2.convexHull(c)), 1)
        rasgos = dict(H=np.median(H), S=np.median(s[y:y+hh, x:x+w][px]), V=np.median(v[y:y+hh, x:x+w][px]),
                      BmR=np.median(bgr[:, 0] - bgr[:, 2]), sol=sol, asp=min(w, hh) / max(w, hh), lado=max(w, hh))
        cx, cy, L = x + w // 2, y + hh // 2, max(w, hh) + 60
        rec = img[max(cy - L, 0):cy + L, max(cx - L, 0):cx + L]
        rec = cv2.resize(rec, (200, 200)); k = len(celdas)
        cv2.putText(rec, str(k), (5, 25), 0, 0.9, (255, 255, 255), 3); cv2.putText(rec, str(k), (5, 25), 0, 0.9, (0, 0, 0), 1)
        celdas.append(rec); filas.append((k, rasgos))
        if len(celdas) >= 48: break
    if len(celdas) >= 48: break
while len(celdas) % 8: celdas.append(np.zeros((200, 200, 3), np.uint8))
grid = np.vstack([np.hstack(celdas[i:i + 8]) for i in range(0, len(celdas), 8)])
cv2.imwrite(out, grid)
for k, r in filas: print(k, " ".join(f"{a}={b:.2f}" if isinstance(b, float) else f"{a}={b}" for a, b in r.items()))
