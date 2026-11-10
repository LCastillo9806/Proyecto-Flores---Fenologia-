"""Balance de blancos por foto usando el follaje como referencia (corrige dominantes entre vuelos)."""
import cv2
import numpy as np

# Proporciones B/G y R/G del follaje en los vuelos de referencia (BL50PM, BL52PM: 9 sep).
REF_BG, REF_RG = 0.575, 0.665


def balancear(img):
    peq = img[::4, ::4]
    hsv = cv2.cvtColor(peq, cv2.COLOR_BGR2HSV).astype(int)
    m = (hsv[..., 0] >= 25) & (hsv[..., 0] <= 90) & (hsv[..., 1] >= 40)
    if m.sum() < 1000:
        return img
    b, g, r = [np.median(peq[..., c][m]) for c in range(3)]
    ganancia = np.array([REF_BG * g / max(b, 1), 1.0, REF_RG * g / max(r, 1)], np.float32)
    return np.clip(img.astype(np.float32) * ganancia, 0, 255).astype(np.uint8)
