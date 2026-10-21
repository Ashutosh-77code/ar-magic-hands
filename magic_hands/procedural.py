"""Built-in artwork, drawn with OpenCV shapes.

These are used automatically when ``assets/mandala.png`` or
``assets/repulsor.png`` are missing. Run ``python scripts/generate_assets.py``
to export them as PNG files you can edit or replace.

Every function returns a square premultiplied BGR "light" sprite: black means
no light, bright pixels are added to the video.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

_ORANGE = (0, 135, 255)
_YELLOW = (40, 205, 255)
_HOT = (170, 240, 255)


def _pt(c: float, r: float, deg: float):
    """Point at radius ``r`` and angle ``deg`` from the centre ``c`` (angle 0 = up)."""
    a = math.radians(deg - 90.0)
    return int(round(c + r * math.cos(a))), int(round(c + r * math.sin(a)))


def build_mandala(size: int = 512, seed: int = 7) -> np.ndarray:
    """A Doctor Strange style magic circle: rings, runes, a star and nested squares."""
    img = np.zeros((size, size, 3), np.uint8)
    c = size / 2.0
    R = size * 0.47
    centre = (int(c), int(c))
    rng = np.random.default_rng(seed)
    aa = cv2.LINE_AA
    t = max(1, size // 170)  # base line thickness scales with sprite size

    # Outer double ring
    cv2.circle(img, centre, int(R), _ORANGE, t * 2, aa)
    cv2.circle(img, centre, int(R * 0.955), _YELLOW, t, aa)

    # Fine tick marks around the ring
    for k in range(96):
        long_tick = k % 4 == 0
        cv2.line(img, _pt(c, R * (0.86 if long_tick else 0.9), k * 3.75),
                 _pt(c, R * 0.94, k * 3.75), _YELLOW if long_tick else _ORANGE, max(1, t // 2 + 1), aa)

    # Rune-like dashes: arcs of random length on two rings
    for ring_r, count in ((0.80, 36), (0.70, 24)):
        for k in range(count):
            start = k * (360.0 / count) + rng.uniform(0, 4)
            length = float(rng.choice([4, 6, 9, 12])) * (24.0 / count)
            cv2.ellipse(img, centre, (int(R * ring_r), int(R * ring_r)), -90, start, start + length,
                        _ORANGE if k % 3 else _YELLOW, t * 2, aa)

    # Twelve-point star {12/5}
    star_r = R * 0.66
    for k in range(12):
        cv2.line(img, _pt(c, star_r, k * 30), _pt(c, star_r, (k + 5) * 30), _ORANGE, t, aa)
        cv2.circle(img, _pt(c, star_r, k * 30), max(2, t * 2), _HOT, -1, aa)

    # Two squares rotated 45 degrees (the classic sling ring look)
    sq_r = R * 0.50
    for offset in (0.0, 45.0):
        poly = np.array([_pt(c, sq_r, offset + 90 * k) for k in range(4)], np.int32)
        cv2.polylines(img, [poly], True, _YELLOW, t * 2, aa)

    # Inner rings and centre
    cv2.circle(img, centre, int(R * 0.40), _ORANGE, t * 2, aa)
    cv2.circle(img, centre, int(R * 0.30), _YELLOW, t, aa)
    for k in range(8):
        cv2.line(img, _pt(c, R * 0.12, k * 45), _pt(c, R * 0.30, k * 45), _ORANGE, t, aa)
    cv2.circle(img, centre, int(R * 0.10), _HOT, t, aa)
    cv2.circle(img, centre, max(2, int(R * 0.035)), _HOT, -1, aa)
    return img


def build_repulsor(size: int = 512) -> np.ndarray:
    """An Iron Man style repulsor: concentric rings, a triangle and a glowing core."""
    img = np.zeros((size, size, 3), np.uint8)
    c = size / 2.0
    R = size * 0.46
    centre = (int(c), int(c))
    aa = cv2.LINE_AA
    t = max(1, size // 120)
    blue, pale, white = (255, 170, 60), (255, 225, 160), (255, 245, 225)

    cv2.circle(img, centre, int(R), blue, t * 2, aa)
    cv2.circle(img, centre, int(R * 0.88), pale, t, aa)
    cv2.circle(img, centre, int(R * 0.62), blue, t * 2, aa)
    for k in range(6):
        cv2.circle(img, _pt(c, R * 0.75, k * 60), max(2, int(R * 0.05)), white, -1, aa)
    for offset in (0.0, 60.0):
        tri = np.array([_pt(c, R * 0.62, offset + 120 * k) for k in range(3)], np.int32)
        cv2.polylines(img, [tri], True, pale, t, aa)
    glow = np.zeros_like(img)
    cv2.circle(glow, centre, int(R * 0.30), white, -1, aa)
    glow = cv2.GaussianBlur(glow, (0, 0), size / 40.0)
    return cv2.add(img, glow)
