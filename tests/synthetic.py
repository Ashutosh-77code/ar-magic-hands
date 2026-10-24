"""Fake hands for tests and previews: no camera or MediaPipe needed.

``make_hand`` builds a 21-landmark hand from a template (fingers up, y axis
up, wrist at the origin, middle knuckle one unit away) and then scales,
rotates and moves it into pixel coordinates.
"""

from __future__ import annotations

import math

import numpy as np

from magic_hands.hand_tracker import Hand

# Open palm, fingers up. Units: 1.0 = wrist to middle knuckle.
_OPEN = [
    (0.00, 0.00),                                              # 0 wrist
    (-0.45, 0.25), (-0.72, 0.55), (-0.95, 0.80), (-1.12, 1.00),   # thumb
    (-0.35, 0.95), (-0.40, 1.45), (-0.42, 1.75), (-0.43, 2.00),   # index
    (0.00, 1.00), (0.00, 1.60), (0.00, 1.95), (0.00, 2.25),       # middle
    (0.33, 0.95), (0.37, 1.50), (0.39, 1.80), (0.40, 2.05),       # ring
    (0.62, 0.85), (0.70, 1.25), (0.74, 1.50), (0.77, 1.70),       # pinky
]

# Fist: fingers curled back toward the palm, thumb tucked.
_FIST = list(_OPEN)
for _first, _x in ((5, -0.35), (9, 0.00), (13, 0.33), (17, 0.62)):
    _FIST[_first + 1] = (_x, 1.30)
    _FIST[_first + 2] = (_x, 1.15)
    _FIST[_first + 3] = (_x, 0.78)
_FIST[2], _FIST[3], _FIST[4] = (-0.55, 0.45), (-0.45, 0.62), (-0.30, 0.72)

# Pinch: thumb tip touches the index tip, other fingers stay open.
_PINCH = list(_OPEN)
_PINCH[2], _PINCH[3], _PINCH[4] = (-0.60, 0.85), (-0.50, 1.40), (-0.44, 1.97)

# Point: a fist with only the index finger stretched out.
_POINT = list(_FIST)
_POINT[5:9] = _OPEN[5:9]

# Peace sign: index and middle fingers stretched out.
_PEACE = list(_POINT)
_PEACE[9:13] = _OPEN[9:13]

POSES = {"open": _OPEN, "fist": _FIST, "pinch": _PINCH, "point": _POINT, "peace": _PEACE}


def make_hand(pose: str = "open", center=(640.0, 460.0), scale: float = 150.0,
              angle_deg: float = 0.0, slot: int = 0, label: str = "Right") -> Hand:
    """Create a Hand. ``center`` is the wrist position in pixels, ``angle_deg`` is clockwise."""
    pts = np.array(POSES[pose], np.float32)
    pts[:, 1] *= -1.0                       # template is y-up, images are y-down
    a = math.radians(angle_deg)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], np.float32)
    pts = pts @ rot.T                       # clockwise on screen
    pts = pts * scale + np.asarray(center, np.float32)
    return Hand(slot=slot, label=label, score=0.99, landmarks=pts.astype(np.float32))
