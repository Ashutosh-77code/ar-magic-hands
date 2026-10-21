"""Combines the camera frame with the light layer drawn by an effect."""

from __future__ import annotations

import cv2
import numpy as np


def bloom(light: np.ndarray) -> np.ndarray:
    """Soft glow around everything bright in ``light``.

    Works on shrunken copies (1/4 and 1/8 size, made with ``cv2.pyrDown``) and
    scales the result back up. That is far cheaper than blurring the
    full-resolution frame, and looks the same because glows are soft anyway.
    """
    h, w = light.shape[:2]
    quarter = cv2.pyrDown(cv2.pyrDown(light))
    eighth = cv2.pyrDown(quarter)
    tight = cv2.GaussianBlur(quarter, (0, 0), 2.0)    # hugs the shapes
    wide = cv2.GaussianBlur(eighth, (0, 0), 3.5)      # large soft halo
    wide = cv2.resize(wide, (quarter.shape[1], quarter.shape[0]), interpolation=cv2.INTER_LINEAR)
    glow = cv2.addWeighted(tight, 0.7, wide, 1.0, 0)
    return cv2.resize(glow, (w, h), interpolation=cv2.INTER_LINEAR)


def composite(frame: np.ndarray, light: np.ndarray, bloom_strength: float = 1.0) -> np.ndarray:
    """Return ``frame`` + ``light`` + bloom, as additive blending.

    ``cv2.add`` saturates at 255 instead of wrapping around, so overlapping
    lights simply get brighter, exactly like real light.
    """
    if not light.any():
        return frame
    out = cv2.add(frame, light)
    if bloom_strength > 0.0:
        out = cv2.addWeighted(out, 1.0, bloom(light), bloom_strength, 0)
    return out
