"""Colour helpers for the rainbow-style effects."""

from __future__ import annotations

import colorsys
from typing import Tuple


def hue_to_bgr(hue: float, saturation: float = 1.0, value: float = 1.0) -> Tuple[int, int, int]:
    """Convert a hue (0..1, wraps around) to a BGR tuple of ints for OpenCV drawing."""
    r, g, b = colorsys.hsv_to_rgb(hue % 1.0, saturation, value)
    return int(b * 255), int(g * 255), int(r * 255)
