"""Drawing helpers for "light" effects.

All effects draw into a black ``light`` layer (uint8 BGR, same size as the
camera frame). The compositor then *adds* that layer to the video, so glows
behave like real light instead of stickers pasted on top.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

Color = Tuple[int, int, int]


# --------------------------------------------------------------------------
# Small math / colour helpers
# --------------------------------------------------------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    """Limit ``x`` to the range [lo, hi]."""
    return lo if x < lo else hi if x > hi else x


def smoothstep(x: float) -> float:
    """Ease curve: 0 -> 0, 1 -> 1, with soft start and end."""
    x = clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def ipt(p) -> Tuple[int, int]:
    """Convert any (x, y) pair to the plain Python ints OpenCV drawing needs."""
    return int(round(float(p[0]))), int(round(float(p[1])))


def scale_color(color: Color, k: float) -> Color:
    """Multiply a colour's brightness by ``k`` (clamped to 0..255)."""
    k = max(0.0, k)
    return int(min(255, color[0] * k)), int(min(255, color[1] * k)), int(min(255, color[2] * k))


def make_palette(stops: Sequence[Tuple[float, Color]], size: int = 256) -> np.ndarray:
    """Build a (size, 3) uint8 colour ramp from ``(position 0..1, BGR)`` stops."""
    positions = [s[0] for s in stops]
    xs = np.linspace(0.0, 1.0, size)
    palette = np.zeros((size, 3), np.uint8)
    for channel in range(3):
        values = [s[1][channel] for s in stops]
        palette[:, channel] = np.interp(xs, positions, values).astype(np.uint8)
    return palette


# --------------------------------------------------------------------------
# Glows
# --------------------------------------------------------------------------
_GLOW_SIZE = 128


@functools.lru_cache(maxsize=1)
def _glow_base() -> np.ndarray:
    """A soft round falloff as a 3-channel uint8 image (white in the middle, fading to black)."""
    lin = np.linspace(-1.0, 1.0, _GLOW_SIZE, dtype=np.float32)
    xx, yy = np.meshgrid(lin, lin)
    r = np.sqrt(xx * xx + yy * yy)
    falloff = np.exp(-4.5 * r * r)
    edge = np.clip((1.0 - r) / 0.3, 0.0, 1.0)
    edge = edge * edge * (3.0 - 2.0 * edge)  # force exactly zero at the border
    gray = np.round(falloff * edge * 255.0).astype(np.uint8)
    return cv2.merge([gray, gray, gray])


def add_glow(light: np.ndarray, center, radius: float, color: Color,
             intensity: float = 1.0, max_radius: int = 280) -> None:
    """Add a soft round glow to the light layer (additive, clipped to the frame)."""
    r = int(min(radius, max_radius))
    if r < 2 or intensity <= 0.0:
        return
    cx, cy = ipt(center)
    size = 2 * r + 1
    h, w = light.shape[:2]
    x0, y0 = cx - r, cy - r
    x1, y1 = x0 + size, y0 + size
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if cx0 >= cx1 or cy0 >= cy1:
        return  # completely off-screen

    sprite = cv2.resize(_glow_base(), (size, size), interpolation=cv2.INTER_LINEAR)
    visible = sprite[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    tint = (float(color[0]) * intensity, float(color[1]) * intensity, float(color[2]) * intensity, 0.0)
    colored = cv2.multiply(visible, tint, scale=1.0 / 255.0)   # tint the white glow
    roi = light[cy0:cy1, cx0:cx1]
    cv2.add(roi, colored, dst=roi)                              # additive, saturating


# --------------------------------------------------------------------------
# Sprites (images such as the mandala PNG)
# --------------------------------------------------------------------------
def to_light_sprite(img: np.ndarray) -> np.ndarray:
    """Convert any loaded image into a square, premultiplied BGR uint8 "light" sprite.

    * BGRA images: colour is multiplied by alpha (transparent = no light).
    * BGR / grayscale images: brightness itself is the amount of light.
    """
    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        bgr = (img[:, :, :3].astype(np.float32) * alpha).astype(np.uint8)
    else:
        bgr = img[:, :, :3].copy()

    # Pad to a square so rotating around the centre never clips the corners.
    h, w = bgr.shape[:2]
    side = max(h, w)
    square = np.zeros((side, side, 3), np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    square[y0:y0 + h, x0:x0 + w] = bgr
    return square


def load_light_sprite(path: Path) -> Optional[np.ndarray]:
    """Load an image file as a light sprite, or return None if it is missing/unreadable."""
    try:
        path = Path(path)
        if not path.is_file():
            return None
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        return to_light_sprite(img)
    except Exception:  # corrupt file, permissions, ...: fall back to the built-in shapes
        return None


def rotate_scale_sprite(sprite: np.ndarray, angle_deg: float, out_size: int) -> np.ndarray:
    """Rotate a square sprite clockwise by ``angle_deg`` and resize it to ``out_size`` px.

    Uses a single ``cv2.warpAffine`` call, so rotation and scaling cost one pass.
    """
    out_size = int(clamp(out_size, 8, 1600))
    h, w = sprite.shape[:2]
    scale = out_size / float(max(h, w))
    if scale < 0.7:
        # Shrinking a lot: pre-shrink with INTER_AREA for a clean result.
        sprite = cv2.resize(sprite, (out_size, out_size), interpolation=cv2.INTER_AREA)
        h, w = sprite.shape[:2]
        scale = 1.0
    # OpenCV treats positive angles as counter-clockwise, so negate for clockwise.
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -angle_deg, scale)
    matrix[0, 2] += out_size / 2.0 - w / 2.0
    matrix[1, 2] += out_size / 2.0 - h / 2.0
    return cv2.warpAffine(sprite, matrix, (out_size, out_size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def add_sprite(light: np.ndarray, sprite: np.ndarray, center) -> None:
    """Additively blend a premultiplied BGR sprite onto the light layer, centred on ``center``."""
    sh, sw = sprite.shape[:2]
    cx, cy = ipt(center)
    x0, y0 = cx - sw // 2, cy - sh // 2
    h, w = light.shape[:2]
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x0 + sw, w), min(y0 + sh, h)
    if cx0 >= cx1 or cy0 >= cy1:
        return
    roi = light[cy0:cy1, cx0:cx1]
    part = sprite[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    cv2.add(roi, part, dst=roi)


# --------------------------------------------------------------------------
# Lightning
# --------------------------------------------------------------------------
def zigzag(a, b, rng: np.random.Generator, segments: int, amplitude: float) -> np.ndarray:
    """Jagged path from ``a`` to ``b``: evenly spaced points pushed sideways at random.

    The two end points stay fixed so the bolt stays attached to the joints.
    """
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    delta = b - a
    length = float(np.hypot(delta[0], delta[1]))
    segments = max(1, int(segments))
    if length < 1e-3:
        return np.stack([a, b])
    normal = np.array([-delta[1], delta[0]], np.float32) / length
    t = np.linspace(0.0, 1.0, segments + 1, dtype=np.float32)[:, None]
    points = a + delta * t
    offsets = rng.uniform(-amplitude, amplitude, (segments + 1, 1)).astype(np.float32)
    offsets[0] = 0.0
    offsets[-1] = 0.0
    return points + normal * offsets


def draw_polyline(light: np.ndarray, points: np.ndarray, color: Color, thickness: int) -> None:
    """Anti-aliased open polyline."""
    cv2.polylines(light, [np.round(points).astype(np.int32)], False, color,
                  max(1, int(thickness)), cv2.LINE_AA)
