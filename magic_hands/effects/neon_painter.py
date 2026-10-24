"""Effect 4: Neon Painter.

Point with one finger (index out, the others curled) to paint glowing neon
lines in the air. The colour shifts along the stroke. Hold an open palm for
one second to wipe the canvas, or press 4 again to clear instantly.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from magic_hands import landmarks as lm
from magic_hands.colors import hue_to_bgr
from magic_hands.effects.base import Effect
from magic_hands.gestures import is_pointing
from magic_hands.graphics import add_glow, ipt, scale_color
from magic_hands.hand_tracker import Hand

CLEAR_SECONDS = 1.0    # hold an open palm this long to clear
START_FRAMES = 3       # frames of pointing before a stroke starts (ignores accidental flicks)


class _Pen:
    def __init__(self) -> None:
        self.point_frames = 0
        self.pos: Optional[np.ndarray] = None    # smoothed fingertip while drawing
        self.last: Optional[np.ndarray] = None   # end of the previous segment
        self.cursor: Optional[np.ndarray] = None
        self.hue = 0.0
        self.clear_hold = 0.0


class NeonPainter(Effect):
    name = "Neon Painter"
    hint = "Point with one finger to draw. Hold an open palm for 1 s (or press 4) to clear."
    bloom_strength = 1.3

    def __init__(self, seed: Optional[int] = None) -> None:
        self._pens: Dict[int, _Pen] = {}
        self._hands: Sequence[Hand] = ()
        self._glow: Optional[np.ndarray] = None   # soft wide halo of the strokes
        self._core: Optional[np.ndarray] = None   # bright thin centre of the strokes
        self._dissolve: Optional[np.ndarray] = None
        self._dissolve_amt = 0.0
        self.strokes_drawn = 0                    # segments painted so far (handy for tests)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._pens.clear()
        self._hands = ()
        self._glow = self._core = self._dissolve = None
        self._dissolve_amt = 0.0

    def on_reselect(self) -> None:
        self.clear()

    def clear(self) -> None:
        """Wipe the canvas with a short dissolve."""
        if self._glow is None:
            return
        if self._glow.any() or self._core.any():
            self._dissolve = cv2.add(self._glow, self._core)
            self._dissolve_amt = 1.0
        self._glow.fill(0)
        self._core.fill(0)

    # ------------------------------------------------------------------
    def update(self, hands: Sequence[Hand], dt: float, size: Tuple[int, int]) -> None:
        w, h = size
        if self._glow is None or self._glow.shape[:2] != (h, w):
            self._glow = np.zeros((h, w, 3), np.uint8)
            self._core = np.zeros((h, w, 3), np.uint8)
            self._dissolve = None
        self._hands = hands
        self._dissolve_amt = max(0.0, self._dissolve_amt - dt / 0.45)

        seen = set()
        for hand in hands:
            seen.add(hand.slot)
            pen = self._pens.setdefault(hand.slot, _Pen())

            # Open palm held for a second clears the canvas.
            if hand.is_open:
                pen.clear_hold += dt
            else:
                pen.clear_hold = max(0.0, pen.clear_hold - dt * 3.0)
            if pen.clear_hold >= CLEAR_SECONDS:
                self.clear()
                for other in self._pens.values():
                    other.clear_hold = 0.0
                continue

            if not is_pointing(hand):
                pen.point_frames = 0
                pen.pos = pen.last = pen.cursor = None   # lifting the finger ends the stroke
                continue

            tip = hand.landmarks[lm.INDEX_TIP]
            pen.cursor = tip
            pen.point_frames += 1
            pen.pos = tip.copy() if pen.pos is None else pen.pos + (tip - pen.pos) * 0.65
            if pen.point_frames < START_FRAMES:
                continue
            if pen.last is not None:
                self._stroke(pen, pen.last, pen.pos, hand.scale)
            pen.last = pen.pos.copy()

        for slot in [s for s in self._pens if s not in seen]:
            del self._pens[slot]

    def _stroke(self, pen: _Pen, a: np.ndarray, b: np.ndarray, scale: float) -> None:
        """Paint one segment as a neon tube: wide soft halo + coloured tube + white-hot centre."""
        pen.hue += float(np.linalg.norm(b - a)) * 0.0012       # colour flows along the line
        color = hue_to_bgr(pen.hue)
        thickness = max(3, int(scale * 0.055))
        pa, pb = ipt(a), ipt(b)
        cv2.line(self._glow, pa, pb, scale_color(color, 0.45), thickness * 3, cv2.LINE_AA)
        cv2.line(self._core, pa, pb, color, thickness, cv2.LINE_AA)
        centre = tuple(int(c * 0.35 + 255 * 0.65) for c in color)
        cv2.line(self._core, pa, pb, centre, max(1, thickness // 3), cv2.LINE_AA)
        self.strokes_drawn += 1

    # ------------------------------------------------------------------
    def draw(self, light: np.ndarray) -> None:
        if self._glow is None or self._glow.shape != light.shape:
            return
        if self._dissolve is not None and self._dissolve_amt > 0.0:
            cv2.add(light, cv2.convertScaleAbs(self._dissolve, alpha=self._dissolve_amt), dst=light)
        cv2.add(light, self._glow, dst=light)
        cv2.add(light, self._core, dst=light)

        for hand in self._hands:
            pen = self._pens.get(hand.slot)
            if pen is None:
                continue
            if pen.cursor is not None:                          # glowing pen tip
                color = hue_to_bgr(pen.hue)
                add_glow(light, pen.cursor, hand.scale * 0.2, color, 0.7)
                cv2.circle(light, ipt(pen.cursor), max(3, int(hand.scale * 0.06)),
                           scale_color((255, 255, 255), 0.85), 2, cv2.LINE_AA)
            if pen.clear_hold > 0.08:                           # "clearing..." progress ring
                sweep = 360.0 * min(1.0, pen.clear_hold / CLEAR_SECONDS)
                r = int(hand.scale * 0.9)
                cv2.ellipse(light, ipt(hand.palm_center), (r, r), 0, -90, -90 + sweep,
                            (230, 230, 255), max(2, int(hand.scale * 0.04)), cv2.LINE_AA)
