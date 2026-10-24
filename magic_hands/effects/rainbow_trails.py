"""Effect 5: Rainbow Trails.

Every fingertip leaves a glowing ribbon that cycles through the rainbow and
fades away. No gesture needed: just wave your hands around.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from magic_hands.colors import hue_to_bgr
from magic_hands.effects.base import Effect
from magic_hands.graphics import add_glow, ipt, scale_color
from magic_hands.hand_tracker import Hand

LIFE = 0.9   # seconds a point of the ribbon stays visible


class _Trail:
    def __init__(self, hue: float) -> None:
        self.hue = hue
        self.width = 8.0
        self.points: List[List[float]] = []   # [x, y, age]


class RainbowTrails(Effect):
    name = "Rainbow Trails"
    hint = "Wave your hands. Every fingertip leaves a glowing rainbow ribbon."
    bloom_strength = 1.3

    def __init__(self, seed: Optional[int] = None) -> None:
        self._trails: Dict[Tuple[int, int], _Trail] = {}
        self._time = 0.0

    def reset(self) -> None:
        self._trails.clear()

    # ------------------------------------------------------------------
    def update(self, hands: Sequence[Hand], dt: float, size: Tuple[int, int]) -> None:
        self._time += dt

        # Age the existing ribbons and drop points that have faded out.
        for trail in self._trails.values():
            for point in trail.points:
                point[2] += dt
            trail.points = [p for p in trail.points if p[2] < LIFE]

        # Add this frame's fingertip positions.
        for hand in hands:
            for finger, tip in enumerate(hand.fingertips):
                trail = self._trails.setdefault((hand.slot, finger), _Trail(hue=finger * 0.19))
                trail.width = hand.scale * 0.075
                trail.points.append([float(tip[0]), float(tip[1]), 0.0])

        for key in [k for k, t in self._trails.items() if not t.points]:
            del self._trails[key]

    # ------------------------------------------------------------------
    def draw(self, light: np.ndarray) -> None:
        for trail in self._trails.values():
            pts = trail.points
            if len(pts) < 2:
                continue
            segments = []
            for i in range(1, len(pts)):
                a, b = pts[i - 1], pts[i]
                age = b[2] / LIFE
                fade = (1.0 - age) ** 1.4
                if fade < 0.03:
                    continue
                color = hue_to_bgr(trail.hue + self._time * 0.08 + age * 0.4)
                segments.append((ipt(a), ipt(b), color, fade, max(1, int(trail.width * fade))))

            for pa, pb, color, fade, thick in segments:            # soft halo first
                cv2.line(light, pa, pb, scale_color(color, 0.35 * fade), int(thick * 2.4) + 1, cv2.LINE_AA)
            for pa, pb, color, fade, thick in segments:            # bright ribbon on top
                cv2.line(light, pa, pb, scale_color(color, fade), thick, cv2.LINE_AA)

            head = pts[-1]
            add_glow(light, (head[0], head[1]), trail.width * 2.4,
                     hue_to_bgr(trail.hue + self._time * 0.08), 0.45)
