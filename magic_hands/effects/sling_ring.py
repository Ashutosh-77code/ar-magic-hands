"""Effect 2: Doctor Strange sling ring.

Pinch your thumb and index finger together and a rotating magic circle opens
at the pinch point, throwing off orange sparks. Tilt your hand to spin the
circle faster (tilt right) or reverse it (tilt left).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from magic_hands import config
from magic_hands.effects.base import Effect
from magic_hands.graphics import (add_glow, add_sprite, ipt, load_light_sprite, make_palette,
                                  rotate_scale_sprite, smoothstep)
from magic_hands.hand_tracker import Hand
from magic_hands.particles import ParticleSystem
from magic_hands.procedural import build_mandala

# Sparks start pale yellow and cool to deep orange as they fade.
_SPARK_PALETTE = make_palette([(0.0, (190, 250, 255)), (0.3, (0, 215, 255)),
                               (0.7, (0, 135, 255)), (1.0, (0, 50, 210))])


class _HandState:
    def __init__(self) -> None:
        self.pinching = False   # hysteresis state
        self.amt = 0.0          # 0..1: how open the ring is (eases in and out)
        self.rotation = 0.0     # accumulated rotation in degrees
        self.spark_acc = 0.0    # fractional sparks carried to the next frame
        self.diameter = 1.0
        self.speed = 0.0        # deg/s, for debugging and tests


class DoctorStrangeSlingRing(Effect):
    name = "Doctor Strange Sling Ring"
    hint = "Pinch your thumb and index finger together. Tilt your hand to change the spin."
    bloom_strength = 0.8

    def __init__(self, assets_dir: Optional[Path] = None, seed: Optional[int] = None) -> None:
        self._states: Dict[int, _HandState] = {}
        self._visible: List[Tuple[Hand, _HandState]] = []
        self._sparks = ParticleSystem(_SPARK_PALETTE, capacity=600, drag=1.2, shrink=0.7, seed=seed)
        # Use assets/mandala.png (RGBA) if present, otherwise the built-in drawing.
        loaded = load_light_sprite(Path(assets_dir) / "mandala.png") if assets_dir else None
        self._mandala = loaded if loaded is not None else build_mandala()
        self.using_custom_image = loaded is not None

    def reset(self) -> None:
        self._states.clear()
        self._visible.clear()
        self._sparks.clear()

    # ------------------------------------------------------------------
    def update(self, hands: Sequence[Hand], dt: float, size: Tuple[int, int]) -> None:
        self._visible = []
        seen = set()
        for hand in hands:
            seen.add(hand.slot)
            st = self._states.setdefault(hand.slot, _HandState())

            # Pinch with hysteresis: it must close past PINCH_ON to start, and open past PINCH_OFF to stop.
            # A fist (thumb touching index by accident) never counts as a pinch.
            ratio = hand.pinch_ratio
            valid = hand.pinch_possible
            if st.pinching:
                if ratio > config.PINCH_OFF or not valid:
                    st.pinching = False
            elif valid and ratio < config.PINCH_ON:
                st.pinching = True

            if st.pinching:
                st.amt = min(1.0, st.amt + dt / 0.22)
            else:
                st.amt = max(0.0, st.amt - dt / 0.15)

            # Spin speed depends on the hand's angle: upright = steady spin,
            # tilted right = faster, tilted left = slower or reversed.
            tilt = math.sin(math.radians(hand.angle_deg))
            st.speed = config.RING_BASE_SPEED + config.RING_TILT_GAIN * tilt
            st.rotation += st.speed * dt * (0.4 + 0.6 * st.amt)

            st.diameter = hand.scale * config.RING_DIAMETER * smoothstep(st.amt)

            # Sparks fly outward from the edge of the circle.
            if st.amt > 0.3:
                st.spark_acc += config.RING_SPARK_RATE * dt
                count = int(st.spark_acc)
                st.spark_acc -= count
                if count:
                    s = hand.scale
                    self._sparks.emit(hand.pinch_center, count, speed=(0.7 * s, 1.9 * s),
                                      life=(0.35, 0.65), size=(1.8, 3.6),
                                      ring_radius=st.diameter * 0.47)

            self._visible.append((hand, st))

        for slot in [s for s in self._states if s not in seen]:
            del self._states[slot]
        self._sparks.update(dt)

    # ------------------------------------------------------------------
    def draw(self, light: np.ndarray) -> None:
        for hand, st in self._visible:
            if st.amt <= 0.02 or st.diameter < 12:
                continue
            c = ipt(hand.pinch_center)
            a = smoothstep(st.amt)
            flicker = 0.92 + 0.08 * math.sin(st.rotation * 0.2)

            add_glow(light, c, st.diameter * 0.55, config.RING_ORANGE, 0.20 * a)

            # Two layers spinning in opposite directions give the circle depth.
            outer = rotate_scale_sprite(self._mandala, st.rotation, int(st.diameter))
            inner = rotate_scale_sprite(self._mandala, -st.rotation * 1.6, int(st.diameter * 0.58))
            add_sprite(light, cv2.convertScaleAbs(outer, alpha=a * flicker), c)
            add_sprite(light, cv2.convertScaleAbs(inner, alpha=0.6 * a * flicker), c)

        self._sparks.draw(light)

    @property
    def spark_count(self) -> int:
        """Number of live sparks (handy for tests and the HUD)."""
        return len(self._sparks)
