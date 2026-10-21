"""Effect 1: Iron Man repulsor.

Open your palm toward the camera and a repulsor glows in its centre. Hold the
pose and it charges: brighter glow, bigger rings, energy rushing inward, and
finally shockwave pulses. Close your hand and it powers down.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from magic_hands import config
from magic_hands.effects.base import Effect
from magic_hands.graphics import (add_glow, add_sprite, clamp, ipt, load_light_sprite, make_palette,
                                  rotate_scale_sprite, scale_color, smoothstep)
from magic_hands.hand_tracker import Hand
from magic_hands.particles import ParticleSystem

# Colour of the energy particles, from white-hot to deep blue.
_SPARK_PALETTE = make_palette([(0.0, (255, 255, 255)), (0.4, (255, 225, 150)), (1.0, (255, 110, 20))])

_BLUE = config.REPULSOR_BLUE
_CORE = config.REPULSOR_CORE
_WHITE = (255, 255, 255)


class _HandState:
    """Per-hand animation state (kept between frames)."""

    def __init__(self) -> None:
        self.open_amt = 0.0       # 0..1: how "on" the repulsor is (eases in and out)
        self.charge = 0.0         # 0..1: how long the palm has been held open
        self.phase = 0.0          # animation clock for pulsing and spinning
        self.pulse_timer = 0.0    # time until the next shockwave
        self.pulses: List[float] = []   # ages of live shockwaves
        self.spark_acc = 0.0      # fractional sparks carried to the next frame
        self.radius = 1.0         # current glow radius in px


class IronManRepulsor(Effect):
    name = "Iron Man Repulsor"
    hint = "Hold an open palm to the camera. The longer you hold it, the more it charges."
    bloom_strength = 1.15

    def __init__(self, assets_dir: Optional[Path] = None, seed: Optional[int] = None) -> None:
        self._states: Dict[int, _HandState] = {}
        self._visible: List[Tuple[Hand, _HandState]] = []
        self._sparks = ParticleSystem(_SPARK_PALETTE, capacity=900, drag=0.3, shrink=0.5, seed=seed)
        # Optional artwork: if assets/repulsor.png exists it replaces the drawn rings.
        self._sprite = load_light_sprite(Path(assets_dir) / "repulsor.png") if assets_dir else None

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

            if hand.is_open:
                st.open_amt = min(1.0, st.open_amt + dt / 0.18)
                st.charge = min(1.0, st.charge + dt / config.REPULSOR_CHARGE_SECONDS)
            else:
                st.open_amt = max(0.0, st.open_amt - dt / 0.12)
                st.charge = max(0.0, st.charge - dt * config.REPULSOR_DISCHARGE_RATE)

            st.phase += dt * (6.0 + 14.0 * st.charge)
            pulse = 1.0 + 0.05 * math.sin(st.phase) + 0.03 * math.sin(st.phase * 2.3)
            st.radius = (hand.scale * config.REPULSOR_RADIUS * (0.75 + 0.25 * smoothstep(st.open_amt))
                         * (1.0 + 0.3 * st.charge) * pulse)

            # Energy rushing into the palm while charging.
            if st.open_amt > 0.5 and st.charge > 0.05:
                st.spark_acc += (15.0 + 90.0 * st.charge) * dt
                count = int(st.spark_acc)
                st.spark_acc -= count
                if count:
                    r = st.radius
                    self._sparks.emit(hand.palm_center, count, speed=(-4.4 * r, -3.2 * r),
                                      life=(0.28, 0.38), size=(1.5, 3.5), ring_radius=r * 1.7)

            # Shockwaves once fully charged.
            if st.charge >= 0.98 and st.open_amt > 0.5:
                st.pulse_timer -= dt
                if st.pulse_timer <= 0.0:
                    st.pulses.append(0.0)
                    st.pulse_timer = 0.85
            st.pulses = [age + dt for age in st.pulses if age + dt < 0.7]

            self._visible.append((hand, st))

        for slot in [s for s in self._states if s not in seen]:
            del self._states[slot]  # hand left the frame
        self._sparks.update(dt)

    # ------------------------------------------------------------------
    def draw(self, light: np.ndarray) -> None:
        for hand, st in self._visible:
            if st.open_amt <= 0.01:
                continue
            a = smoothstep(st.open_amt)
            ch = st.charge
            R = st.radius
            c = ipt(hand.palm_center)

            # Soft blue halo, then a white-hot core.
            add_glow(light, c, R * (1.9 + 0.8 * ch), _BLUE, a * (0.22 + 0.40 * ch))
            add_glow(light, c, R * 0.80, _CORE, a * (0.22 + 0.30 * ch))
            add_glow(light, c, R * 0.34, _WHITE, a * (0.35 + 0.40 * ch))

            if self._sprite is not None:
                art = rotate_scale_sprite(self._sprite, st.phase * 12.0, int(R * 2.6))
                add_sprite(light, cv2.convertScaleAbs(art, alpha=a * (0.6 + 0.4 * ch)), c)
            else:
                self._draw_rings(light, c, R, st, a, ch)

            # Charge meter: an arc that sweeps around the repulsor.
            if 0.02 < ch < 0.999:
                cv2.ellipse(light, c, (int(R * 1.3), int(R * 1.3)), 0, -90, -90 + 360.0 * ch,
                            scale_color(_CORE, a), max(2, int(R * 0.05)), cv2.LINE_AA)

            # Shockwaves.
            for age in st.pulses:
                fade = 1.0 - age / 0.7
                cv2.circle(light, c, int(R * (1.0 + 2.6 * age / 0.7)), scale_color(_BLUE, fade * a),
                           max(1, int(R * 0.05 * fade) + 1), cv2.LINE_AA)

        self._sparks.draw(light)

    # ------------------------------------------------------------------
    @staticmethod
    def _draw_rings(light: np.ndarray, c: Tuple[int, int], R: float, st: _HandState,
                    a: float, ch: float) -> None:
        """The built-in repulsor artwork: two rings, counter-rotating triangles and six lights."""
        bright = a * (0.6 + 0.4 * ch)
        ring_color = scale_color(_BLUE, bright)
        pale = scale_color(_CORE, bright)
        thick = max(2, int(R * 0.07))

        cv2.circle(light, c, int(R), ring_color, thick, cv2.LINE_AA)
        cv2.circle(light, c, int(R * 0.66), ring_color, max(1, int(R * 0.045)), cv2.LINE_AA)

        for direction, offset in ((1.0, 0.0), (-1.0, 60.0)):
            base = math.degrees(st.phase * 0.35) * direction + offset
            tri = np.array([[c[0] + R * 0.66 * math.cos(math.radians(base + 120 * k - 90)),
                             c[1] + R * 0.66 * math.sin(math.radians(base + 120 * k - 90))]
                            for k in range(3)], np.int32)
            cv2.polylines(light, [tri], True, pale, max(1, int(R * 0.03)), cv2.LINE_AA)

        for k in range(6):
            ang = math.radians(-math.degrees(st.phase * 0.5) + k * 60.0)
            p = (c[0] + R * 0.86 * math.cos(ang), c[1] + R * 0.86 * math.sin(ang))
            cv2.circle(light, ipt(p), max(2, int(R * 0.06)), pale, -1, cv2.LINE_AA)
