"""Effect 7: Galaxy and black hole.

Open palm: a spiral galaxy spins above your hand.
Make a fist: it collapses into a bright, fast-spinning black hole.
Open your hand again: supernova!
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from magic_hands.effects.base import Effect
from magic_hands.graphics import add_glow, ipt, make_palette, scale_color, smoothstep
from magic_hands.hand_tracker import Hand
from magic_hands.particles import ParticleSystem

N_STARS = 420
TILT = -0.35   # radians: tilts the galaxy disc

# Star colour from the centre (warm white) out to the rim (blue).
_STAR_PALETTE = make_palette([(0.0, (215, 240, 255)), (0.35, (120, 175, 255)),
                              (0.65, (205, 110, 255)), (1.0, (255, 130, 70))])
_NOVA_PALETTE = make_palette([(0.0, (230, 250, 255)), (0.4, (60, 200, 255)), (1.0, (0, 60, 220))])


class _State:
    def __init__(self) -> None:
        self.gal = 0.0          # 0..1 how visible the galaxy is
        self.collapse = 0.0     # 0 = open galaxy, 1 = black hole
        self.rot = 0.0
        self.armed = False      # a black hole is ready to go supernova when you open your hand
        self.center = np.zeros(2, np.float32)
        self.radius = 100.0
        self.scale = 100.0
        self.rings: List[float] = []


class GalaxyHands(Effect):
    name = "Galaxy"
    hint = "Open palm: galaxy. Fist: black hole. Open again: supernova!"
    bloom_strength = 1.2

    def __init__(self, seed: Optional[int] = None) -> None:
        rng = np.random.default_rng(seed)
        self._u = (rng.random(N_STARS) ** 0.65).astype(np.float32)               # distance from the centre, 0..1
        arm = rng.integers(0, 3, N_STARS)
        jitter = rng.normal(0.0, 0.14, N_STARS)
        self._theta0 = (arm * (2.0 * math.pi / 3.0) + self._u * 5.0 + jitter).astype(np.float32)
        self._bright = rng.uniform(0.45, 1.0, N_STARS).astype(np.float32)
        self._big = rng.random(N_STARS) < 0.35                                    # some larger stars
        self._states: Dict[int, _State] = {}
        self._flash = 0.0
        self._sparks = ParticleSystem(_NOVA_PALETTE, capacity=900, drag=1.0, shrink=0.6, seed=seed)
        self.supernova_count = 0

    def reset(self) -> None:
        self._states.clear()
        self._sparks.clear()
        self._flash = 0.0

    # ------------------------------------------------------------------
    def update(self, hands: Sequence[Hand], dt: float, size: Tuple[int, int]) -> None:
        self._flash = max(0.0, self._flash - dt / 0.6)
        seen = set()
        for hand in hands:
            seen.add(hand.slot)
            st = self._states.setdefault(hand.slot, _State())

            if hand.is_open:
                st.gal = min(1.0, st.gal + dt / 0.3)
                st.collapse = max(0.0, st.collapse - dt / 0.35)
                if st.armed:
                    st.armed = False
                    self._supernova(st, hand)
            elif hand.is_fist:
                st.gal = min(1.0, st.gal + dt / 0.3)
                st.collapse = min(1.0, st.collapse + dt / 0.4)
                if st.collapse > 0.8:
                    st.armed = True
            else:                                           # any other pose: fade out
                st.gal = max(0.0, st.gal - dt / 0.3)
                st.collapse = max(0.0, st.collapse - dt / 0.35)
                if st.gal < 0.05:
                    st.armed = False

            st.rot += dt * (0.9 + 3.2 * smoothstep(st.collapse))
            st.center = hand.palm_center.copy()
            st.scale = hand.scale
            st.radius = hand.scale * 1.4 * (1.0 - 0.7 * smoothstep(st.collapse))
            st.rings = [age + dt for age in st.rings if age + dt < 0.9]

        for slot in [s for s in self._states if s not in seen]:
            del self._states[slot]
        self._sparks.update(dt)

    def _supernova(self, st: _State, hand: Hand) -> None:
        self.supernova_count += 1
        self._flash = 1.0
        st.rings.append(0.0)
        self._sparks.emit(hand.palm_center, 110, speed=(1.2 * hand.scale, 4.5 * hand.scale),
                          life=(0.5, 1.1), size=(1.5, 4.0))

    # ------------------------------------------------------------------
    def draw(self, light: np.ndarray) -> None:
        for st in self._states.values():
            for age in st.rings:                           # supernova shockwave
                fade = 1.0 - age / 0.9
                cv2.circle(light, ipt(st.center), int(st.scale * (0.6 + 6.0 * age / 0.9)),
                           scale_color((60, 200, 255), fade), max(2, int(9 * fade)), cv2.LINE_AA)

            if st.gal <= 0.02:
                continue
            s = smoothstep(st.collapse)
            flatten = 0.62 - 0.38 * s                      # the disc flattens as it collapses
            cx, cy = float(st.center[0]), float(st.center[1])

            # Differential rotation: stars near the centre orbit faster than those at the rim.
            angle = self._theta0 + st.rot * (1.7 - 1.2 * self._u)
            r = self._u * st.radius
            x = r * np.cos(angle)
            y = r * np.sin(angle) * flatten
            px = cx + x * math.cos(TILT) - y * math.sin(TILT)
            py = cy + x * math.sin(TILT) + y * math.cos(TILT)

            colors = _STAR_PALETTE[(self._u * 255).astype(np.int32)].astype(np.float32)
            colors *= (self._bright * st.gal * (1.5 + 0.8 * s))[:, None]
            colors = np.clip(colors, 0, 255).astype(np.int32)
            for i in np.argsort(self._bright):
                c = colors[i]
                cv2.circle(light, (int(px[i]), int(py[i])), 2 if self._big[i] else 1,
                           (int(c[0]), int(c[1]), int(c[2])), -1, cv2.LINE_AA)

            # Soft coloured clouds drifting along the spiral arms.
            for k, cloud in enumerate(((255, 120, 60), (230, 90, 220), (255, 160, 90))):
                a = st.rot * 0.6 + k * (2.0 * math.pi / 3.0)
                rr = st.radius * 0.55
                gx, gy = rr * math.cos(a), rr * math.sin(a) * flatten
                add_glow(light, (cx + gx * math.cos(TILT) - gy * math.sin(TILT),
                                 cy + gx * math.sin(TILT) + gy * math.cos(TILT)),
                         st.radius * 0.5, cloud, 0.22 * st.gal * (1.0 - 0.6 * s))
            add_glow(light, (cx, cy), st.radius * 0.6, (200, 235, 255), (0.38 + 0.45 * s) * st.gal)
            if s > 0.4:                                    # bright accretion ring around the black hole
                ring = (int(st.radius * 0.9), max(2, int(st.radius * 0.9 * flatten)))
                cv2.ellipse(light, ipt((cx, cy)), ring, math.degrees(TILT), 0, 360,
                            scale_color((120, 200, 255), s * st.gal), max(2, int(st.scale * 0.04)), cv2.LINE_AA)

        if self._flash > 0.0:
            for st in self._states.values():
                add_glow(light, st.center, st.scale * 3.0, (230, 250, 255), self._flash * 0.9)
        self._sparks.draw(light)
