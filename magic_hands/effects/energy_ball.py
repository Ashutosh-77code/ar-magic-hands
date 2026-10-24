"""Effect 6: Energy Ball (Kamehameha style).

Bring both hands together and an energy ball forms between your palms. Pull
your hands apart slowly to make it bigger, and the longer you hold it the
more it charges. Snap your hands apart quickly to blast it.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from magic_hands.effects.base import Effect
from magic_hands.graphics import add_glow, clamp, draw_polyline, ipt, make_palette, scale_color, zigzag
from magic_hands.hand_tracker import Hand
from magic_hands.particles import ParticleSystem

ACTIVE_RATIO = 6.5        # the ball exists while the palms are within this many hand-scales
CHARGE_SECONDS = 1.6
BLAST_SPEED = 3.0         # hands moving apart faster than this many hand-scales per second = blast
BLAST_COOLDOWN = 1.0

GLOW = (255, 170, 40)
CORE = (255, 245, 220)
_SPARK_PALETTE = make_palette([(0.0, (255, 255, 255)), (0.4, (255, 215, 120)), (1.0, (255, 110, 20))])


class EnergyBall(Effect):
    name = "Energy Ball"
    bloom_strength = 1.3

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)
        self._sparks = ParticleSystem(_SPARK_PALETTE, capacity=900, drag=0.9, shrink=0.6, seed=seed)
        self.blast_count = 0
        self.reset()

    def reset(self) -> None:
        self._sparks.clear()
        self._amt = 0.0
        self._charge = 0.0
        self._radius = 1.0
        self._phase = 0.0
        self._flash = 0.0
        self._cooldown = 0.0
        self._rings: List[float] = []
        self._pair: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._center = np.zeros(2, np.float32)
        self._angle = 0.0
        self._scale = 100.0
        self._prev_dist: Optional[float] = None
        self._sep_speed = 0.0
        self._spark_acc = 0.0
        self._two_hands = False

    @property
    def hint(self) -> str:  # type: ignore[override]
        if self._two_hands:
            return "Pull your hands apart to grow the ball. Snap them apart to blast!"
        return "Show both hands and bring your palms close together."

    # ------------------------------------------------------------------
    def update(self, hands: Sequence[Hand], dt: float, size: Tuple[int, int]) -> None:
        self._phase += dt * (5.0 + 8.0 * self._charge)
        self._flash = max(0.0, self._flash - dt / 0.5)
        self._cooldown = max(0.0, self._cooldown - dt)
        self._rings = [age + dt for age in self._rings if age + dt < 0.8]
        self._two_hands = len(hands) >= 2

        active = False
        if self._two_hands:
            a, b = hands[0], hands[1]
            pa, pb = a.palm_center, b.palm_center
            dist = float(np.linalg.norm(pb - pa))
            avg = (a.scale + b.scale) / 2.0
            active = dist / avg <= ACTIVE_RATIO
            if active:
                self._pair = (pa.copy(), pb.copy())
                self._center = (pa + pb) / 2.0
                self._angle = math.atan2(float(pb[1] - pa[1]), float(pb[0] - pa[0]))
                self._scale = avg

                target = clamp(0.28 * dist, 0.3 * avg, 1.9 * avg) * (0.45 + 0.55 * self._charge)
                if self._amt <= 0.0:
                    self._radius = target
                self._radius += (target - self._radius) * min(1.0, dt * 10.0)

                # How fast are the hands moving apart? (smoothed)
                if self._prev_dist is not None:
                    speed = (dist - self._prev_dist) / max(dt, 1e-3)
                    self._sep_speed = 0.6 * self._sep_speed + 0.4 * speed
                self._prev_dist = dist
                if (self._charge > 0.5 and self._cooldown <= 0.0
                        and self._sep_speed > BLAST_SPEED * avg):
                    self._blast()

        if active:
            self._amt = min(1.0, self._amt + dt / 0.25)
            self._charge = min(1.0, self._charge + dt / CHARGE_SECONDS)
            self._spark_acc += (30.0 + 80.0 * self._charge) * dt
            count = int(self._spark_acc)
            self._spark_acc -= count
            if count:
                self._sparks.emit(self._center, count, speed=(0.3 * self._scale, 1.1 * self._scale),
                                  life=(0.3, 0.6), size=(1.5, 3.2), ring_radius=self._radius)
        else:
            self._pair = None
            self._prev_dist = None
            self._sep_speed = 0.0
            self._amt = max(0.0, self._amt - dt / 0.2)
            self._charge = max(0.0, self._charge - dt * 1.5)

        self._sparks.update(dt)

    def _blast(self) -> None:
        """Release the stored energy: shockwave, flash and a burst of particles."""
        self.blast_count += 1
        self._rings.append(0.0)
        self._flash = 1.0
        self._charge = 0.0
        self._cooldown = BLAST_COOLDOWN
        self._sparks.emit(self._center, 90, speed=(1.5 * self._scale, 5.0 * self._scale),
                          life=(0.4, 0.9), size=(2.0, 4.5))

    # ------------------------------------------------------------------
    def draw(self, light: np.ndarray) -> None:
        if self._amt > 0.02 and self._pair is not None:
            a = self._amt
            ch = self._charge
            R = self._radius * (0.8 + 0.2 * a)
            c = ipt(self._center)
            pulse = 1.0 + 0.06 * math.sin(self._phase * 1.7)

            add_glow(light, c, R * 2.6 * pulse, GLOW, a * (0.18 + 0.30 * ch))
            add_glow(light, c, R * 1.15 * pulse, GLOW, a * (0.45 + 0.30 * ch))
            add_glow(light, c, R * 0.6, CORE, a * (0.20 + 0.30 * ch))
            add_glow(light, c, R * 0.28, (255, 255, 255), a * (0.30 + 0.35 * ch))

            # Orbiting energy dots, tilted along the line between your hands.
            cos_t, sin_t = math.cos(self._angle), math.sin(self._angle)
            for k in range(26):
                ang = self._phase * (1.0 + 0.15 * (k % 4)) + k * (2.0 * math.pi / 26.0)
                rad = R * (1.15 + 0.22 * math.sin(self._phase * 0.7 + k))
                x, y = rad * math.cos(ang), rad * math.sin(ang) * 0.5
                p = (self._center[0] + x * cos_t - y * sin_t, self._center[1] + x * sin_t + y * cos_t)
                cv2.circle(light, ipt(p), 2 if k % 3 else 3, scale_color(CORE, a), -1, cv2.LINE_AA)

            # Crackling tendrils (re-randomised every frame) and conduits from each palm.
            bolts = []
            for _ in range(4 + int(4 * ch)):
                ang = self._rng.uniform(0.0, 2.0 * math.pi)
                d = np.array([math.cos(ang), math.sin(ang)], np.float32)
                length = R * (1.3 + 0.9 * self._rng.random())
                bolts.append(zigzag(self._center + d * R * 0.7, self._center + d * length,
                                    self._rng, 5, length * 0.12))
            for palm in self._pair:
                direction = palm - self._center
                norm = float(np.linalg.norm(direction))
                if norm > 1.0:
                    surface = self._center + direction / norm * R * 0.9
                    bolts.append(zigzag(palm, surface, self._rng, 6, max(3.0, norm * 0.06)))
            thick = max(1, int(self._scale * 0.025))
            for path in bolts:
                draw_polyline(light, path, scale_color(GLOW, a), thick + 2)
            for path in bolts:
                draw_polyline(light, path, scale_color(CORE, a), max(1, thick - 1))

        # Blast effects stay visible even after the ball itself fades.
        for age in self._rings:
            fade = 1.0 - age / 0.8
            radius = int(self._scale * (0.8 + 7.0 * age / 0.8))
            cv2.circle(light, ipt(self._center), radius, scale_color(GLOW, fade), max(2, int(8 * fade)), cv2.LINE_AA)
            cv2.circle(light, ipt(self._center), int(radius * 0.92), scale_color(CORE, fade * 0.8), 2, cv2.LINE_AA)
        if self._flash > 0.0:
            add_glow(light, self._center, self._scale * 3.5, CORE, self._flash * 0.9)
        self._sparks.draw(light)
