"""A small, fast particle engine built on NumPy arrays.

One ``ParticleSystem`` owns a fixed-size pool of particles. Particles are
spawned with ``emit``, moved with ``update`` and drawn with ``draw``. They
fade through a colour palette as they age and are removed when their life ends.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import cv2
import numpy as np


class ParticleSystem:
    """Spawns, moves, fades and draws particles.

    Args:
        palette: (N, 3) uint8 BGR colour ramp. Index 0 = newborn, last = about to die.
        capacity: maximum number of live particles (extra emits are dropped).
        accel: constant acceleration in px/s^2, e.g. (0, -200) pushes flames upward.
        drag: fraction of velocity lost per second (0 = none).
        turbulence: random sideways wobble in px/s^2 (flickering flames).
        shrink: how much particles shrink over their life (0 = never, 1 = to nothing).
        fade_power: brightness falloff; 1 = linear fade, higher = fades faster at the end.
    """

    def __init__(self, palette: np.ndarray, capacity: int = 1500,
                 accel: Tuple[float, float] = (0.0, 0.0), drag: float = 0.0,
                 turbulence: float = 0.0, shrink: float = 0.6, fade_power: float = 1.0,
                 seed: Optional[int] = None) -> None:
        self.palette = palette
        self.capacity = int(capacity)
        self.accel = np.asarray(accel, np.float32)
        self.drag = float(drag)
        self.turbulence = float(turbulence)
        self.shrink = float(shrink)
        self.fade_power = float(fade_power)
        self._rng = np.random.default_rng(seed)

        self.pos = np.zeros((self.capacity, 2), np.float32)
        self.vel = np.zeros((self.capacity, 2), np.float32)
        self.life = np.zeros(self.capacity, np.float32)      # seconds left
        self.max_life = np.ones(self.capacity, np.float32)   # life at birth
        self.size = np.ones(self.capacity, np.float32)       # radius in px at birth
        self.count = 0

    def __len__(self) -> int:
        return self.count

    def clear(self) -> None:
        """Remove every particle."""
        self.count = 0

    # ------------------------------------------------------------------
    def emit(self, origin, count: Union[int, float], speed: Tuple[float, float] = (50.0, 150.0),
             life: Tuple[float, float] = (0.4, 0.8), size: Tuple[float, float] = (2.0, 4.0),
             direction: Optional[float] = None, spread: float = math.pi,
             ring_radius: float = 0.0) -> int:
        """Spawn ``count`` particles at each origin point. Returns how many were created.

        Args:
            origin: one (x, y) point or an array of points.
            count: particles to spawn per origin point.
            speed: (min, max) speed in px/s. Negative speeds move *toward* the origin.
            life: (min, max) lifetime in seconds.
            size: (min, max) starting radius in px.
            direction: launch angle in radians (image coords: -pi/2 is straight up).
                       ``None`` launches in every direction.
            spread: random +/- angle around ``direction`` (ignored when it is None).
            ring_radius: spawn on a circle of this radius around the origin,
                         moving along the radial direction (outward for positive speed).
        """
        count = int(count)
        origins = np.atleast_2d(np.asarray(origin, np.float32))
        total = min(count * len(origins), self.capacity - self.count)
        if count <= 0 or total <= 0:
            return 0

        base = np.repeat(origins, count, axis=0)[:total]
        if direction is None:
            angle = self._rng.uniform(0.0, 2.0 * math.pi, total)
        else:
            angle = direction + self._rng.uniform(-spread, spread, total)
        unit = np.stack([np.cos(angle), np.sin(angle)], axis=1).astype(np.float32)

        s = slice(self.count, self.count + total)
        self.pos[s] = base + unit * ring_radius
        self.vel[s] = unit * self._rng.uniform(speed[0], speed[1], total)[:, None].astype(np.float32)
        lifetimes = self._rng.uniform(life[0], life[1], total).astype(np.float32)
        self.life[s] = lifetimes
        self.max_life[s] = lifetimes
        self.size[s] = self._rng.uniform(size[0], size[1], total).astype(np.float32)
        self.count += total
        return total

    # ------------------------------------------------------------------
    def update(self, dt: float) -> None:
        """Advance every particle by ``dt`` seconds and drop the dead ones."""
        n = self.count
        if n == 0:
            return
        vel = self.vel[:n]
        vel += self.accel * dt
        if self.drag > 0.0:
            vel *= max(0.0, 1.0 - self.drag * dt)
        if self.turbulence > 0.0:
            vel[:, 0] += self._rng.normal(0.0, self.turbulence * dt, n).astype(np.float32)
        self.pos[:n] += vel * dt
        self.life[:n] -= dt

        alive = self.life[:n] > 0.0
        kept = int(alive.sum())
        if kept < n:
            for arr in (self.pos, self.vel, self.life, self.max_life, self.size):
                arr[:kept] = arr[:n][alive]
            self.count = kept

    # ------------------------------------------------------------------
    def draw(self, light: np.ndarray, brightness: float = 1.0) -> None:
        """Draw every particle as a soft-coloured dot on the light layer."""
        n = self.count
        if n == 0:
            return
        age = np.clip(1.0 - self.life[:n] / self.max_life[:n], 0.0, 1.0)
        index = (age * (len(self.palette) - 1)).astype(np.int32)
        fade = (1.0 - age) ** self.fade_power
        colors = self.palette[index].astype(np.float32) * (fade * brightness)[:, None]
        colors = np.clip(colors, 0, 255).astype(np.int32)
        radii = np.maximum(1, (self.size[:n] * (1.0 - self.shrink * age)).astype(np.int32))
        xs = self.pos[:n, 0].astype(np.int32)
        ys = self.pos[:n, 1].astype(np.int32)

        # Draw dim particles first so bright ones are never painted over by dim ones.
        for i in np.argsort(fade):
            c = colors[i]
            cv2.circle(light, (int(xs[i]), int(ys[i])), int(radii[i]),
                       (int(c[0]), int(c[1]), int(c[2])), -1, cv2.LINE_AA)
