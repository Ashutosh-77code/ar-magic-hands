"""Effect 3: Elemental hands (fire and lightning).

Always on while selected, using the full 21-point hand skeleton.
Press 3 again to switch between the two variants.

* Fire: the skeleton is drawn into a mask, blurred heavily, and fed into a
  small "heat" buffer that drifts upward and cools every frame. A colour ramp
  turns that heat into flames. Flame particles also rise from the fingertips.
* Lightning: jagged bolts run along every bone, with random branches and arcs
  between neighbouring fingertips. The shapes are re-randomised every frame.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from magic_hands import config
from magic_hands import landmarks as lm
from magic_hands.effects.base import Effect
from magic_hands.graphics import add_glow, draw_polyline, ipt, make_palette, scale_color, zigzag
from magic_hands.hand_tracker import Hand
from magic_hands.particles import ParticleSystem

# Heat (0 = cold, 1 = hottest) -> flame colour, in BGR.
FIRE_LUT = make_palette([
    (0.00, (0, 0, 0)),
    (0.15, (0, 0, 70)),
    (0.35, (0, 40, 210)),
    (0.60, (0, 130, 255)),
    (0.82, (30, 190, 255)),
    (1.00, (110, 235, 255)),
])

# Flame particle colour over its life: bright yellow -> orange -> red -> gone.
FLAME_PALETTE = make_palette([
    (0.00, (150, 250, 255)),
    (0.30, (0, 200, 255)),
    (0.65, (0, 100, 255)),
    (1.00, (0, 20, 150)),
])


# ==========================================================================
# Fire
# ==========================================================================
class FireHands(Effect):
    name = "Fire"
    hint = "Your hands are on fire. Move them to whip the flames around."
    bloom_strength = 0.7

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)
        self._flames = ParticleSystem(FLAME_PALETTE, capacity=1200, accel=(0.0, -140.0), drag=0.4,
                                      turbulence=900.0, shrink=0.7, fade_power=0.9, seed=seed)
        self._heat: Optional[np.ndarray] = None
        self._xx = self._yy = self._wave = None
        self._time = 0.0
        self._scale_lo = 40.0          # hand size in simulation pixels, remembered between frames
        self._full_size = (0, 0)

    def reset(self) -> None:
        self._heat = None
        self._flames.clear()

    # ------------------------------------------------------------------
    def update(self, hands: Sequence[Hand], dt: float, size: Tuple[int, int]) -> None:
        w, h = size
        self._full_size = size
        self._time += dt

        # The flame simulation runs on a small grid (much faster), then is scaled up for display.
        k = config.FIRE_BUFFER_WIDTH / float(w)
        lw, lh = config.FIRE_BUFFER_WIDTH, max(8, int(round(h * k)))
        if self._heat is None or self._heat.shape != (lh, lw):
            self._heat = np.zeros((lh, lw), np.float32)
            yy, xx = np.mgrid[0:lh, 0:lw]
            self._xx, self._yy = xx.astype(np.float32), yy.astype(np.float32)
            self._wave = (self._yy * 0.14 + self._xx * 0.04).astype(np.float32)  # sway pattern

        # 1. Hand skeleton -> mask -> heavy blur ("glow" shape of the hand).
        mask = np.zeros((lh, lw), np.uint8)
        if hands:
            self._scale_lo = float(np.mean([hd.scale for hd in hands])) * k
            for hand in hands:
                self._draw_hand_mask(mask, hand.landmarks * k, hand.scale * k)
            mask = cv2.GaussianBlur(mask, (0, 0), max(2.0, self._scale_lo * 0.20))

        # 2. Heat buffer: rises, wobbles sideways, cools, and is re-heated by the hand.
        frame_rate_factor = dt * 30.0  # constants are tuned at 30 FPS
        rise = config.FIRE_RISE * self._scale_lo * dt
        sway = max(0.6, self._scale_lo * 0.03)
        map_x = self._xx + sway * np.sin(self._wave + np.float32(self._time * 7.0))
        map_y = self._yy + np.float32(rise)  # sampling from below = content moves up
        heat = cv2.remap(self._heat, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        heat *= config.FIRE_DECAY ** frame_rate_factor
        heat += (mask.astype(np.float32) / 255.0) * config.FIRE_GAIN * frame_rate_factor
        noise = cv2.resize(self._rng.random((lh // 6 + 2, lw // 6 + 2)).astype(np.float32), (lw, lh),
                           interpolation=cv2.INTER_LINEAR)
        heat *= 0.86 + 0.28 * noise  # random flicker
        heat = cv2.GaussianBlur(heat, (0, 0), 1.0)
        np.clip(heat, 0.0, 1.6, out=heat)
        self._heat = heat

        # 3. Flame particles rising from every fingertip.
        for hand in hands:
            s = hand.scale
            for tip in hand.fingertips:
                count = self._rng.poisson(config.FIRE_TIP_RATE * dt)
                if count:
                    self._flames.emit(tip, count, speed=(0.9 * s, 2.0 * s), life=(0.35, 0.75),
                                      size=(0.05 * s, 0.11 * s), direction=-math.pi / 2, spread=0.55)
        self._flames.update(dt)

    @staticmethod
    def _draw_hand_mask(mask: np.ndarray, pts: np.ndarray, scale: float) -> None:
        """Draw the thick skeleton (bones + joints) into ``mask`` (255 = hot)."""
        pts_i = np.round(pts).astype(np.int32)
        thickness = max(2, int(scale * 0.22))
        # The palm is only lightly filled (shrunk toward its centre) so it glows softly.
        palm = pts_i[list(lm.PALM_OUTLINE)]
        centre = palm.mean(axis=0)
        inner = np.round(centre + (palm - centre) * 0.6).astype(np.int32)
        cv2.fillConvexPoly(mask, cv2.convexHull(inner), 200)
        for a, b in lm.HAND_CONNECTIONS:
            cv2.line(mask, ipt(pts_i[a]), ipt(pts_i[b]), 255, thickness, cv2.LINE_AA)
        for i, p in enumerate(pts_i):
            radius = thickness * (1.3 if i in lm.FINGERTIPS else 0.8)
            cv2.circle(mask, ipt(p), int(radius), 255, -1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    def draw(self, light: np.ndarray) -> None:
        if self._heat is not None and self._heat.max() > 0.02:
            index = (np.clip(self._heat * 0.75, 0.0, 1.0) * 255.0).astype(np.uint8)
            fire = FIRE_LUT[index]                                    # heat -> colour
            h, w = light.shape[:2]
            fire = cv2.resize(fire, (w, h), interpolation=cv2.INTER_LINEAR)
            cv2.add(light, fire, dst=light)                           # additive, like real flames
        self._flames.draw(light)


# ==========================================================================
# Lightning
# ==========================================================================
class LightningHands(Effect):
    name = "Lightning"
    hint = "Electricity crackles along your fingers. Spread them to make longer arcs."
    bloom_strength = 1.4   # the compositor's cv2.addWeighted glow gives the bolts their halo

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)
        self._bolts: List[Tuple[np.ndarray, int, int]] = []   # (points, glow thickness, core thickness)
        self._hands: Sequence[Hand] = ()
        self._flicker = 1.0

    def reset(self) -> None:
        self._bolts = []
        self._hands = ()

    # ------------------------------------------------------------------
    def update(self, hands: Sequence[Hand], dt: float, size: Tuple[int, int]) -> None:
        """Generate this frame's bolts. Every point is re-randomised, so bolts never look the same twice."""
        self._hands = hands
        self._bolts = []
        self._flicker = float(self._rng.uniform(0.72, 1.0))
        rng = self._rng

        for hand in hands:
            pts = hand.landmarks
            s = hand.scale
            glow_t = max(2, int(s * 0.045))
            core_t = max(1, int(s * 0.012))

            # A jagged bolt along every bone.
            for a, b in lm.HAND_CONNECTIONS:
                length = float(np.linalg.norm(pts[b] - pts[a]))
                segments = int(np.clip(length / (s * 0.11), 2, 8))
                path = zigzag(pts[a], pts[b], rng, segments, max(2.0, length * 0.22))
                self._bolts.append((path, glow_t, core_t))

            # Random branches that fly off the joints.
            for i in range(lm.NUM_LANDMARKS):
                if rng.random() < config.LIGHTNING_BRANCH_CHANCE:
                    angle = rng.uniform(0.0, 2.0 * math.pi)
                    length = s * rng.uniform(0.3, 0.9)
                    end = pts[i] + length * np.array([math.cos(angle), math.sin(angle)], np.float32)
                    self._bolts.append((zigzag(pts[i], end, rng, 5, length * 0.20),
                                        max(1, glow_t - 1), core_t))

            # Arcs jumping between neighbouring fingertips.
            for a, b in lm.TIP_NEIGHBOURS:
                if rng.random() < config.LIGHTNING_ARC_CHANCE:
                    gap = float(np.linalg.norm(pts[b] - pts[a]))
                    self._bolts.append((zigzag(pts[a], pts[b], rng, 6, max(3.0, gap * 0.3)),
                                        glow_t, core_t))

    # ------------------------------------------------------------------
    def draw(self, light: np.ndarray) -> None:
        glow_color = scale_color(config.LIGHTNING_GLOW, self._flicker)
        core_color = scale_color(config.LIGHTNING_CORE, self._flicker)
        for path, glow_t, core_t in self._bolts:      # thick blue pass
            draw_polyline(light, path, glow_color, glow_t)
        for path, glow_t, core_t in self._bolts:      # thin white-hot pass on top
            draw_polyline(light, path, core_color, core_t)
        for hand in self._hands:                      # energy orb in the palm
            add_glow(light, hand.palm_center, hand.scale * 0.65, config.LIGHTNING_GLOW,
                     0.30 * self._flicker)


# ==========================================================================
# Wrapper used by the app: key "3" selects it, pressing "3" again switches variant
# ==========================================================================
class ElementalHands(Effect):
    hint = "Press 3 again to switch between Fire and Lightning."

    def __init__(self, seed: Optional[int] = None) -> None:
        self._variants: List[Effect] = [FireHands(seed), LightningHands(seed)]
        self._index = 0

    @property
    def active(self) -> Effect:
        return self._variants[self._index]

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"Elemental Hands: {self.active.name}"

    @property
    def bloom_strength(self) -> float:  # type: ignore[override]
        return self.active.bloom_strength

    def update(self, hands: Sequence[Hand], dt: float, size: Tuple[int, int]) -> None:
        self.active.update(hands, dt, size)

    def draw(self, light: np.ndarray) -> None:
        self.active.draw(light)

    def reset(self) -> None:
        for variant in self._variants:
            variant.reset()

    def on_reselect(self) -> None:
        self._index = (self._index + 1) % len(self._variants)
        self.active.reset()
