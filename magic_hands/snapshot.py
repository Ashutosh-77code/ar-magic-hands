"""Peace-sign photo booth.

Hold a "V" sign for one second, a 3-2-1 countdown starts, and the frame is
saved when it reaches zero. You can lower your hand during the countdown.
The app calls ``update`` every frame and ``draw`` after the HUD.
"""

from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np

from magic_hands.gestures import is_peace
from magic_hands.hand_tracker import Hand
from magic_hands.ui import ACCENT, FONT, WHITE, put_text

HOLD_SECONDS = 1.0        # how long to hold the V sign
COUNTDOWN_SECONDS = 3.0   # then count down
FLASH_SECONDS = 0.35      # white flash after the photo
COOLDOWN_SECONDS = 2.0    # ignore the gesture right after a photo


class GestureSnapshot:
    def __init__(self) -> None:
        self.hold = 0.0
        self.countdown = None      # seconds left, or None when idle
        self.flash = 0.0
        self.cooldown = 0.0
        self.photos_taken = 0

    def reset(self) -> None:
        self.hold = 0.0
        self.countdown = None
        self.flash = 0.0

    def update(self, hands: Sequence[Hand], dt: float) -> bool:
        """Advance the timers. Returns True on the one frame where a photo should be saved."""
        self.flash = max(0.0, self.flash - dt)
        self.cooldown = max(0.0, self.cooldown - dt)

        if self.countdown is not None:
            self.countdown -= dt
            if self.countdown <= 0.0:
                self.countdown = None
                self.flash = FLASH_SECONDS
                self.cooldown = COOLDOWN_SECONDS
                self.photos_taken += 1
                return True
            return False

        if self.cooldown <= 0.0 and any(is_peace(hand) for hand in hands):
            self.hold += dt
            if self.hold >= HOLD_SECONDS:
                self.hold = 0.0
                self.countdown = COUNTDOWN_SECONDS
        else:
            self.hold = max(0.0, self.hold - dt * 2.0)
        return False

    def draw(self, img: np.ndarray) -> None:
        """Progress bar while holding, big countdown number, and the flash."""
        h, w = img.shape[:2]
        s = max(0.6, h / 720.0)

        if self.countdown is not None:
            text = str(int(math.ceil(self.countdown)))
            size = cv2.getTextSize(text, FONT, 6.0 * s, int(10 * s))[0]
            put_text(img, text, ((w - size[0]) // 2, (h + size[1]) // 2), 6.0 * s, WHITE, int(10 * s))
        elif self.hold > 0.05:
            bar_w, bar_h = int(260 * s), int(10 * s)
            x0, y0 = (w - bar_w) // 2, int(60 * s)
            put_text(img, "Peace sign detected... hold it", (x0 - int(30 * s), y0 - int(12 * s)), 0.6 * s, WHITE, 1)
            cv2.rectangle(img, (x0, y0), (x0 + bar_w, y0 + bar_h), (60, 60, 60), -1)
            cv2.rectangle(img, (x0, y0), (x0 + int(bar_w * min(1.0, self.hold / HOLD_SECONDS)), y0 + bar_h),
                          ACCENT, -1)

        if self.flash > 0.0:
            alpha = min(1.0, self.flash / FLASH_SECONDS)
            white = np.full_like(img, 255)
            cv2.addWeighted(img, 1.0 - alpha, white, alpha, 0, dst=img)
