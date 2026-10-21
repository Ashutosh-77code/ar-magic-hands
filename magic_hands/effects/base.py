"""The interface every effect implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence, Tuple

import numpy as np

from magic_hands.hand_tracker import Hand


class Effect(ABC):
    """A visual effect driven by tracked hands.

    Each frame the app calls ``update`` (move the simulation forward) and then
    ``draw`` (paint light onto a black layer). The app adds that layer to the
    video, so effects never touch the camera image directly.
    """

    #: Name shown in the HUD.
    name: str = "Effect"
    #: One-line instruction shown in the HUD.
    hint: str = ""
    #: How strongly the compositor adds a soft glow around the light (0 = none).
    bloom_strength: float = 1.0

    @abstractmethod
    def update(self, hands: Sequence[Hand], dt: float, size: Tuple[int, int]) -> None:
        """Advance the effect by ``dt`` seconds. ``size`` is the frame (width, height)."""

    @abstractmethod
    def draw(self, light: np.ndarray) -> None:
        """Paint this frame's light onto ``light`` (black uint8 BGR, same size as the frame)."""

    def reset(self) -> None:
        """Forget all state (called when the user switches to this effect)."""

    def on_reselect(self) -> None:
        """Called when the user presses this effect's key while it is already active."""
