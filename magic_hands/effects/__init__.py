"""The effect classes and a factory that builds them in key order (1, 2, 3)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from magic_hands.effects.base import Effect
from magic_hands.effects.elemental import ElementalHands, FireHands, LightningHands
from magic_hands.effects.repulsor import IronManRepulsor
from magic_hands.effects.sling_ring import DoctorStrangeSlingRing

__all__ = ["Effect", "IronManRepulsor", "DoctorStrangeSlingRing", "ElementalHands",
           "FireHands", "LightningHands", "create_effects"]


def create_effects(assets_dir: Optional[Path] = None, seed: Optional[int] = None) -> List[Effect]:
    """Return the effects in the order of keys 1, 2 and 3."""
    return [
        IronManRepulsor(assets_dir, seed),
        DoctorStrangeSlingRing(assets_dir, seed),
        ElementalHands(seed),
    ]
