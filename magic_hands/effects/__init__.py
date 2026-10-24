"""The effect classes and a factory that builds them in key order (1, 2, 3, ...)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from magic_hands.effects.base import Effect
from magic_hands.effects.elemental import ElementalHands, FireHands, LightningHands
from magic_hands.effects.energy_ball import EnergyBall
from magic_hands.effects.galaxy import GalaxyHands
from magic_hands.effects.neon_painter import NeonPainter
from magic_hands.effects.rainbow_trails import RainbowTrails
from magic_hands.effects.repulsor import IronManRepulsor
from magic_hands.effects.sling_ring import DoctorStrangeSlingRing

__all__ = ["Effect", "IronManRepulsor", "DoctorStrangeSlingRing", "ElementalHands", "FireHands",
           "LightningHands", "NeonPainter", "RainbowTrails", "EnergyBall", "GalaxyHands",
           "create_effects"]


def create_effects(assets_dir: Optional[Path] = None, seed: Optional[int] = None) -> List[Effect]:
    """Return the effects in the order of keys 1 to 7.

    To add your own effect: write a subclass of ``Effect``, import it above and
    append it here. The new key (8, 9, ...) works automatically.
    """
    return [
        IronManRepulsor(assets_dir, seed),     # 1
        DoctorStrangeSlingRing(assets_dir, seed),  # 2
        ElementalHands(seed),                  # 3
        NeonPainter(seed),                     # 4
        RainbowTrails(seed),                   # 5
        EnergyBall(seed),                      # 6
        GalaxyHands(seed),                     # 7
    ]
