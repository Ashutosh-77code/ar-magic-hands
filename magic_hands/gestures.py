"""Extra hand gestures built on top of ``Hand.extended``.

``Hand.extended`` is a tuple of five booleans: (thumb, index, middle, ring, pinky).
"""

from __future__ import annotations

from magic_hands.hand_tracker import Hand


def is_pointing(hand: Hand) -> bool:
    """Index finger out, middle/ring/pinky curled. The thumb is ignored."""
    _thumb, index, middle, ring, pinky = hand.extended
    return index and not (middle or ring or pinky)


def is_peace(hand: Hand) -> bool:
    """Index and middle fingers out (a "V"), ring and pinky curled. The thumb is ignored."""
    _thumb, index, middle, ring, pinky = hand.extended
    return index and middle and not (ring or pinky)
