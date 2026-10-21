"""MediaPipe hand landmark indices and skeleton connections.

Defined here (instead of imported from MediaPipe) so the rest of the code,
and the unit tests, do not need MediaPipe installed.

Reference: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
"""

from __future__ import annotations

from typing import Tuple

WRIST = 0

THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

NUM_LANDMARKS = 21

# Fingertips in thumb -> pinky order.
FINGERTIPS: Tuple[int, ...] = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)

# (tip, pip) pairs for the four non-thumb fingers. A finger counts as
# "extended" when its tip is farther from the wrist than its PIP joint.
FINGER_TIP_PIP: Tuple[Tuple[int, int], ...] = (
    (INDEX_TIP, INDEX_PIP),
    (MIDDLE_TIP, MIDDLE_PIP),
    (RING_TIP, RING_PIP),
    (PINKY_TIP, PINKY_PIP),
)

# Bones of the hand skeleton as (from, to) landmark pairs.
HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                   # outer edge of the palm
)

# Landmarks that outline the palm (used to fill the palm in the fire mask).
PALM_OUTLINE: Tuple[int, ...] = (WRIST, THUMB_CMC, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)

# Neighbouring fingertip pairs (lightning arcs jump between these).
TIP_NEIGHBOURS: Tuple[Tuple[int, int], ...] = (
    (THUMB_TIP, INDEX_TIP),
    (INDEX_TIP, MIDDLE_TIP),
    (MIDDLE_TIP, RING_TIP),
    (RING_TIP, PINKY_TIP),
)
