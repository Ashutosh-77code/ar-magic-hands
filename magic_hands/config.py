"""Tunable constants. Change values here to restyle the whole project.

Colours are BGR tuples (OpenCV order), not RGB.
Distances are expressed relative to the hand's own size ("scale"), which
makes every effect behave the same near and far from the camera.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Gesture detection
# --------------------------------------------------------------------------
# A finger is "extended" when dist(tip, wrist) > dist(pip, wrist) * this ratio.
FINGER_EXTENDED_RATIO = 1.12
# The thumb is "extended" when its tip is at least this far (in hand scales)
# from the base of the index finger.
THUMB_EXTENDED_RATIO = 0.50
# A hand is "open" when at least this many of the 5 digits are extended.
OPEN_MIN_EXTENDED = 4
# Pinch uses hysteresis so it does not flicker at the threshold.
# ratio = dist(thumb tip, index tip) / hand scale
PINCH_ON = 0.30
PINCH_OFF = 0.50

# --------------------------------------------------------------------------
# Hand tracking
# --------------------------------------------------------------------------
# Landmark smoothing: 0 = raw (jittery), 0.9 = very smooth (laggy).
# Fast movements automatically bypass the smoothing, so it never feels sluggish.
LANDMARK_SMOOTHING = 0.35

# --------------------------------------------------------------------------
# Effect 1: Iron Man repulsor
# --------------------------------------------------------------------------
REPULSOR_CHARGE_SECONDS = 1.8     # time holding an open palm to reach full charge
REPULSOR_DISCHARGE_RATE = 2.5     # how fast charge drains when the hand closes
REPULSOR_RADIUS = 0.55            # palm glow radius, in hand scales
REPULSOR_BLUE = (255, 170, 60)
REPULSOR_CORE = (255, 235, 205)

# --------------------------------------------------------------------------
# Effect 2: Doctor Strange sling ring
# --------------------------------------------------------------------------
RING_DIAMETER = 2.7               # mandala diameter, in hand scales
RING_BASE_SPEED = 45.0            # degrees per second with the hand upright
RING_TILT_GAIN = 260.0            # extra deg/s per unit of hand tilt (tilt right = faster)
RING_SPARK_RATE = 90.0            # sparks per second (~45 alive at once)
RING_ORANGE = (0, 135, 255)
RING_YELLOW = (40, 200, 255)

# --------------------------------------------------------------------------
# Effect 3: Elemental hands
# --------------------------------------------------------------------------
FIRE_BUFFER_WIDTH = 360           # the flame simulation runs at this width, then upscales
FIRE_DECAY = 0.90                 # heat kept per frame at 30 FPS
FIRE_GAIN = 0.13                  # heat added by the hand mask per frame at 30 FPS
FIRE_RISE = 2.6                   # how fast flames rise, in hand scales per second
FIRE_TIP_RATE = 55.0              # flame particles per second, per fingertip

LIGHTNING_CORE = (255, 255, 235)
LIGHTNING_GLOW = (255, 150, 50)
LIGHTNING_BRANCH_CHANCE = 0.10    # per joint, per frame
LIGHTNING_ARC_CHANCE = 0.55       # per fingertip pair, per frame

