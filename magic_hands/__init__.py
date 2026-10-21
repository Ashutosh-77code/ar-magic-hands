"""AR Magic Hands: gesture-driven augmented reality effects for your webcam.

The package is split into small, independent pieces:

    hand_tracker   MediaPipe wrapper that turns camera frames into ``Hand`` objects
    particles      A tiny NumPy particle engine (sparks, flames)
    graphics       Additive-light drawing helpers (glows, sprites, lightning paths)
    compositor     Blends the "light layer" onto the camera frame with bloom
    effects/       One class per effect (Iron Man, Sling Ring, Elemental)
    app            The webcam loop, keyboard controls and HUD
"""

__version__ = "1.0.0"
