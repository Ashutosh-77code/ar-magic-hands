"""On-screen display: status panel, mode chips, help overlay and debug skeleton."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from magic_hands import landmarks as lm
from magic_hands.hand_tracker import Hand

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (245, 245, 245)
MUTED = (190, 190, 190)
ACCENT = (255, 190, 80)      # light blue (BGR)
DARK = (25, 25, 25)


def put_text(img: np.ndarray, text: str, org, scale: float, color=WHITE, thickness: int = 1) -> None:
    """Text with a dark outline so it stays readable on any background."""
    x, y = int(org[0]), int(org[1])
    cv2.putText(img, text, (x, y), FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def _panel(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, opacity: float = 0.55) -> None:
    """Translucent dark rectangle."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x0 >= x1 or y0 >= y1:
        return
    roi = img[y0:y1, x0:x1]
    cv2.addWeighted(roi, 1.0 - opacity, np.zeros_like(roi), opacity, 0, dst=roi)


def draw_skeleton(img: np.ndarray, hands: Sequence[Hand]) -> None:
    """Debug view: the tracked bones and joints."""
    for hand in hands:
        pts = np.round(hand.landmarks).astype(int)
        for a, b in lm.HAND_CONNECTIONS:
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), (255, 255, 255), 1, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, tuple(p), 3, (0, 255, 120), -1, cv2.LINE_AA)


def draw_hud(img: np.ndarray, effect_name: str, hint: str, fps: float, hands: Sequence[Hand],
             mode_index: int, mode_labels: Sequence[str], show_help: bool = False,
             recording: bool = False) -> None:
    """Draw the heads-up display in place."""
    h, w = img.shape[:2]
    s = max(0.6, h / 720.0)           # scale the UI with the frame size
    pad = int(14 * s)
    line = int(26 * s)

    # ---- Status panel (top left) -------------------------------------
    gestures = ", ".join(hand.gesture for hand in hands) if hands else "none"
    _panel(img, pad // 2, pad // 2, int(w * 0.46) if w > 900 else w - pad // 2, pad + line * 4 + pad)
    y = pad + line - 4
    put_text(img, "AR MAGIC HANDS", (pad, y), 0.62 * s, ACCENT, 2)
    y += line
    put_text(img, f"Mode: {effect_name}", (pad, y), 0.55 * s)
    y += line
    put_text(img, f"FPS: {fps:4.0f}    Hands: {len(hands)} ({gestures})", (pad, y), 0.5 * s, MUTED)
    y += line
    put_text(img, hint, (pad, y), 0.45 * s, MUTED)

    if recording:
        cv2.circle(img, (w - pad - int(8 * s), pad + int(8 * s)), int(7 * s), (0, 0, 230), -1, cv2.LINE_AA)
        put_text(img, "REC", (w - pad - int(60 * s), pad + int(14 * s)), 0.5 * s, WHITE, 1)

    if not hands:
        text = "Show your hand to the camera"
        size = cv2.getTextSize(text, FONT, 0.8 * s, 2)[0]
        put_text(img, text, ((w - size[0]) // 2, int(h * 0.5)), 0.8 * s, WHITE, 2)

    # ---- Mode chips (bottom left) ------------------------------------
    x = pad
    chip_h = int(34 * s)
    y0 = h - pad - chip_h
    for i, label in enumerate(mode_labels):
        text = f"{i + 1}  {label}"
        tw, th = cv2.getTextSize(text, FONT, 0.55 * s, 1)[0]
        x1 = x + tw + int(24 * s)
        if x1 > w - pad and x > pad:          # no room left: continue on a row above
            x, y0 = pad, y0 - chip_h - int(6 * s)
            x1 = x + tw + int(24 * s)
        active = i == mode_index
        _panel(img, x, y0, x1, y0 + int(34 * s), 0.75 if active else 0.5)
        if active:
            cv2.rectangle(img, (x, y0), (x1, y0 + int(34 * s)), ACCENT, 2, cv2.LINE_AA)
        put_text(img, text, (x + int(12 * s), y0 + int(23 * s)), 0.55 * s,
                 ACCENT if active else MUTED, 1)
        x = x1 + int(10 * s)
    put_text(img, "H: help", (x + int(6 * s), y0 + int(23 * s)), 0.5 * s, MUTED)

    # ---- Help overlay --------------------------------------------------
    if show_help:
        lines = [
            "CONTROLS",
            "1   Iron Man repulsor      (open palm, hold to charge)",
            "2   Doctor Strange ring    (pinch thumb + index, tilt to spin)",
            "3   Elemental hands        (press again: Fire / Lightning)",
            "4   Neon painter           (point to draw, open palm 1 s to clear)",
            "5   Rainbow trails         (wave your fingers)",
            "6   Energy ball            (two hands; snap apart to blast)",
            "7   Galaxy                 (open: galaxy, fist: black hole)",
            "V sign for 1 s: photo with countdown",
            "S   screenshot    D   hand skeleton    H   help    Q / ESC   quit",
        ]
        bw, bh = int(660 * s), int((len(lines) + 1) * line)
        x0, y0 = (w - bw) // 2, (h - bh) // 2
        _panel(img, x0, y0, x0 + bw, y0 + bh, 0.75)
        for i, text in enumerate(lines):
            put_text(img, text, (x0 + int(20 * s), y0 + int((i + 1.2) * line)),
                     0.62 * s if i == 0 else 0.52 * s, ACCENT if i == 0 else WHITE, 2 if i == 0 else 1)
