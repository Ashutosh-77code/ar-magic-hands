"""The webcam application: capture -> track -> effect -> composite -> display."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

from magic_hands import __version__, config
from magic_hands.compositor import composite
from magic_hands.effects import Effect, create_effects
from magic_hands.hand_tracker import Hand, HandTracker
from magic_hands.snapshot import GestureSnapshot
from magic_hands.ui import draw_hud, draw_skeleton

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WINDOW_NAME = "AR Magic Hands"
MODE_LABELS = ["Iron Man", "Sling Ring", "Elemental", "Neon Paint", "Trails", "Energy Ball", "Galaxy"]

CAMERA_BACKENDS = {
    "auto": cv2.CAP_ANY,
    "dshow": getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
    "msmf": getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
    "v4l2": getattr(cv2, "CAP_V4L2", cv2.CAP_ANY),
    "avfoundation": getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY),
}


@dataclass
class Settings:
    """Everything the command line can configure."""
    camera: int = 0
    width: int = 1280
    height: int = 720
    mirror: bool = True
    max_hands: int = 2
    process_width: int = 640
    smoothing: float = config.LANDMARK_SMOOTHING
    model: Optional[Path] = None
    assets: Path = field(default_factory=lambda: PROJECT_ROOT / "assets")
    mode: int = 1
    record: Optional[Path] = None
    backend: str = "auto"
    snapshot: bool = True   # peace-sign photo booth


class MagicHandsApp:
    """Owns the tracker and effects, and runs the main loop.

    ``process_frame`` contains all the per-frame logic and needs no window or
    camera, so it can be unit-tested and reused in other scripts.
    """

    def __init__(self, settings: Settings, tracker: Optional[HandTracker] = None,
                 effects: Optional[List[Effect]] = None) -> None:
        self.settings = settings
        self.tracker = tracker or HandTracker(
            max_hands=settings.max_hands, process_width=settings.process_width,
            smoothing=settings.smoothing, model_path=settings.model)
        self.effects = effects or create_effects(settings.assets)
        self.mode = min(max(settings.mode, 1), len(self.effects)) - 1
        self.show_help = False
        self.show_skeleton = False
        self.fps = 0.0
        self.hands: Sequence[Hand] = []
        self.snapshot = GestureSnapshot() if settings.snapshot else None
        self._light: Optional[np.ndarray] = None
        self._screenshot_requested = False

    # ------------------------------------------------------------------
    @property
    def effect(self) -> Effect:
        return self.effects[self.mode]

    def select_mode(self, index: int) -> None:
        """Switch effect. Pressing the key of the active effect lets it react (e.g. toggle Fire/Lightning)."""
        if index < 0 or index >= len(self.effects):
            return
        if index == self.mode:
            self.effect.on_reselect()
        else:
            self.mode = index
            self.effect.reset()

    def handle_key(self, key: int) -> bool:
        """React to a key press. Returns False when the app should quit."""
        if key in (ord("q"), ord("Q"), 27):
            return False
        if ord("1") <= key <= ord("9"):
            self.select_mode(key - ord("1"))
        elif key in (ord("h"), ord("H")):
            self.show_help = not self.show_help
        elif key in (ord("d"), ord("D")):
            self.show_skeleton = not self.show_skeleton
        elif key in (ord("s"), ord("S")):
            self._screenshot_requested = True
        return True

    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray, dt: float) -> np.ndarray:
        """Track hands, run the active effect, and return the finished frame (no HUD)."""
        self.hands = self.tracker.process(frame)
        if self.snapshot is not None and self.snapshot.update(self.hands, dt):
            self._screenshot_requested = True   # the V-sign countdown finished

        h, w = frame.shape[:2]
        if self._light is None or self._light.shape != frame.shape:
            self._light = np.zeros_like(frame)
        else:
            self._light.fill(0)

        effect = self.effect
        effect.update(self.hands, dt, (w, h))
        effect.draw(self._light)
        return composite(frame, self._light, effect.bloom_strength)

    # ------------------------------------------------------------------
    def run(self) -> int:
        s = self.settings
        cap = open_camera(s.camera, s.width, s.height, s.backend)
        writer = None
        previous = time.perf_counter()
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        print_controls()

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Could not read a frame from the camera. Is another app using it?")
                    return 1
                if s.mirror:
                    frame = cv2.flip(frame, 1)

                now = time.perf_counter()
                dt = min(max(now - previous, 1.0 / 240.0), 0.1)
                previous = now
                instant_fps = 1.0 / dt
                self.fps = 0.9 * self.fps + 0.1 * instant_fps if self.fps else instant_fps

                out = self.process_frame(frame, dt)

                # Clean output (no HUD) goes to screenshots and recordings.
                if self._screenshot_requested:
                    self._screenshot_requested = False
                    print("Saved", save_screenshot(out))
                if s.record:
                    if writer is None:
                        writer = open_writer(s.record, out.shape[1], out.shape[0])
                    writer.write(out)

                if self.show_skeleton:
                    draw_skeleton(out, self.hands)
                draw_hud(out, self.effect.name, self.effect.hint, self.fps, self.hands, self.mode,
                         MODE_LABELS[:len(self.effects)], self.show_help, writer is not None)
                if self.snapshot is not None:
                    self.snapshot.draw(out)
                cv2.imshow(WINDOW_NAME, out)

                key = cv2.waitKey(1) & 0xFF
                if key != 255 and not self.handle_key(key):
                    break
                if window_closed():
                    break
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            self.tracker.close()
            cv2.destroyAllWindows()
        return 0


# ==========================================================================
# Helpers
# ==========================================================================
def open_camera(index: int, width: int, height: int, backend: str = "auto") -> cv2.VideoCapture:
    """Open the webcam or raise a helpful error."""
    cap = cv2.VideoCapture(index, CAMERA_BACKENDS.get(backend, cv2.CAP_ANY))
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {index}. Try --camera 1, close other apps that use the "
            f"webcam, or on macOS allow camera access for your terminal.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # lowest latency (ignored by some drivers)
    return cap


def open_writer(path: Path, width: int, height: int) -> cv2.VideoWriter:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create the video file {path}.")
    print(f"Recording to {path}")
    return writer


def save_screenshot(image: np.ndarray) -> Path:
    folder = PROJECT_ROOT / "screenshots"
    folder.mkdir(exist_ok=True)
    path = folder / f"ar_magic_hands_{datetime.now():%Y%m%d_%H%M%S}.png"
    cv2.imwrite(str(path), image)
    return path


def window_closed() -> bool:
    """True when the user closed the window with the X button."""
    try:
        return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def print_controls() -> None:
    print(f"AR Magic Hands v{__version__}")
    print("  1 Iron Man   2 Sling Ring   3 Elemental (press again: Fire/Lightning)")
    print("  4 Neon Painter   5 Rainbow Trails   6 Energy Ball   7 Galaxy")
    print("  Hold a peace sign (V) for 1 s to take a photo with a countdown")
    print("  H help   D skeleton   S screenshot   Q or ESC quit")


def parse_args(argv: Optional[Sequence[str]] = None) -> Settings:
    p = argparse.ArgumentParser(prog="ar-magic-hands", description="Gesture-based AR effects for your hands.")
    p.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    p.add_argument("--width", type=int, default=1280, help="requested capture width")
    p.add_argument("--height", type=int, default=720, help="requested capture height")
    p.add_argument("--no-mirror", action="store_true", help="do not flip the image horizontally")
    p.add_argument("--max-hands", type=int, default=2, choices=(1, 2), help="hands to track")
    p.add_argument("--process-width", type=int, default=640,
                   help="frames are shrunk to this width for hand detection (lower = faster)")
    p.add_argument("--smoothing", type=float, default=config.LANDMARK_SMOOTHING,
                   help="landmark smoothing 0 (none) to 0.9 (heavy)")
    p.add_argument("--mode", type=int, default=1, choices=tuple(range(1, 8)), help="starting effect (1-7)")
    p.add_argument("--model", type=Path, help="path to hand_landmarker.task (downloaded automatically if omitted)")
    p.add_argument("--assets", type=Path, default=PROJECT_ROOT / "assets", help="folder with mandala.png / repulsor.png")
    p.add_argument("--record", type=Path, help="record the output to this .mp4 file")
    p.add_argument("--backend", choices=sorted(CAMERA_BACKENDS), default="auto", help="camera backend")
    p.add_argument("--no-snapshot", action="store_true", help="turn off the peace-sign photo booth")
    a = p.parse_args(argv)
    return Settings(camera=a.camera, width=a.width, height=a.height, mirror=not a.no_mirror,
                    max_hands=a.max_hands, process_width=a.process_width, smoothing=a.smoothing,
                    model=a.model, assets=a.assets, mode=a.mode, record=a.record, backend=a.backend,
                    snapshot=not a.no_snapshot)


def main(argv: Optional[Sequence[str]] = None) -> int:
    settings = parse_args(argv)
    try:
        return MagicHandsApp(settings).run()
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
