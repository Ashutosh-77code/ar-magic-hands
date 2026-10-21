"""Hand tracking: camera frame in, list of ``Hand`` objects out.

``HandTracker`` wraps the MediaPipe *Tasks* HandLandmarker (the old
``mp.solutions.hands`` API was removed from recent MediaPipe releases).
MediaPipe is imported lazily, so everything else in the project (and the unit
tests) works without it.
"""

from __future__ import annotations

import functools
import math
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from magic_hands import config
from magic_hands import landmarks as lm

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"


# ==========================================================================
# Hand: geometry and gestures for one detected hand
# ==========================================================================
@dataclass(eq=False)
class Hand:
    """One tracked hand, with all positions in full-resolution pixel coordinates.

    Attributes:
        slot: 0 for the left-most hand on screen, 1 for the next. Stable between frames.
        label: "Left" or "Right" as reported by MediaPipe.
        score: handedness confidence, 0..1.
        landmarks: (21, 2) float32 array of (x, y) pixels, MediaPipe landmark order.
    """

    slot: int
    label: str
    score: float
    landmarks: np.ndarray

    # ---- Geometry ----------------------------------------------------
    @functools.cached_property
    def wrist(self) -> np.ndarray:
        return self.landmarks[lm.WRIST]

    @functools.cached_property
    def palm_center(self) -> np.ndarray:
        """Midpoint between the wrist and the base of the middle finger."""
        return (self.landmarks[lm.WRIST] + self.landmarks[lm.MIDDLE_MCP]) / 2.0

    @functools.cached_property
    def scale(self) -> float:
        """Hand size in pixels: distance from the wrist to the middle knuckle.

        It shrinks as the hand moves away from the camera, so effects sized by
        it automatically follow the hand's distance.
        """
        return max(1.0, float(np.linalg.norm(self.landmarks[lm.MIDDLE_MCP] - self.landmarks[lm.WRIST])))

    @functools.cached_property
    def angle_deg(self) -> float:
        """Hand direction in degrees: 0 = fingers up, +90 = fingers pointing right (clockwise)."""
        v = self.landmarks[lm.MIDDLE_MCP] - self.landmarks[lm.WRIST]
        return math.degrees(math.atan2(float(v[0]), float(-v[1])))

    @functools.cached_property
    def fingertips(self) -> np.ndarray:
        """(5, 2) array: thumb, index, middle, ring, pinky tips."""
        return self.landmarks[list(lm.FINGERTIPS)]

    # ---- Gestures ----------------------------------------------------
    @functools.cached_property
    def extended(self) -> Tuple[bool, bool, bool, bool, bool]:
        """Which of (thumb, index, middle, ring, pinky) are stretched out.

        Uses distances only (no angles), so it works at any hand rotation.
        """
        pts = self.landmarks
        wrist = pts[lm.WRIST]
        thumb_gap = np.linalg.norm(pts[lm.THUMB_TIP] - pts[lm.INDEX_MCP]) / self.scale
        flags = [bool(thumb_gap > config.THUMB_EXTENDED_RATIO)]
        for tip, pip in lm.FINGER_TIP_PIP:
            reach_tip = np.linalg.norm(pts[tip] - wrist)
            reach_pip = np.linalg.norm(pts[pip] - wrist)
            flags.append(bool(reach_tip > reach_pip * config.FINGER_EXTENDED_RATIO))
        return tuple(flags)  # type: ignore[return-value]

    @property
    def extended_count(self) -> int:
        return sum(self.extended)

    @property
    def is_open(self) -> bool:
        """True when the hand is open (at least four of five digits extended)."""
        return self.extended_count >= config.OPEN_MIN_EXTENDED

    @property
    def is_fist(self) -> bool:
        return self.extended_count == 0

    @functools.cached_property
    def pinch_ratio(self) -> float:
        """Thumb-to-index gap divided by hand scale (small = pinching)."""
        gap = np.linalg.norm(self.landmarks[lm.THUMB_TIP] - self.landmarks[lm.INDEX_TIP])
        return float(gap / self.scale)

    @functools.cached_property
    def pinch_possible(self) -> bool:
        """False for a fist, where the thumb often touches the index finger by accident.

        A real pinch keeps at least one of the middle, ring or pinky fingers extended.
        """
        return any(self.extended[2:])

    @functools.cached_property
    def pinch_center(self) -> np.ndarray:
        """Midpoint between the thumb tip and the index fingertip."""
        return (self.landmarks[lm.THUMB_TIP] + self.landmarks[lm.INDEX_TIP]) / 2.0

    @property
    def gesture(self) -> str:
        """A readable name for the HUD: pinch, open, fist or other."""
        if self.pinch_possible and self.pinch_ratio < config.PINCH_ON:
            return "pinch"
        if self.is_open:
            return "open"
        if self.is_fist:
            return "fist"
        return "other"


# ==========================================================================
# Model download + MediaPipe Tasks backend
# ==========================================================================
def ensure_model(path: Optional[Path] = None) -> Path:
    """Return the path of the hand landmark model, downloading it on first run (a few MB, one time)."""
    path = Path(path) if path else DEFAULT_MODEL_PATH
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand tracking model to {path} ...")
    partial = path.with_suffix(path.suffix + ".part")
    try:
        urllib.request.urlretrieve(MODEL_URL, partial)
        partial.replace(path)
    except Exception as error:
        if partial.exists():
            partial.unlink()
        raise RuntimeError(
            f"Could not download the hand tracking model ({error}).\n"
            f"Download it manually from:\n  {MODEL_URL}\n"
            f"and save it as:\n  {path}"
        ) from error
    return path


class _TasksDetector:
    """Thin wrapper around mediapipe.tasks.python.vision.HandLandmarker (video mode)."""

    def __init__(self, model_path: Optional[Path], max_hands: int, detection_confidence: float,
                 presence_confidence: float, tracking_confidence: float) -> None:
        import mediapipe as mp  # imported lazily so tests don't need MediaPipe
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(ensure_model(model_path))),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=presence_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> List[Tuple[np.ndarray, str, float]]:
        """Return a list of (normalised (21, 2) landmarks, label, score) per hand."""
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        hands = []
        for points, handedness in zip(result.hand_landmarks, result.handedness):
            xy = np.array([[p.x, p.y] for p in points], np.float32)
            best = handedness[0]
            hands.append((xy, best.category_name, float(best.score)))
        return hands

    def close(self) -> None:
        self._landmarker.close()


# ==========================================================================
# HandTracker: what the rest of the app uses
# ==========================================================================
class HandTracker:
    """Detects hands in BGR camera frames and returns smoothed ``Hand`` objects.

    Args:
        max_hands: how many hands to track.
        process_width: frames wider than this are shrunk before detection. Landmarks
            are normalised, so they are scaled back up to the full-size frame.
        smoothing: 0..1 landmark smoothing (see ``config.LANDMARK_SMOOTHING``).
        model_path: custom location for ``hand_landmarker.task``.
        detector: inject a fake detector (used by the unit tests).
    """

    def __init__(self, max_hands: int = 2, process_width: int = 640,
                 smoothing: float = config.LANDMARK_SMOOTHING,
                 detection_confidence: float = 0.6, presence_confidence: float = 0.5,
                 tracking_confidence: float = 0.5, model_path: Optional[Path] = None,
                 detector=None) -> None:
        self.process_width = int(process_width)
        self.smoothing = float(min(max(smoothing, 0.0), 0.95))
        self._detector = detector or _TasksDetector(
            model_path, max_hands, detection_confidence, presence_confidence, tracking_confidence)
        self._previous: List[np.ndarray] = []
        self._last_timestamp = 0

    def process(self, frame_bgr: np.ndarray) -> List[Hand]:
        """Find hands in one frame. Returns hands sorted left to right."""
        h, w = frame_bgr.shape[:2]
        if w > self.process_width:
            small = cv2.resize(frame_bgr, (self.process_width, int(round(h * self.process_width / w))),
                               interpolation=cv2.INTER_AREA)
        else:
            small = frame_bgr
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        detections = self._detector.detect(rgb, self._next_timestamp())

        found = []
        for xy, label, score in detections:
            pixels = np.asarray(xy, np.float32).copy()
            pixels[:, 0] *= w   # normalised -> full-resolution pixels
            pixels[:, 1] *= h
            found.append((pixels, label, score))
        found.sort(key=lambda item: float(item[0][lm.WRIST, 0]))  # stable slots: left to right

        pixel_sets = self._smooth([item[0] for item in found])
        return [Hand(slot=i, label=found[i][1], score=found[i][2], landmarks=pixel_sets[i])
                for i in range(len(found))]

    def close(self) -> None:
        if hasattr(self._detector, "close"):
            self._detector.close()

    # ------------------------------------------------------------------
    def _next_timestamp(self) -> int:
        """MediaPipe's video mode needs strictly increasing millisecond timestamps."""
        stamp = int(time.perf_counter() * 1000.0)
        self._last_timestamp = max(stamp, self._last_timestamp + 1)
        return self._last_timestamp

    def _smooth(self, current: Sequence[np.ndarray]) -> List[np.ndarray]:
        """Blend each hand with its previous position to remove jitter.

        The blend weight grows with hand speed, so quick movements are not delayed.
        """
        if len(current) != len(self._previous):
            self._previous = [c.copy() for c in current]  # hand count changed: start fresh
            return self._previous.copy()

        result = []
        for now, before in zip(current, self._previous):
            scale = max(1.0, float(np.linalg.norm(now[lm.MIDDLE_MCP] - now[lm.WRIST])))
            speed = float(np.mean(np.linalg.norm(now - before, axis=1))) / scale
            weight_new = min(1.0, (1.0 - self.smoothing) + speed * 4.0)
            result.append((before + (now - before) * weight_new).astype(np.float32))
        self._previous = [r.copy() for r in result]
        return result
