"""HandTracker with a fake detector (no MediaPipe, no camera)."""

import numpy as np

from magic_hands.hand_tracker import HandTracker
from tests.synthetic import make_hand


class FakeDetector:
    """Returns whatever hands we queue up, as normalised landmarks."""

    def __init__(self):
        self.queue = []
        self.timestamps = []
        self.last_shape = None
        self.closed = False

    def detect(self, rgb, timestamp_ms):
        self.timestamps.append(timestamp_ms)
        self.last_shape = rgb.shape
        return self.queue

    def close(self):
        self.closed = True


def normalised(hand, w, h):
    pts = hand.landmarks.copy()
    pts[:, 0] /= w
    pts[:, 1] /= h
    return pts, hand.label, hand.score


def test_landmarks_are_mapped_to_full_resolution_pixels():
    det = FakeDetector()
    tracker = HandTracker(process_width=320, smoothing=0.0, detector=det)
    frame = np.zeros((720, 1280, 3), np.uint8)
    original = make_hand("open", (640, 600), 150)
    det.queue = [normalised(original, 1280, 720)]
    hands = tracker.process(frame)
    assert len(hands) == 1
    assert np.allclose(hands[0].landmarks, original.landmarks, atol=0.01)
    assert det.last_shape == (180, 320, 3)               # detector saw the shrunken frame


def test_small_frames_are_not_upscaled():
    det = FakeDetector()
    tracker = HandTracker(process_width=640, detector=det)
    tracker.process(np.zeros((240, 320, 3), np.uint8))
    assert det.last_shape == (240, 320, 3)


def test_hands_get_stable_slots_sorted_left_to_right():
    det = FakeDetector()
    tracker = HandTracker(smoothing=0.0, detector=det)
    right = make_hand("open", (900, 600), 150, label="Right")
    left = make_hand("open", (300, 600), 150, label="Left")
    det.queue = [normalised(right, 1280, 720), normalised(left, 1280, 720)]   # reported in the "wrong" order
    hands = tracker.process(np.zeros((720, 1280, 3), np.uint8))
    assert [h.slot for h in hands] == [0, 1]
    assert hands[0].wrist[0] < hands[1].wrist[0]
    assert hands[0].label == "Left"


def test_no_hands_returns_an_empty_list():
    det = FakeDetector()
    tracker = HandTracker(detector=det)
    assert tracker.process(np.zeros((480, 640, 3), np.uint8)) == []


def test_timestamps_strictly_increase():
    det = FakeDetector()
    tracker = HandTracker(detector=det)
    frame = np.zeros((120, 160, 3), np.uint8)
    for _ in range(50):
        tracker.process(frame)
    assert all(b > a for a, b in zip(det.timestamps, det.timestamps[1:]))


def test_smoothing_reduces_jitter_but_fast_motion_still_follows():
    det = FakeDetector()
    tracker = HandTracker(smoothing=0.6, detector=det)
    frame = np.zeros((720, 1280, 3), np.uint8)
    base = make_hand("open", (640, 600), 150)
    det.queue = [normalised(base, 1280, 720)]
    tracker.process(frame)

    jitter = make_hand("open", (644, 600), 150)                 # 4 px of noise
    det.queue = [normalised(jitter, 1280, 720)]
    smoothed = tracker.process(frame)[0]
    assert 640.0 < smoothed.wrist[0] < 644.0                    # moved only part of the way

    jump = make_hand("open", (940, 600), 150)                   # a fast swipe
    det.queue = [normalised(jump, 1280, 720)]
    followed = tracker.process(frame)[0]
    assert followed.wrist[0] > 900.0                            # not left behind


def test_smoothing_resets_when_the_number_of_hands_changes():
    det = FakeDetector()
    tracker = HandTracker(smoothing=0.9, detector=det)
    frame = np.zeros((720, 1280, 3), np.uint8)
    one = make_hand("open", (300, 600), 150)
    det.queue = [normalised(one, 1280, 720)]
    tracker.process(frame)
    two = make_hand("open", (900, 600), 150)
    det.queue = [normalised(one, 1280, 720), normalised(two, 1280, 720)]
    hands = tracker.process(frame)
    assert np.allclose(hands[1].landmarks, two.landmarks, atol=0.01)     # new hand not dragged from elsewhere


def test_close_closes_the_detector():
    det = FakeDetector()
    HandTracker(detector=det).close()
    assert det.closed
