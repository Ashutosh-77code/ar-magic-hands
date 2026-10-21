"""The app's per-frame pipeline and key handling, without a camera or window."""

import numpy as np

from magic_hands.app import MagicHandsApp, Settings, parse_args
from magic_hands.hand_tracker import HandTracker
from magic_hands.ui import draw_hud, draw_skeleton
from tests.synthetic import make_hand
from tests.test_tracker import FakeDetector, normalised


def make_app(mode=1):
    det = FakeDetector()
    app = MagicHandsApp(Settings(mode=mode), tracker=HandTracker(smoothing=0.0, detector=det))
    return app, det


def frame():
    return np.full((720, 1280, 3), 50, np.uint8)


def test_process_frame_adds_light_for_the_active_effect():
    app, det = make_app(mode=1)
    det.queue = [normalised(make_hand("open", (640, 600), 160), 1280, 720)]
    base = frame()
    out = app.process_frame(base.copy(), 1 / 30)
    assert out.shape == base.shape and (out >= base).all() and (out > base).any()
    assert len(app.hands) == 1


def test_process_frame_without_hands_returns_the_unchanged_frame():
    app, det = make_app(mode=2)
    base = frame()
    out = app.process_frame(base.copy(), 1 / 30)
    assert np.array_equal(out, base)


def test_keys_switch_modes_and_pressing_3_twice_toggles_fire_and_lightning():
    app, _ = make_app()
    assert app.mode == 0
    app.handle_key(ord("2")); assert app.mode == 1
    app.handle_key(ord("3")); assert app.mode == 2 and "Fire" in app.effect.name
    app.handle_key(ord("3")); assert "Lightning" in app.effect.name
    app.handle_key(ord("3")); assert "Fire" in app.effect.name
    app.handle_key(ord("1")); assert app.mode == 0
    app.handle_key(ord("9")); assert app.mode == 0          # unknown mode is ignored


def test_quit_keys_and_toggles():
    app, _ = make_app()
    assert app.handle_key(ord("q")) is False
    assert app.handle_key(27) is False
    assert app.handle_key(ord("x")) is True
    app.handle_key(ord("h")); assert app.show_help
    app.handle_key(ord("d")); assert app.show_skeleton
    app.handle_key(ord("s")); assert app._screenshot_requested


def test_every_mode_renders_through_the_full_pipeline():
    for key in "123":
        app, det = make_app()
        app.handle_key(ord(key))
        det.queue = [normalised(make_hand("open", (400, 600), 150, slot=0), 1280, 720),
                     normalised(make_hand("pinch", (900, 600), 150, slot=1), 1280, 720)]
        for _ in range(10):
            out = app.process_frame(frame(), 1 / 30)
        assert out.shape == (720, 1280, 3)


def test_frame_size_changes_are_handled():
    app, det = make_app(mode=3)
    app.process_frame(frame(), 1 / 30)
    app.process_frame(np.zeros((480, 640, 3), np.uint8), 1 / 30)


def test_hud_and_skeleton_draw_without_errors_at_several_sizes():
    for w, h in ((1280, 720), (640, 480), (1920, 1080)):
        img = np.zeros((h, w, 3), np.uint8)
        hands = [make_hand("open", (w // 2, int(h * 0.8)), h / 5)]
        draw_hud(img, "Test", "hint", 59.9, hands, 1, ["A", "B", "C"], show_help=True, recording=True)
        draw_hud(img, "Test", "hint", 0.0, [], 0, ["A", "B", "C"])
        draw_skeleton(img, hands)
        assert img.any()


def test_command_line_parsing():
    s = parse_args(["--mode", "3", "--no-mirror", "--camera", "1", "--max-hands", "1"])
    assert (s.mode, s.mirror, s.camera, s.max_hands) == (3, False, 1, 1)
    d = parse_args([])
    assert d.mirror and d.camera == 0 and d.mode == 1
