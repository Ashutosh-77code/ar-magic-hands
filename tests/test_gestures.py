"""Geometry and gesture detection on synthetic hands."""

import math

import numpy as np

from magic_hands import config
from tests.synthetic import make_hand


def test_scale_is_wrist_to_middle_knuckle_distance():
    hand = make_hand("open", scale=123.0)
    assert abs(hand.scale - 123.0) < 0.5


def test_palm_center_is_midpoint_of_wrist_and_middle_base():
    hand = make_hand("open", center=(300.0, 500.0), scale=200.0)
    expected = (hand.landmarks[0] + hand.landmarks[9]) / 2.0
    assert np.allclose(hand.palm_center, expected)
    # Fingers point up, so the palm centre sits half a hand-scale above the wrist.
    assert abs(hand.palm_center[0] - 300.0) < 1.0
    assert abs(hand.palm_center[1] - (500.0 - 100.0)) < 1.0


def test_scale_follows_distance_from_camera():
    near = make_hand("open", scale=240.0)
    far = make_hand("open", scale=60.0)
    assert near.scale > 3.5 * far.scale


def test_open_hand_is_detected():
    hand = make_hand("open")
    assert hand.is_open and not hand.is_fist
    assert hand.extended_count == 5
    assert hand.gesture == "open"


def test_fist_is_not_open():
    hand = make_hand("fist")
    assert hand.is_fist and not hand.is_open
    assert hand.extended_count == 0
    assert hand.gesture == "fist"


def test_pinch_ratio_small_when_pinching_and_large_when_open():
    pinch = make_hand("pinch")
    open_hand = make_hand("open")
    assert pinch.pinch_ratio < config.PINCH_ON
    assert open_hand.pinch_ratio > config.PINCH_OFF
    assert pinch.gesture == "pinch"


def test_pinch_center_is_between_thumb_and_index_tips():
    hand = make_hand("pinch")
    expected = (hand.landmarks[4] + hand.landmarks[8]) / 2.0
    assert np.allclose(hand.pinch_center, expected)


def test_gestures_do_not_depend_on_hand_rotation():
    for angle in (-120, -60, 0, 45, 90, 180):
        assert make_hand("open", angle_deg=angle).is_open, angle
        assert make_hand("fist", angle_deg=angle).is_fist, angle
        assert make_hand("pinch", angle_deg=angle).pinch_ratio < config.PINCH_ON, angle


def test_gestures_do_not_depend_on_hand_size():
    for scale in (40, 90, 200, 400):
        assert make_hand("open", scale=scale).is_open
        assert make_hand("pinch", scale=scale).pinch_ratio < config.PINCH_ON


def test_angle_zero_is_upright_and_positive_is_clockwise():
    assert abs(make_hand("open", angle_deg=0).angle_deg) < 0.5
    assert abs(make_hand("open", angle_deg=90).angle_deg - 90.0) < 0.5      # pointing right
    assert abs(make_hand("open", angle_deg=-90).angle_deg + 90.0) < 0.5     # pointing left
    assert abs(abs(make_hand("open", angle_deg=180).angle_deg) - 180.0) < 0.5


def test_fingertips_array_matches_landmarks():
    hand = make_hand("open")
    assert hand.fingertips.shape == (5, 2)
    assert np.allclose(hand.fingertips[1], hand.landmarks[8])


def test_a_fist_is_never_a_pinch_even_if_the_thumb_touches_the_index_finger():
    fist = make_hand("fist")
    assert fist.pinch_ratio < config.PINCH_ON      # thumb tip really is close to the index tip...
    assert not fist.pinch_possible                 # ...but with every other finger curled it is a fist
    assert make_hand("pinch").pinch_possible
