"""Each effect, rendered on synthetic hands at 1280x720."""

import numpy as np

from magic_hands.effects import (DoctorStrangeSlingRing, ElementalHands, FireHands, IronManRepulsor,
                                 LightningHands, create_effects)
from tests.synthetic import make_hand

SIZE = (1280, 720)
DT = 1.0 / 30.0


def run(effect, hands, seconds):
    """Run the effect for ``seconds`` and return the last frame's light layer."""
    light = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
    for _ in range(max(1, int(seconds / DT))):
        light.fill(0)
        effect.update(hands, DT, SIZE)
        effect.draw(light)
    return light


def energy(light, box=None):
    """Total light, optionally inside a box (x0, y0, x1, y1)."""
    if box:
        x0, y0, x1, y1 = box
        light = light[max(0, y0):y1, max(0, x0):x1]
    return int(light.sum())


# ------------------------------------------------------------------ Iron Man
def test_repulsor_glows_at_the_palm_centre_when_open():
    hand = make_hand("open", center=(640, 560), scale=160)
    light = run(IronManRepulsor(seed=1), [hand], 0.8)
    cx, cy = map(int, hand.palm_center)
    near = energy(light, (cx - 120, cy - 120, cx + 120, cy + 120))
    assert near > 0.8 * energy(light)                 # almost all light is around the palm
    assert light[cy, cx].max() > 150                  # bright core


def test_repulsor_is_off_for_a_fist_or_no_hand():
    effect = IronManRepulsor(seed=1)
    assert energy(run(effect, [make_hand("fist")], 0.5)) == 0
    assert energy(run(IronManRepulsor(seed=1), [], 0.5)) == 0


def test_repulsor_gets_brighter_the_longer_the_palm_is_held_open():
    effect = IronManRepulsor(seed=1)
    hand = make_hand("open", scale=160)
    early = energy(run(effect, [hand], 0.3))
    late = energy(run(effect, [hand], 2.5))
    assert late > 1.3 * early


def test_repulsor_size_follows_distance_from_camera():
    near = energy(run(IronManRepulsor(seed=1), [make_hand("open", scale=220)], 0.8))
    far = energy(run(IronManRepulsor(seed=1), [make_hand("open", scale=60)], 0.8))
    assert near > 3 * far


def test_repulsor_powers_down_when_the_hand_closes():
    effect = IronManRepulsor(seed=1)
    run(effect, [make_hand("open")], 2.0)
    assert energy(run(effect, [make_hand("fist")], 1.0)) < 0.05 * 255 * 1280 * 720 * 0.01 + 1


def test_repulsor_tracks_two_hands_independently():
    effect = IronManRepulsor(seed=1)
    hands = [make_hand("open", (400, 560), 150, slot=0), make_hand("open", (880, 560), 150, slot=1)]
    light = run(effect, hands, 0.6)
    assert energy(light, (250, 300, 560, 700)) > 0 and energy(light, (730, 300, 1040, 700)) > 0
    assert energy(light, (560, 0, 730, 720)) < 0.2 * energy(light)


# ------------------------------------------------------------ Doctor Strange
def test_sling_ring_appears_only_when_pinching_and_centres_on_the_pinch():
    effect = DoctorStrangeSlingRing(seed=1)
    assert energy(run(effect, [make_hand("open")], 0.6)) == 0
    hand = make_hand("pinch", center=(640, 560), scale=160)
    light = run(DoctorStrangeSlingRing(seed=1), [hand], 1.0)
    px, py = map(int, hand.pinch_center)
    radius = int(hand.scale * 1.5)
    assert energy(light, (px - radius, py - radius, px + radius, py + radius)) > 0.8 * energy(light)


def test_sling_ring_ignores_a_fist():
    assert energy(run(DoctorStrangeSlingRing(seed=1), [make_hand("fist")], 1.0)) == 0


def test_sling_ring_hysteresis_keeps_the_ring_while_the_pinch_loosens_slightly():
    effect = DoctorStrangeSlingRing(seed=1)
    run(effect, [make_hand("pinch")], 0.6)
    assert effect._states[0].pinching
    # A half-open pinch (ratio between ON and OFF) must not switch the ring off.
    loose = make_hand("pinch")
    loose.landmarks[4] = loose.landmarks[8] + np.array([-0.4 * loose.scale, 0.0], np.float32)
    assert 0.3 < loose.pinch_ratio < 0.5
    run(effect, [loose], 0.2)
    assert effect._states[0].pinching


def test_sling_ring_spin_speed_depends_on_hand_angle():
    def speed(angle):
        effect = DoctorStrangeSlingRing(seed=1)
        run(effect, [make_hand("pinch", angle_deg=angle)], 0.3)
        return effect._states[0].speed
    assert speed(60) > speed(0) > speed(-60)


def test_sling_ring_keeps_roughly_30_to_50_sparks_alive():
    effect = DoctorStrangeSlingRing(seed=1)
    run(effect, [make_hand("pinch", scale=170)], 2.0)
    assert 25 <= effect.spark_count <= 70


def test_sling_ring_rotates_over_time():
    effect = DoctorStrangeSlingRing(seed=1)
    hand = make_hand("pinch")
    first = run(effect, [hand], 1.0).copy()
    second = run(effect, [hand], 0.4)
    assert not np.array_equal(first, second)


def test_sling_ring_uses_a_custom_png_when_provided():
    import tempfile
    from pathlib import Path
    import cv2
    with tempfile.TemporaryDirectory() as tmp:
        rgba = np.zeros((128, 128, 4), np.uint8)
        cv2.circle(rgba, (64, 64), 50, (255, 0, 255, 255), 6)     # magenta ring
        cv2.imwrite(str(Path(tmp) / "mandala.png"), rgba)
        effect = DoctorStrangeSlingRing(assets_dir=Path(tmp), seed=1)
        assert effect.using_custom_image
        light = run(effect, [make_hand("pinch", scale=160)], 1.0)
        assert light.any()
    assert not DoctorStrangeSlingRing(assets_dir=Path("/definitely/missing"), seed=1).using_custom_image


# ------------------------------------------------------------------- Fire
def test_fire_burns_on_and_above_the_hand():
    hand = make_hand("open", center=(640, 600), scale=160)
    light = run(FireHands(seed=1), [hand], 1.5)
    above_hand = energy(light, (440, 0, 840, 280))      # flames rise above the fingertips
    assert above_hand > 0
    assert light[..., 2].sum() > light[..., 0].sum() * 2   # fire is red/orange, not blue


def test_fire_dies_out_after_the_hands_leave():
    effect = FireHands(seed=1)
    burning = energy(run(effect, [make_hand("open")], 1.5))
    gone = energy(run(effect, [], 3.0))
    assert gone < 0.05 * burning


def test_fire_survives_a_frame_size_change():
    effect = FireHands(seed=1)
    effect.update([make_hand("open")], DT, (1280, 720))
    light = np.zeros((480, 640, 3), np.uint8)
    effect.update([make_hand("open", (320, 400), 90)], DT, (640, 480))
    effect.draw(light)


# -------------------------------------------------------------- Lightning
def test_lightning_is_re_randomised_every_frame():
    effect = LightningHands(seed=1)
    hand = make_hand("open")
    light_a = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
    light_b = np.zeros_like(light_a)
    effect.update([hand], DT, SIZE); effect.draw(light_a)
    effect.update([hand], DT, SIZE); effect.draw(light_b)
    assert light_a.any() and not np.array_equal(light_a, light_b)


def test_lightning_is_blue_white_and_stays_near_the_hand():
    hand = make_hand("open", center=(640, 560), scale=150)
    light = run(LightningHands(seed=1), [hand], 0.3)
    assert light[..., 0].sum() >= light[..., 2].sum()          # blue channel dominates
    xs = np.nonzero(light.any(axis=(0, 2)))[0]
    assert xs.min() > 640 - 4 * hand.scale and xs.max() < 640 + 4 * hand.scale


def test_lightning_draws_nothing_without_hands():
    assert energy(run(LightningHands(seed=1), [], 0.3)) == 0


def test_lightning_uses_strong_bloom_for_its_glow():
    assert LightningHands().bloom_strength > FireHands().bloom_strength


# ---------------------------------------------------------- Elemental / registry
def test_elemental_switches_between_fire_and_lightning_when_reselected():
    effect = ElementalHands(seed=1)
    assert "Fire" in effect.name
    effect.on_reselect()
    assert "Lightning" in effect.name
    effect.on_reselect()
    assert "Fire" in effect.name


def test_create_effects_returns_three_effects_in_key_order():
    names = [e.name for e in create_effects(None, seed=1)]
    assert names[0].startswith("Iron Man") and names[1].startswith("Doctor Strange")
    assert names[2].startswith("Elemental")


def test_every_effect_survives_every_pose_and_empty_frames():
    for effect in create_effects(None, seed=2):
        for pose in ("open", "fist", "pinch"):
            run(effect, [make_hand(pose)], 0.2)
        run(effect, [], 0.1)
        run(effect, [make_hand("open", (-200, -200), 100)], 0.1)       # hand off-screen
        run(effect, [make_hand("open", (1400, 900), 600)], 0.1)        # huge, partly off-screen
        effect.reset()
