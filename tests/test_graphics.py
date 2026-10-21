"""Drawing helpers, sprite handling and compositing."""

import tempfile
from pathlib import Path

import cv2
import numpy as np

from magic_hands.compositor import bloom, composite
from magic_hands.graphics import (add_glow, add_sprite, load_light_sprite, make_palette,
                                  rotate_scale_sprite, to_light_sprite, zigzag)
from magic_hands.procedural import build_mandala, build_repulsor


def test_palette_hits_its_end_colours():
    pal = make_palette([(0.0, (0, 0, 0)), (1.0, (200, 100, 50))])
    assert pal.shape == (256, 3)
    assert tuple(pal[0]) == (0, 0, 0)
    assert abs(int(pal[-1][0]) - 200) <= 1


def test_add_glow_is_brightest_in_the_middle_and_additive():
    light = np.zeros((200, 200, 3), np.uint8)
    add_glow(light, (100, 100), 60, (255, 100, 0), 1.0)
    assert light[100, 100, 0] > light[100, 150, 0] > 0
    before = light[100, 100].copy()
    add_glow(light, (100, 100), 60, (255, 100, 0), 1.0)
    assert (light[100, 100] >= before).all()


def test_add_glow_off_screen_and_edges_do_not_crash():
    light = np.zeros((100, 100, 3), np.uint8)
    add_glow(light, (-300, -300), 50, (255, 255, 255))   # fully outside
    assert not light.any()
    add_glow(light, (0, 0), 50, (255, 255, 255))          # partly outside
    assert light.any()
    add_glow(light, (50, 50), 1, (255, 255, 255))         # too small: ignored
    add_glow(light, (50, 50), 50, (255, 255, 255), 0.0)   # zero intensity: ignored


def test_add_sprite_clips_at_every_edge():
    light = np.zeros((100, 100, 3), np.uint8)
    sprite = np.full((40, 40, 3), 200, np.uint8)
    for centre in ((0, 0), (99, 99), (-30, 50), (50, 130), (50, 50)):
        add_sprite(light, sprite, centre)
    assert light.any()
    add_sprite(light, sprite, (-500, -500))   # fully outside: no crash


def test_rotate_scale_sprite_output_size_and_clockwise_rotation():
    sprite = np.zeros((100, 100, 3), np.uint8)
    sprite[5:15, 45:55] = 255                # marker at the TOP centre
    out = rotate_scale_sprite(sprite, 90.0, 100)
    assert out.shape == (100, 100, 3)
    ys, xs = np.nonzero(out[:, :, 0])
    assert xs.mean() > 80 and abs(ys.mean() - 50) < 8     # clockwise: top moves to the RIGHT
    assert rotate_scale_sprite(sprite, 33.0, 37).shape == (37, 37, 3)
    assert rotate_scale_sprite(sprite, 0.0, 250).shape == (250, 250, 3)


def test_zigzag_keeps_endpoints_and_is_random_in_between():
    rng = np.random.default_rng(3)
    a, b = np.array([10.0, 10.0]), np.array([200.0, 120.0])
    p1 = zigzag(a, b, rng, 6, 12.0)
    p2 = zigzag(a, b, rng, 6, 12.0)
    assert p1.shape == (7, 2)
    assert np.allclose(p1[0], a) and np.allclose(p1[-1], b)
    assert not np.allclose(p1[1:-1], p2[1:-1])
    assert zigzag(a, a, rng, 4, 5.0).shape[1] == 2        # zero-length bone is safe


def test_to_light_sprite_premultiplies_alpha_and_pads_to_square():
    rgba = np.zeros((20, 40, 4), np.uint8)
    rgba[:, :, :3] = 255
    rgba[:, :20, 3] = 255                    # left half opaque, right half transparent
    sprite = to_light_sprite(rgba)
    assert sprite.shape[0] == sprite.shape[1] == 40
    assert sprite[20, 10].min() == 255 and sprite[20, 30].max() == 0
    assert to_light_sprite(np.full((8, 8), 120, np.uint8)).shape == (8, 8, 3)
    assert to_light_sprite(np.full((8, 8, 3), 120, np.uint8)).shape == (8, 8, 3)


def test_load_light_sprite_handles_missing_corrupt_and_real_files():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        assert load_light_sprite(tmp / "missing.png") is None
        (tmp / "broken.png").write_bytes(b"this is not a png")
        assert load_light_sprite(tmp / "broken.png") is None
        rgba = np.zeros((30, 30, 4), np.uint8)
        rgba[10:20, 10:20] = (0, 128, 255, 255)
        cv2.imwrite(str(tmp / "ok.png"), rgba)
        sprite = load_light_sprite(tmp / "ok.png")
        assert sprite is not None and sprite.shape == (30, 30, 3)
        assert tuple(sprite[15, 15]) == (0, 128, 255)


def test_procedural_art_is_square_and_has_light():
    for art in (build_mandala(256), build_repulsor(256)):
        assert art.shape == (256, 256, 3) and art.dtype == np.uint8
        assert art.max() > 200
        assert art[0, 0].max() == 0           # corners stay dark


def test_composite_is_additive_and_never_darkens():
    frame = np.full((120, 160, 3), 80, np.uint8)
    assert composite(frame, np.zeros_like(frame)) is frame      # nothing to add
    light = np.zeros_like(frame)
    cv2.circle(light, (80, 60), 15, (255, 200, 100), -1)
    out = composite(frame, light, 1.0)
    assert (out >= frame).all()
    assert out[60, 80, 0] > 200
    assert out[0, 0].tolist() >= frame[0, 0].tolist()


def test_bloom_spreads_light_beyond_the_source():
    light = np.zeros((160, 240, 3), np.uint8)
    cv2.circle(light, (120, 80), 6, (255, 255, 255), -1)
    glow = bloom(light)
    assert glow.shape == light.shape
    assert glow[80, 150].max() > 0 and light[80, 150].max() == 0
