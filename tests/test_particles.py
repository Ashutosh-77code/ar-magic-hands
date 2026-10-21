"""The particle engine: spawning, motion, fading and removal."""

import math

import numpy as np

from magic_hands.graphics import make_palette
from magic_hands.particles import ParticleSystem

PALETTE = make_palette([(0.0, (255, 255, 255)), (1.0, (0, 0, 255))])


def test_emit_creates_count_per_origin():
    ps = ParticleSystem(PALETTE, capacity=100, seed=1)
    assert ps.emit((50, 50), 7) == 7
    assert ps.emit([(0, 0), (10, 10), (20, 20)], 4) == 12
    assert len(ps) == 19


def test_capacity_is_never_exceeded():
    ps = ParticleSystem(PALETTE, capacity=10, seed=1)
    assert ps.emit((0, 0), 50) == 10
    assert ps.emit((0, 0), 5) == 0
    assert len(ps) == 10


def test_particles_die_after_their_life():
    ps = ParticleSystem(PALETTE, capacity=100, seed=1)
    ps.emit((0, 0), 20, life=(0.5, 0.5))
    ps.update(0.3)
    assert len(ps) == 20
    ps.update(0.3)
    assert len(ps) == 0


def test_update_moves_particles_along_velocity():
    ps = ParticleSystem(PALETTE, capacity=10, seed=1)
    ps.emit((100, 100), 1, speed=(100, 100), direction=0.0, spread=0.0, life=(2, 2))
    ps.update(0.5)
    assert abs(ps.pos[0, 0] - 150.0) < 1e-3 and abs(ps.pos[0, 1] - 100.0) < 1e-3


def test_direction_up_moves_toward_smaller_y():
    ps = ParticleSystem(PALETTE, capacity=50, seed=1)
    ps.emit((100, 100), 30, speed=(80, 120), direction=-math.pi / 2, spread=0.4, life=(1, 1))
    ps.update(0.2)
    assert np.all(ps.pos[:30, 1] < 100.0)


def test_ring_spawn_puts_particles_on_the_circle_moving_outward():
    ps = ParticleSystem(PALETTE, capacity=100, seed=1)
    ps.emit((200, 200), 50, speed=(60, 60), ring_radius=80.0, life=(1, 1))
    start = np.linalg.norm(ps.pos[:50] - np.array([200, 200]), axis=1)
    assert np.allclose(start, 80.0, atol=0.01)
    ps.update(0.1)
    after = np.linalg.norm(ps.pos[:50] - np.array([200, 200]), axis=1)
    assert np.all(after > start)


def test_negative_speed_moves_particles_inward():
    ps = ParticleSystem(PALETTE, capacity=100, seed=1)
    ps.emit((200, 200), 40, speed=(-100, -100), ring_radius=80.0, life=(1, 1))
    start = np.linalg.norm(ps.pos[:40] - np.array([200, 200]), axis=1)
    ps.update(0.1)
    after = np.linalg.norm(ps.pos[:40] - np.array([200, 200]), axis=1)
    assert np.all(after < start)


def test_acceleration_and_drag():
    ps = ParticleSystem(PALETTE, capacity=10, accel=(0, -100), seed=1)
    ps.emit((0, 0), 1, speed=(0, 0), life=(5, 5))
    ps.update(1.0)
    assert ps.vel[0, 1] < -99.0
    draggy = ParticleSystem(PALETTE, capacity=10, drag=1.0, seed=1)
    draggy.emit((0, 0), 1, speed=(100, 100), direction=0.0, spread=0.0, life=(5, 5))
    draggy.update(0.5)
    assert abs(draggy.vel[0, 0] - 50.0) < 1e-3


def test_draw_paints_pixels_and_handles_offscreen_particles():
    ps = ParticleSystem(PALETTE, capacity=100, seed=1)
    ps.emit((50, 50), 10, size=(3, 3), life=(1, 1))
    ps.emit((-500, -500), 5, life=(1, 1))        # far off-screen
    light = np.zeros((100, 100, 3), np.uint8)
    ps.draw(light)
    assert light.any()


def test_clear_and_empty_draw():
    ps = ParticleSystem(PALETTE, capacity=10, seed=1)
    ps.emit((0, 0), 5)
    ps.clear()
    assert len(ps) == 0
    ps.update(0.1)
    ps.draw(np.zeros((10, 10, 3), np.uint8))   # must not raise
