"""Neon Painter, Rainbow Trails, Energy Ball, Galaxy and the peace-sign photo booth."""

import cv2
import numpy as np

from magic_hands.app import MagicHandsApp, Settings
from magic_hands.effects import EnergyBall, GalaxyHands, NeonPainter, RainbowTrails, create_effects
from magic_hands.gestures import is_peace, is_pointing
from magic_hands.hand_tracker import HandTracker
from magic_hands.snapshot import COUNTDOWN_SECONDS, HOLD_SECONDS, GestureSnapshot
from tests.synthetic import make_hand
from tests.test_tracker import FakeDetector, normalised

SIZE = (1280, 720)
DT = 1.0 / 30.0


def light_buffer():
    return np.zeros((SIZE[1], SIZE[0], 3), np.uint8)


def step(effect, hands, light=None):
    light = light if light is not None else light_buffer()
    light.fill(0)
    effect.update(hands, DT, SIZE)
    effect.draw(light)
    return light


def energy(light, box=None):
    if box:
        x0, y0, x1, y1 = box
        light = light[max(0, y0):y1, max(0, x0):x1]
    return int(light.sum())


# ------------------------------------------------------------------ gestures
def test_pointing_and_peace_gestures():
    assert is_pointing(make_hand("point")) and not is_pointing(make_hand("peace"))
    assert is_peace(make_hand("peace")) and not is_peace(make_hand("point"))
    for pose in ("open", "fist", "pinch"):
        assert not is_pointing(make_hand(pose)) and not is_peace(make_hand(pose)), pose


def test_new_gestures_work_at_any_rotation():
    for angle in (-90, -30, 0, 45, 120):
        assert is_pointing(make_hand("point", angle_deg=angle)), angle
        assert is_peace(make_hand("peace", angle_deg=angle)), angle


# ------------------------------------------------------------------ Neon Painter
def paint_line(effect, frames=25, pose="point"):
    light = light_buffer()
    for i in range(frames):
        hand = make_hand(pose, center=(300 + 25 * i, 600), scale=150)
        light = step(effect, [hand], light)
    return light


def test_neon_painter_draws_while_pointing_and_keeps_the_line():
    effect = NeonPainter(seed=1)
    paint_line(effect)
    assert effect.strokes_drawn > 10
    after = step(effect, [])                                  # hand leaves: the drawing stays
    assert after.any()
    ys, xs = np.nonzero(after.any(axis=2))
    assert xs.max() - xs.min() > 300                          # a long horizontal stroke


def test_neon_painter_ignores_open_hands_fists_and_flicks():
    for pose in ("open", "fist", "peace"):
        effect = NeonPainter(seed=1)
        paint_line(effect, pose=pose)
        assert effect.strokes_drawn == 0, pose
    effect = NeonPainter(seed=1)
    for i in range(2):                                        # shorter than START_FRAMES
        step(effect, [make_hand("point", center=(300 + 30 * i, 600), scale=150)])
    assert effect.strokes_drawn == 0


def test_neon_painter_stroke_colour_changes_along_the_line():
    effect = NeonPainter(seed=1)
    painted = paint_line(effect, frames=40)
    hsv = cv2.cvtColor(painted, cv2.COLOR_BGR2HSV)
    bright = hsv[(hsv[:, :, 2] > 150) & (hsv[:, :, 1] > 120)]
    assert len(set((bright[:, 0] // 15).tolist())) >= 3      # several different hues


def test_neon_painter_clears_with_an_open_palm_or_when_the_key_is_pressed_again():
    effect = NeonPainter(seed=1)
    paint_line(effect)
    palm = make_hand("open", center=(640, 600), scale=150)
    for _ in range(int(1.2 / DT)):
        step(effect, [palm])
    for _ in range(int(0.6 / DT)):                           # let the dissolve finish
        gone = step(effect, [palm])
    assert energy(gone, (0, 0, 1280, 300)) == 0 and not effect._glow.any() and not effect._core.any()

    effect = NeonPainter(seed=1)
    paint_line(effect)
    effect.on_reselect()
    assert not effect._glow.any() and not effect._core.any()


def test_neon_painter_lifting_the_finger_starts_a_new_stroke_without_joining_lines():
    effect = NeonPainter(seed=1)
    for i in range(12):
        step(effect, [make_hand("point", center=(200 + 15 * i, 300), scale=150)])
    for _ in range(3):
        step(effect, [make_hand("fist", center=(380, 300), scale=150)])
    before = effect.strokes_drawn
    for i in range(12):
        step(effect, [make_hand("point", center=(900 + 15 * i, 600), scale=150)])
    canvas = effect._core
    assert effect.strokes_drawn > before
    assert canvas[450, 600:800].max() == 0                    # nothing was drawn between the two strokes


# ------------------------------------------------------------------ Rainbow Trails
def test_trails_follow_fingertips_in_many_colours_and_fade_away():
    effect = RainbowTrails(seed=1)
    light = light_buffer()
    for i in range(30):
        light = step(effect, [make_hand("open", center=(200 + 30 * i, 500), scale=140)], light)
    hsv = cv2.cvtColor(light, cv2.COLOR_BGR2HSV)
    vivid = hsv[(hsv[:, :, 2] > 120) & (hsv[:, :, 1] > 100)]
    assert len(set((vivid[:, 0] // 12).tolist())) >= 5        # rainbow, not one colour
    for _ in range(int(1.6 / DT)):
        light = step(effect, [], light)
    assert not light.any()
    assert not effect._trails                                  # memory is released too


def test_trails_draw_one_ribbon_per_fingertip():
    effect = RainbowTrails(seed=1)
    for i in range(10):
        step(effect, [make_hand("open", center=(300 + 20 * i, 500), scale=140)])
    assert len(effect._trails) == 5
    for i in range(10):
        step(effect, [make_hand("open", (300, 500), 140, slot=0), make_hand("open", (900, 500), 140, slot=1)])
    assert len(effect._trails) == 10


# ------------------------------------------------------------------ Energy Ball
def two_hands(distance, scale=140.0, y=500):
    return [make_hand("open", (640 - distance / 2, y), scale, slot=0),
            make_hand("open", (640 + distance / 2, y), scale, slot=1)]


def test_energy_ball_needs_two_hands_close_together():
    assert energy(step(EnergyBall(seed=1), [make_hand("open")])) == 0
    effect = EnergyBall(seed=1)
    for _ in range(15):
        light = step(effect, two_hands(500))
    assert energy(light) > 0
    far = EnergyBall(seed=1)
    for _ in range(15):
        light = step(far, two_hands(1200, scale=100))         # more than 6.5 hand-scales apart
    assert energy(light) == 0


def test_energy_ball_forms_between_the_palms_and_grows_with_distance_and_charge():
    hands = two_hands(450)
    mid = (640, int(hands[0].palm_center[1]))
    effect = EnergyBall(seed=1)
    for _ in range(12):
        light = step(effect, hands)
    assert energy(light, (mid[0] - 220, mid[1] - 220, mid[0] + 220, mid[1] + 220)) > 0.7 * energy(light)
    early = effect._radius
    for _ in range(60):
        step(effect, hands)
    assert effect._radius > early                              # charging makes it bigger
    near = EnergyBall(seed=1)
    wide = EnergyBall(seed=1)
    for _ in range(60):
        step(near, two_hands(300)); step(wide, two_hands(700))
    assert wide._radius > near._radius


def test_energy_ball_blasts_when_hands_snap_apart_after_charging_but_not_when_moved_slowly():
    effect = EnergyBall(seed=1)
    for _ in range(60):
        step(effect, two_hands(300))
    for i in range(1, 30):                                     # slow, gentle separation
        step(effect, two_hands(300 + 4 * i))
    assert effect.blast_count == 0
    for i in range(6):                                         # fast snap
        light = step(effect, two_hands(420 + 90 * i))
    assert effect.blast_count == 1
    assert effect._cooldown > 0 and effect._charge < 0.5       # the stored energy was released

    uncharged = EnergyBall(seed=1)
    for _ in range(3):
        step(uncharged, two_hands(300))
    for i in range(6):
        step(uncharged, two_hands(300 + 100 * i))
    assert uncharged.blast_count == 0                          # needs charge first


def test_energy_ball_hint_changes_with_the_number_of_hands():
    effect = EnergyBall(seed=1)
    step(effect, [make_hand("open")])
    one = effect.hint
    step(effect, two_hands(400))
    assert one != effect.hint


# ------------------------------------------------------------------ Galaxy
def test_galaxy_appears_on_an_open_palm_and_not_on_a_pointing_hand():
    hand = make_hand("open", center=(640, 560), scale=150)
    effect = GalaxyHands(seed=1)
    for _ in range(15):
        light = step(effect, [hand])
    cx, cy = map(int, hand.palm_center)
    near = energy(light, (cx - 260, cy - 260, cx + 260, cy + 260))
    assert near > 0.8 * energy(light) > 0
    gone = GalaxyHands(seed=1)
    for _ in range(15):
        light = step(gone, [make_hand("point")])
    assert energy(light) == 0


def test_galaxy_collapses_into_a_black_hole_then_goes_supernova_when_reopened():
    effect = GalaxyHands(seed=1)
    open_hand, fist = make_hand("open", scale=150), make_hand("fist", scale=150)
    for _ in range(15):
        step(effect, [open_hand])
    wide = effect._states[0].radius
    for _ in range(20):
        step(effect, [fist])
    state = effect._states[0]
    assert state.collapse > 0.9 and state.radius < 0.5 * wide and state.armed
    assert effect.supernova_count == 0
    step(effect, [open_hand])
    assert effect.supernova_count == 1 and not state.armed
    assert len(effect._sparks) > 50
    for _ in range(3):
        step(effect, [open_hand])
    assert effect.supernova_count == 1                         # only one explosion


def test_galaxy_does_not_explode_without_a_collapse_first():
    effect = GalaxyHands(seed=1)
    for pose in ("open", "other_gap", "open", "fist", "fist"):
        hand = make_hand("pinch" if pose == "other_gap" else pose, scale=150)
        step(effect, [hand])
    assert effect.supernova_count == 0


def test_galaxy_spins_over_time_and_spins_faster_when_collapsed():
    effect = GalaxyHands(seed=1)
    a = step(effect, [make_hand("open")]).copy()
    for _ in range(5):
        b = step(effect, [make_hand("open")])
    assert not np.array_equal(a, b)
    slow = effect._states[0].rot
    for _ in range(20):
        step(effect, [make_hand("fist")])
    before = effect._states[0].rot
    step(effect, [make_hand("fist")])
    assert effect._states[0].rot - before > (slow / 6) * 1.5   # per-frame spin is faster now


# ------------------------------------------------------------------ Photo booth
def test_peace_sign_starts_a_countdown_and_takes_one_photo():
    booth = GestureSnapshot()
    peace = [make_hand("peace")]
    for _ in range(int((HOLD_SECONDS + 0.2) / DT)):
        assert booth.update(peace, DT) is False
    assert booth.countdown is not None
    shots = 0
    for _ in range(int((COUNTDOWN_SECONDS + 0.5) / DT)):      # hand lowered during the countdown
        shots += booth.update([], DT)
    assert shots == 1 and booth.photos_taken == 1


def test_photo_booth_ignores_other_gestures_and_short_holds_and_has_a_cooldown():
    booth = GestureSnapshot()
    for pose in ("open", "fist", "point", "pinch"):
        for _ in range(60):
            booth.update([make_hand(pose)], DT)
    assert booth.countdown is None
    for _ in range(int(0.6 / DT)):
        booth.update([make_hand("peace")], DT)
    booth.update([], 1.0)                                      # hold decays when the sign is dropped
    assert booth.hold < 0.1 and booth.countdown is None

    for _ in range(int((HOLD_SECONDS + 0.2) / DT)):
        booth.update([make_hand("peace")], DT)
    for _ in range(int((COUNTDOWN_SECONDS + 0.2) / DT)):
        booth.update([make_hand("peace")], DT)
    assert booth.photos_taken == 1
    for _ in range(int(1.0 / DT)):                             # still holding the sign right after the photo
        booth.update([make_hand("peace")], DT)
    assert booth.countdown is None                             # cooldown: no instant second photo


def test_photo_booth_overlay_draws_at_every_stage():
    booth = GestureSnapshot()
    img = np.full((720, 1280, 3), 80, np.uint8)
    for _ in range(int(0.5 / DT)):
        booth.update([make_hand("peace")], DT)
    booth.draw(img)                                            # progress bar
    for _ in range(int(0.7 / DT)):
        booth.update([make_hand("peace")], DT)
    booth.draw(img)                                            # countdown number
    for _ in range(int(3.2 / DT)):
        booth.update([], DT)
    flash = img.copy()
    booth.draw(flash)                                          # flash
    assert flash.mean() >= img.mean()


def test_app_requests_a_screenshot_when_the_countdown_ends():
    det = FakeDetector()
    app = MagicHandsApp(Settings(), tracker=HandTracker(smoothing=0.0, detector=det))
    frame = np.full((720, 1280, 3), 40, np.uint8)
    det.queue = [normalised(make_hand("peace", (640, 600), 150), 1280, 720)]
    for _ in range(int(1.3 / DT)):
        app.process_frame(frame.copy(), DT)
    assert not app._screenshot_requested and app.snapshot.countdown is not None
    det.queue = []
    for _ in range(int(3.3 / DT)):
        app.process_frame(frame.copy(), DT)
    assert app._screenshot_requested
    off = MagicHandsApp(Settings(snapshot=False), tracker=HandTracker(detector=FakeDetector()))
    assert off.snapshot is None


def test_new_modes_are_reachable_with_keys_4_to_7_and_have_labels():
    det = FakeDetector()
    app = MagicHandsApp(Settings(), tracker=HandTracker(detector=det))
    names = []
    for key in "4567":
        app.handle_key(ord(key))
        names.append(app.effect.name)
    assert names == ["Neon Painter", "Rainbow Trails", "Energy Ball", "Galaxy"]
    from magic_hands.app import MODE_LABELS
    assert len(MODE_LABELS) == len(app.effects) == 7


def test_new_effects_survive_every_pose_extreme_positions_and_resizes():
    poses = ("open", "fist", "pinch", "point", "peace")
    for effect in create_effects(None, seed=3)[3:]:
        for pose in poses:
            for _ in range(5):
                effect.update([make_hand(pose)], DT, SIZE)
                effect.draw(light_buffer())
        effect.update([make_hand("open", (-300, -300), 80)], DT, SIZE)
        effect.draw(light_buffer())
        effect.update([make_hand("open", (1500, 900), 700)], DT, SIZE)
        effect.draw(light_buffer())
        effect.update(two_hands(300), DT, SIZE)
        effect.draw(light_buffer())
        small = np.zeros((480, 640, 3), np.uint8)
        effect.update([make_hand("point", (300, 380), 90)], DT, (640, 480))
        effect.draw(small)
        effect.reset()
        effect.on_reselect()
