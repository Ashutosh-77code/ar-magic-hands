# AR Magic Hands ✋✨

Real-time, gesture-driven augmented reality for your webcam. Open your palm for an **Iron Man repulsor**, pinch to open a **Doctor Strange sling ring**, or set your hands on **fire** or crackle with **lightning**.

Built with Python, OpenCV, MediaPipe and NumPy. No game engine, no GPU required.

## Effects

| Key | Effect | Trigger | What you get |
|---|---|---|---|
| `1` | **Iron Man Repulsor** | Open palm | A pulsing blue-white repulsor at your palm centre. Its size follows your distance from the camera. Hold the pose and it **charges**: brighter glow, energy rushing inward, a charge meter, then shockwave pulses. |
| `2` | **Doctor Strange Sling Ring** | Pinch (thumb + index) | Two counter-rotating mandalas open at the pinch point and throw off orange sparks. **Tilt your hand** to change the spin speed or reverse it. |
| `3` | **Elemental Hands** | Always on | **Fire:** flames hug the hand skeleton and rise, with flame particles from every fingertip. **Lightning:** jagged bolts run along every bone, with random branches and arcs between fingertips. Press `3` again to switch between them. |

Both hands are tracked at the same time, and each one animates independently.

## Quick start

You need Python 3.9 or newer (3.10 to 3.12 are the safest choices for MediaPipe) and a webcam.

On the first run the hand tracking model (a few MB) is downloaded automatically into `models/`. After that it works offline.

### Controls

| Key | Action |
|---|---|
| `1` `2` `3` | Choose an effect (press `3` again for Fire / Lightning) |
| `H` | Show or hide the help overlay |
| `D` | Show the tracked hand skeleton (debug view) |
| `S` | Save a screenshot to `screenshots/` (without the HUD) |
| `Q` / `Esc` | Quit |

### Command line options

```bash
python main.py --mode 3                 # start with the elemental effect
python main.py --camera 1               # use a different camera
python main.py --width 640 --height 480 # lower resolution = higher FPS
python main.py --record docs/demo.mp4   # record the output (no HUD)
python main.py --no-mirror              # do not flip the image
python main.py --help                   # everything else
```

## How it works

```
 Webcam frame ──► HandTracker ──► [Hand, Hand] ──► Effect.update/draw ──► light layer ──► compositor ──► window
 (full size)      (MediaPipe,      landmarks,        (repulsor, sling      (black image,    (add + bloom)
                   downscaled)     gestures)         ring, elemental)      glows only)
```

1. **Tracking** (`hand_tracker.py`): each frame is shrunk to 640 px wide and sent to MediaPipe's HandLandmarker. The 21 landmarks per hand come back normalised and are scaled to full-resolution pixels. A light, speed-aware smoothing filter removes jitter without adding lag to fast swipes.
2. **Gestures** (`Hand` class), all computed from distances, so they work at any rotation and any distance from the camera:
   - *Scale* = distance from the wrist to the middle knuckle. Everything is sized relative to it, which is why effects shrink and grow as you move your hand.
   - *Palm centre* = midpoint of the wrist and the middle knuckle.
   - *Open hand* = at least 4 of 5 digits extended (a fingertip farther from the wrist than its middle joint).
   - *Pinch* = thumb-to-index distance divided by scale, with **hysteresis** (on below 0.30, off above 0.50) so it never flickers. A fist is never counted as a pinch.
   - *Hand angle* = direction from the wrist to the middle knuckle.
3. **Effects** (`effects/`): every effect implements `update(hands, dt, size)` and `draw(light)`. They paint **only light** onto a black layer.
4. **Additive blending** (`compositor.py`): the light layer is added to the video with `cv2.add`, which saturates instead of wrapping, so overlapping lights get brighter like real light instead of looking like stickers. A cheap bloom (a blurred, downscaled copy of the light, blended with `cv2.addWeighted`) adds the soft halo.
5. **Particles** (`particles.py`): a NumPy particle pool handles spawning, motion (velocity, acceleration, drag, turbulence), fading through a colour palette and removal.

### Techniques worth a look

- **Fire** (`effects/elemental.py`): the hand skeleton is drawn into a mask and blurred heavily. The mask feeds a small "heat" buffer that is warped upward with a sideways sway, cooled each frame and flickered with noise. A colour ramp turns heat into flame colours. The simulation runs at a low resolution and is scaled up, which keeps it fast.
- **Lightning**: each bone becomes a polyline of evenly spaced points pushed sideways at random (`graphics.zigzag`). Every point is re-randomised every frame, and the glow comes from the compositor's `addWeighted` bloom.
- **Sling ring**: a PNG with an alpha channel is converted to a premultiplied "light" sprite, then rotated and scaled in a single `cv2.warpAffine` call.
- **Performance**: detection runs on a downscaled frame, bloom works on 1/4 and 1/8 size copies, and the fire runs on a 360 px wide grid. All effects are timed with a delta-time clock, so animation speed does not depend on FPS.

## Using your own artwork

Drop PNGs into `assets/` to replace the built-in drawings (see [`assets/README.md`](assets/README.md)):

- `assets/mandala.png` replaces the sling ring's magic circle (RGBA, square, design centred).
- `assets/repulsor.png` replaces the repulsor's rings.

If a file is missing, the effect falls back to shapes drawn with OpenCV. Run `python scripts/generate_assets.py` to export the built-in artwork as a starting point.

## Customising

Colours, speeds and thresholds are all in [`magic_hands/config.py`](magic_hands/config.py): charge time, ring spin speed, pinch sensitivity, fire height, lightning density, smoothing and more.

To add an effect, subclass `Effect` in `magic_hands/effects/`, implement `update` and `draw`, and add it to `create_effects()`.

## Project structure

```
ar-magic-hands/
├── main.py                  # entry point
├── requirements.txt
├── magic_hands/
│   ├── app.py               # webcam loop, keyboard controls, CLI
│   ├── hand_tracker.py      # MediaPipe wrapper + Hand geometry/gestures
│   ├── particles.py         # particle engine
│   ├── graphics.py          # glows, sprites, lightning paths, palettes
│   ├── compositor.py        # additive blending + bloom
│   ├── procedural.py        # built-in mandala / repulsor artwork
│   ├── ui.py                # HUD, help overlay, debug skeleton
│   ├── config.py            # all tunable constants
│   ├── landmarks.py         # MediaPipe landmark indices
│   └── effects/
│       ├── base.py          # Effect interface
│       ├── repulsor.py      # Iron Man
│       ├── sling_ring.py    # Doctor Strange
│       └── elemental.py     # Fire + Lightning
├── tests/                   # unit tests (synthetic hands, no camera needed)
├── scripts/generate_assets.py
└── assets/                  # optional PNGs
```

## Tests

The tests use synthetic hands, so they need no camera and no MediaPipe.


They cover gesture detection at different rotations and sizes, the particle engine, drawing helpers, tracker smoothing and slot assignment, every effect, and the app's per-frame pipeline. GitHub Actions runs them on every push.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Could not open camera` | Try `--camera 1`, close other apps using the webcam, or try `--backend dshow` (Windows) / `--backend avfoundation` (macOS). On macOS, allow camera access for your terminal in System Settings. |
| Model download fails | Download [`hand_landmarker.task`](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task) manually and save it as `models/hand_landmarker.task` (or pass `--model path`). |
| `pip install mediapipe` fails | MediaPipe publishes wheels only for certain Python versions. Use a 3.10 to 3.12 virtual environment. |
| Low FPS | Use `--width 640 --height 480`, lower `--process-width`, or close other heavy programs. Good, even lighting also helps tracking. |
| Effects appear on the wrong hand side | You probably used `--no-mirror`. The default mirrored view acts like a mirror. |
| Pinch is hard to trigger | Keep at least one of your middle, ring or pinky fingers extended, and touch thumb tip to index tip. Tweak `PINCH_ON` in `config.py`. |

## Roadmap ideas

- Gesture-triggered one-shot effects (a repulsor blast when you close your fist)
- Face and body effects with MediaPipe's other landmarkers
- Sound effects tied to charge level
- GPU acceleration with OpenCV's CUDA modules

## License

[MIT](LICENSE). 

## Acknowledgements

- [MediaPipe](https://developers.google.com/mediapipe) for hand landmark detection
- [OpenCV](https://opencv.org/) and [NumPy](https://numpy.org/)
- Iron Man and Doctor Strange are Marvel properties. This is an unofficial fan project and uses only original, procedurally drawn artwork.
