"""Export the built-in artwork as editable RGBA PNG files.

    python scripts/generate_assets.py            # writes into ./assets
    python scripts/generate_assets.py --out my_art --size 1024

The exported files are picked up automatically when they sit in ``assets/``.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from magic_hands.procedural import build_mandala, build_repulsor  # noqa: E402


def light_to_rgba(light_bgr: np.ndarray) -> np.ndarray:
    """Turn a black-background "light" image into a PNG whose transparency carries the brightness."""
    alpha = light_bgr.max(axis=2)
    colour = np.zeros_like(light_bgr, np.float32)
    mask = alpha > 0
    colour[mask] = light_bgr[mask].astype(np.float32) * 255.0 / alpha[mask, None]
    return np.dstack([np.clip(colour, 0, 255).astype(np.uint8), alpha])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "assets")
    parser.add_argument("--size", type=int, default=512, help="image size in pixels (square)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for name, art in (("mandala.png", build_mandala(args.size)), ("repulsor.png", build_repulsor(args.size))):
        path = args.out / name
        cv2.imwrite(str(path), light_to_rgba(art))
        print("Wrote", path)


if __name__ == "__main__":
    main()
