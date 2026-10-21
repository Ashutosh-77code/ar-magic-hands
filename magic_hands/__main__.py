"""Allows ``python -m magic_hands`` as an alternative to ``python main.py``."""

from magic_hands.app import main

if __name__ == "__main__":
    raise SystemExit(main())
