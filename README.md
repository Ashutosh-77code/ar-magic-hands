# AR Magic Hands ✋✨

So I wanted to build something that felt like actual magic using just my webcam, Python, and OpenCV. No Unity, no game engine, and definitely no expensive GPU required. 

Just run this, hold your hand up to the camera, and you can shoot Iron Man repulsors, open Doctor Strange portals, set your hands on fire, or literally throw a Kamehameha. 

Built entirely with Python, OpenCV, MediaPipe, and NumPy.

## 🪄 What it actually does (The Effects)

There are 7 different modes you can cycle through:

*   **`1` Iron Man Repulsor:** Open your palm. A glowing blue repulsor sits in your hand. Hold it there and it charges up with energy rushing inward, ending in a shockwave.
*   **`2` Doctor Strange Sling Ring:** Pinch your thumb and index finger together. Golden mandalas open up and spin. Tilt your hand to reverse the spin direction.
*   **`3` Elemental Hands (Fire/Lightning):** Hit `3` to set your skeleton on fire (flames dynamically follow your fingers). Hit `3` again for lightning bolts that arc between your knuckles.
*   **`4` Neon Painter:** Curl your fingers and just point. Draw glowing neon lines in the air. Open your palm for 1 second to clear the canvas.
*   **`5` Rainbow Trails:** Every fingertip leaves a fading rainbow ribbon behind it when you move.
*   **`6` Energy Ball:** Bring both palms close together. A glowing ball forms. Pull your hands apart to grow it, and snap them apart to fire it.
*   **`7` Galaxy:** Open your palm to spawn a spinning galaxy. Close your fist and it collapses into a black hole. Open it again for a supernova.
*   **Peace Sign ✌️ (Photo Booth):** Hold a peace sign for 1 second. It does a 3-2-1 countdown, flashes, and saves a clean photo (no HUD) to your drive. 

It tracks both hands simultaneously and scales the effects based on how close you are to the camera.

## 🚀 How to run it

You need **Python 3.9+** (I highly recommend 3.10-3.12 because MediaPipe can be picky). 

1. Clone the repo and install requirements:
   ```bash
   pip install -r requirements.txt

## Acknowledgements

- [MediaPipe](https://developers.google.com/mediapipe) for hand landmark detection
- [OpenCV](https://opencv.org/) and [NumPy](https://numpy.org/)
- Iron Man and Doctor Strange are Marvel properties. This is an unofficial fan project and uses only original, procedurally drawn artwork.
