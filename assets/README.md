# Optional artwork

AR Magic Hands draws everything with OpenCV shapes by default, so this folder can stay empty.
Drop in your own PNGs to replace the built-in artwork. No code changes needed.

| File | Used by | What it replaces |
|---|---|---|
| `mandala.png` | Doctor Strange sling ring (key 2) | the procedural magic circle |
| `repulsor.png` | Iron Man repulsor (key 1) | the drawn rings and triangles (the glow is still drawn) |

## Tips

- Use a **PNG with a transparent background** (RGBA). Transparent pixels add no light; bright, saturated colours glow best.
- Make it **square** (for example 512x512 or 1024x1024) with the design centred, because it rotates around its centre.
- Images without transparency also work: the brighter the pixel, the more light it adds. Black adds nothing.
- Start from the built-in artwork: `python scripts/generate_assets.py` writes `mandala.png` and `repulsor.png` here so you can edit them.
- Delete a file to go back to the built-in version.
