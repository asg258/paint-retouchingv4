# Wall Recoloring Pipeline

A modular Python pipeline for intelligently recoloring walls in interior photos.
Built with OpenCV and NumPy, with PyTorch coming in for the segmentation stage.

---

## Project Status

| Stage | Status | File |
|-------|--------|------|
| 1. Image loading + LAB preprocessing | ✅ Done | `preprocess.py` |
| 2. Wall segmentation | 🔜 Planned | — |
| 3. Color transfer / recoloring | 🔜 Planned | — |
| 4. Post-processing + output | 🔜 Planned | — |

---

## Stage 1 — Preprocessing (`preprocess.py`)

Before any segmentation or recoloring can happen, the image needs to be in good
shape: decent contrast and enough color signal that downstream models can
distinguish wall from furniture.  This stage handles both.

### What it does

```
RGB image on disk
       │
       ▼
  Load + RGB decode
       │
       ▼
  Convert to LAB color space
  (separates brightness from color — easier to edit each independently)
       │
       ├── L channel → CLAHE contrast enhancement
       │   (boosts local contrast without blowing out highlights)
       │
       └── A + B channels → compute average saturation
           │
           ├── saturation OK?  → keep A/B as-is
           │
           └── saturation low? → scale A/B around neutral midpoint
               (makes washed-out images more colorful, leaves grays alone)
       │
       ▼
  Convert back to RGB
       │
       ▼
  Return processed uint8 RGB array
```

### Tunable parameters

All parameters live at the top of `preprocess.py` so you can adjust them
without touching the logic:

| Parameter | Default | What it controls |
|---|---|---|
| `CLAHE_CLIP_LIMIT` | `2.0` | How aggressively CLAHE boosts local contrast.  Higher = more dramatic. |
| `CLAHE_TILE_GRID_SIZE` | `(8, 8)` | How finely the image is divided for local equalization. |
| `SATURATION_THRESHOLD` | `0.15` | Images below this average saturation get the color boost. |
| `SATURATION_SCALE` | `1.2` | How much A/B channels are scaled when boosting (1.2 = +20%). |

### Public API

```python
from preprocess import preprocess_image

rgb_array = preprocess_image("path/to/room.jpg")
# returns: np.ndarray, shape (H, W, 3), dtype uint8, RGB channel order
```

---

## Setup

```bash
pip install opencv-python numpy
```

PyTorch is not needed yet (required for Stage 2 segmentation).

---

## Quick test

```bash
python preprocess.py path/to/room.jpg
```

Opens a side-by-side **Before / After** window.  Press any key to close.

---

## Design decisions

- **LAB color space** — keeps luminance and chrominance separate.  Editing L
  doesn't shift colors; editing A/B doesn't change brightness.  This makes
  the contrast and saturation steps independent and non-destructive.

- **CLAHE over global histogram equalization** — global equalization can
  over-darken large flat regions (like walls!).  CLAHE works locally, so
  uniform wall areas stay natural while textured areas get crisper.

- **Saturation measured in A–B Euclidean distance** — this matches the
  perceptual definition: how far a pixel is from neutral gray in the color
  plane.  It's fast (just sqrt of two squared terms) and doesn't require
  converting to HSV.

- **Scale around midpoint 128** — cv2 encodes LAB with A/B centered at 128
  (not 0).  Scaling around 128 means pure gray pixels (A=128, B=128) are
  unaffected; only colored pixels move outward.
