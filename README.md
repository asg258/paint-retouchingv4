# Wall Recoloring Pipeline

A modular Python pipeline for intelligently recoloring walls in interior photos.
Built with OpenCV, NumPy, PyTorch, and torchvision.

---

## Project Status

| Stage | Status | File |
|-------|--------|------|
| 1. Image loading + LAB preprocessing | ✅ Done | `preprocess.py` |
| 2. DeepLabV3 semantic segmentation | ✅ Done | `segment.py` |
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

## Stage 2 — Semantic Segmentation (`segment.py`)

Takes the preprocessed RGB image from Stage 1 and produces a **dense probability
mask** — a float array of the same spatial size where each value is
`P(wall | pixel) ∈ [0, 1]`.

### How it works

```
Preprocessed RGB (H, W, 3) uint8
            │
            ▼
    Resize shorter edge to 512 px (configurable)
    Normalise with ImageNet mean/std
    Add batch dimension → (1, 3, H', W') tensor
            │
            ▼
    DeepLabV3+ ResNet101 forward pass
    (ResNet101 extracts multi-scale features via dilated convolutions;
     the ASPP head aggregates them into per-pixel class scores)
            │
            ▼
    Raw logits  (1, num_classes, H', W')
    — one unbounded score per class per pixel
            │
            ▼
    Softmax over class dimension
    P(k | x) = exp(z_k) / Σ_j exp(z_j)
    — converts scores to a probability distribution at each pixel
            │
            ▼
    Extract wall-proxy channel → (H', W')
            │
            ▼
    Bilinear upsample → original (H, W)
            │
            ▼
    float32 NumPy array, values ∈ [0, 1]
```

### The wall class problem

The default torchvision DeepLabV3 was trained on **COCO with Pascal VOC labels
(21 classes)** — there is no "wall" class. We use **class index 0 (background)**
as a proxy.  In interior photos the background class captures walls, ceilings,
and floors — the regions the model cannot assign to any named foreground object.

| Model | Has explicit wall class? | Wall index | Notes |
|---|---|---|---|
| DeepLabV3 (COCO VOC-21, **this file**) | No | 0 (background proxy) | Good enough for first-pass masks |
| DeepLabV3 (ADE20K) | Yes | 1 | Drop-in swap, just change `WALL_CLASS_INDEX` |
| DeepLabV3 (COCO-Stuff 182) | Yes | varies | Best coverage, larger model |

### Why a soft mask, not a binary mask?

A hard 0/1 mask throws away the model's confidence.  Pixels near the boundary
between a wall and a sofa might score P(wall)=0.55 — forcing that to 1 discards
useful uncertainty information.  Downstream blending stages can use the
continuous probability to produce smooth, natural transitions.

### Tunable parameters

| Parameter | Default | What it controls |
|---|---|---|
| `WALL_CLASS_INDEX` | `0` | Which class to extract after softmax |
| `INFERENCE_SIZE` | `512` | Shorter-edge resize before inference. `None` = full resolution. |

### Public API

```python
from segment import load_model, get_deeplab_mask

model, device = load_model()   # load once, reuse for every image

mask = get_deeplab_mask(preprocessed_rgb, model=model, device=device)
# returns: np.ndarray, shape (H, W), dtype float32, values in [0, 1]
```

### Debug visualisation

```bash
python segment.py room.jpg
# opens a 3-panel figure: original | heatmap | overlay
# also saves <image_stem>_wall_mask.png
```

---

## Setup

```bash
pip install opencv-python numpy torch torchvision matplotlib pillow
```

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
