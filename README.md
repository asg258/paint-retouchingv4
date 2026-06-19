# Wall Recoloring Pipeline

A modular Python pipeline for intelligently recoloring walls in interior photos.
Built with OpenCV, NumPy, PyTorch, and torchvision.

---

## Project Status

| Stage | Status | File |
|-------|--------|------|
| 1. Image loading + LAB preprocessing | ✅ Done | `preprocess.py` |
| 2. DeepLabV3 semantic segmentation | ✅ Done | `segment.py` |
| 3. SAM boundary refinement | ✅ Done | `refine.py` |
| 3b. Mask processing (smooth + erode) | ✅ Done | `mask_process.py` |
| 5. Object protection mask | ✅ Done | `protect.py` |
| 6. Wall color detection (K-Means) | ✅ Done | `color_detect.py` |
| 7. Color layer generation | ✅ Done | `color_layer.py` |
| 4. LAB wall recoloring | ✅ Done | `recolor.py` |
| 5. Official color database | ✅ Done | `colors.py` + `colors_valspar.csv` |
| 6. Post-processing + output | 🔜 Planned | — |

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

## Stage 3 — SAM Boundary Refinement (`refine.py`)

DeepLabV3 gives us a good first approximation of where the wall is, but its
boundaries are blurry.  This stage runs Meta's **Segment Anything Model (SAM)**
to fix that: crisp edges, accurate corners, fine details recovered.

### Why DeepLab boundaries are blurry

CNNs process images through stacked convolution layers.  Each convolution
averages pixels inside a small window, and by the time you're 50 layers deep,
the network "sees" a large region rather than a single pixel.  This is great
for recognising objects but terrible for drawing precise boundaries — the wall/
sofa edge comes out as a gradient of uncertain pixels several pixels wide.

### How SAM fixes it

SAM uses a **Vision Transformer (ViT)** that encodes the entire image in one
shot, giving it global context.  Its mask decoder was specifically trained to
produce sharp, pixel-accurate boundaries.  Crucially, SAM is **promptable** —
it doesn't try to classify every pixel; instead, it draws a mask around
whatever region you point it at.

### How we drive SAM from the coarse mask

```
Coarse DeepLab mask (H, W) float32
            │
            ▼
  Threshold into three zones:
  - P > 0.70  → confident wall       → foreground prompts (label = 1)
  - P < 0.30  → confident background → background prompts (label = 0)
  - 0.30–0.70 → uncertain            → ignored (let SAM decide)
            │
            ▼
  Sample a sparse grid of prompt coordinates
  from each zone (max 8 fg + 4 bg points)
            │
            ▼
  predictor.set_image(rgb)   ← encodes image once (~0.1–1 s)
  predictor.predict(coords, labels, multimask_output=True)
            │
            ▼
  SAM returns 3 candidate binary masks + confidence scores
            │
            ▼
  Select best candidate by IoU vs coarse mask (tie-break: SAM score)
            │
            ▼
  Combine:  M_refined = SAM_binary × M_coarse
  (SAM sets the boundary, DeepLab supplies soft interior confidence)
            │
            ▼
  float32 (H, W) refined probability mask
```

### Why SAM output is binary — and why we don't keep it that way

SAM's decoder produces a 0/1 mask.  There is no "I'm 73 % sure this is wall"
signal — it just draws a boundary and fills.  That's fine for edges, but it
destroys the interior confidence information that DeepLab carefully computed.
By multiplying `SAM_binary × M_coarse` we get both: SAM's precise edge AND
DeepLab's soft confidence inside the mask.  Downstream blending stages need
those interior gradients to create natural-looking colour transitions.

### Setup (extra step required)

```bash
# 1. Install the package
pip install git+https://github.com/facebookresearch/segment-anything.git

# 2. Download a checkpoint (ViT-B is the best balance of speed and accuracy)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
# or for maximum quality:
# wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Update `SAM_CHECKPOINT` and `SAM_MODEL_TYPE` at the top of `refine.py` if you
use a different checkpoint.

### Tunable parameters

| Parameter | Default | What it controls |
|---|---|---|
| `SAM_CHECKPOINT` | `"sam_vit_b_01ec64.pth"` | Path to the downloaded checkpoint |
| `SAM_MODEL_TYPE` | `"vit_b"` | Must match the checkpoint (`vit_b`, `vit_l`, `vit_h`) |
| `FG_THRESHOLD` | `0.70` | Mask values above this become foreground prompts |
| `BG_THRESHOLD` | `0.30` | Mask values below this become background prompts |
| `MAX_FG_POINTS` | `8` | Maximum number of foreground prompt points |
| `MAX_BG_POINTS` | `4` | Maximum number of background prompt points |
| `POINT_GRID_SPACING` | `30` | Pixel spacing of the sampling grid |

### Public API

```python
from refine import load_sam_model, refine_mask_with_sam

predictor, device = load_sam_model()   # load once

refined_mask = refine_mask_with_sam(
    image=preprocessed_rgb,   # (H, W, 3) uint8 RGB
    coarse_mask=deeplab_mask, # (H, W) float32 from Stage 2
    predictor=predictor,
)
# returns: np.ndarray, shape (H, W), dtype float32, values in [0, 1]
# edges are sharper than the input coarse_mask
```

### Debug visualisation

```bash
python refine.py room.jpg
# runs all 3 stages end-to-end, opens a 4-panel figure:
#   original + prompts | coarse mask | refined mask | difference
# saves <image_stem>_refined_mask.png
```

---

## Stage 3b — Mask Processing (`mask_process.py`)

Sits between SAM refinement and recoloring. Takes the near-binary SAM mask
and turns it into a clean, soft probability map that blends naturally.

### Why the SAM mask still needs work

SAM produces excellent boundaries but the mask has four remaining problems:

| Problem | Cause | Effect if unaddressed |
|---|---|---|
| Hard 0/1 edges | SAM is a boundary detector, not a blender | Visible seam in the recolored image |
| Boundary pixel contamination | Edge pixels are shared between wall and adjacent objects | Color bleeds onto sofa/curtain edge |
| Salt-and-pepper noise | Small isolated regions from DeepLab errors | Random colored pixels in non-wall areas |
| Jagged contours | SAM traces real texture (curtains, upholstery) | Ragged color boundary |

### What the processing does

```
SAM binary mask  (0 or 1, hard edges, noise)
        |
        v
  1. Erosion  (cv2.erode, ellipse kernel, 5 px)
     — shrinks mask inward: contested boundary pixels are excluded
     — prevents color from reaching pixels shared with adjacent objects
        |
        v
  2. Gaussian blur  (cv2.GaussianBlur, sigma=3, kernel=21)
     — turns hard 0/1 step into a smooth 0 -> 1 gradient
     — formula: M_soft[x,y] = sum G(dx,dy,sigma) * M[x+dx,y+dy]
     — recoloring stage uses these gradients for feathered blending
        |
        v
  3. Noise threshold  (mask[mask < 0.05] = 0)
     — removes the faint halo that blur spreads outward
     — keeps only meaningful transition pixels
        |
        v
  Soft float32 mask  (0.0 -> 1.0, smooth edges, no noise)
```

### Why keep a soft mask (not binary)?

The recoloring stage multiplies each pixel's colour shift by the mask value.
A binary mask = hard edge. A soft mask (0.0 → 1.0 gradient) = the colour
fades in gradually at the boundary — exactly how real paint looks at the edge
of a taped area. Zero extra cost at blend time; dramatically better result.

### Tunable parameters

| Parameter | Default | What it controls |
|---|---|---|
| `EROSION_SIZE` | `5` px | How many pixels to shrink the mask inward |
| `GAUSSIAN_SIGMA` | `3.0` | Width of the soft feather zone |
| `GAUSSIAN_KERNEL_SIZE` | `21` | Gaussian window size (must be odd) |
| `NOISE_THRESHOLD` | `0.05` | Values below this are zeroed out after blur |

### Public API

```python
from mask_process import process_mask

clean_mask = process_mask(sam_mask)
# returns: np.ndarray, shape (H, W), dtype float32, values in [0, 1]
```

### Debug visualisation

```bash
python mask_process.py room.jpg
# runs stages 1-3b, saves <stem>_mask_processing.png
# 4-panel figure: raw mask | processed mask | difference | overlay
```

---

## Stage 5 — Object Protection Mask (`protect.py`)

Builds a **safety buffer around every non-wall region** so the recoloring
step can never bleed onto furniture, appliances, or decor — even if the wall
mask is slightly imperfect.

### Why Stages 3–4 are still not enough

Stage 3b (erosion) and Stage 3 (SAM) both reduce bleed from the **wall side**:
they shrink and clean the wall mask. But there are three remaining failure modes:

| Failure | Root cause | Why erosion/SAM can't fix it alone |
|---|---|---|
| Sub-pixel aliasing at boundary | Edge pixels are a blend of wall + object colours | Even a perfect mask claims some object pixels |
| JPEG / compression artifacts | Block artefacts shift apparent boundary by 2–4 px | The mask follows the compressed boundary, not the true one |
| Low-confidence wall creep | DeepLab assigns P(wall)=0.15 to sofa cushion | Doesn't look bad in the mask but still gets a faint tint |

Stage 5 attacks the problem from the **object side**: expand every object's
footprint outward so it over-claims, rather than under-claims, its territory.

### Mathematical formulation

```
Object mask (raw):
    M_obj = 1 - M_wall

Dilated protection zone:
    M_protect = M_obj ⊕ K       (⊕ = morphological dilation, K = ellipse kernel)

Optional soft edge:
    M_protect = GaussianBlur(M_protect, sigma)

Final safe-to-recolor mask:
    M_final = M_wall * (1 - M_protect)

Blending formula (Stage 8):
    Output = M_final * Recolored + (1 - M_final) * Original
```

### How dilation prevents bleeding

Dilation replaces each pixel with the **maximum** value in its neighbourhood.
Applied to the object mask, it expands every object outward by `dilation_size`
pixels. Any wall pixel within that buffer gets a protection value close to 1,
which zeroes it in `M_final` — the recoloring step never sees it.

Think of it as painter's tape: erosion tidies up the brush, but tape
guarantees zero bleed regardless of brush quality.

### Tunable parameters

| Parameter | Default | What it controls |
|---|---|---|
| `DILATION_SIZE` | `10` px | Width of the safety buffer around objects |
| `PROTECTION_SIGMA` | `2.0` | Gaussian sigma to soften the protection edge |
| `PROTECTION_KERNEL_SIZE` | `15` | Gaussian kernel size (must be odd) |

### Public API

```python
from protect import create_object_protection_mask, apply_protection

protection_mask = create_object_protection_mask(clean_mask)
# returns: (H, W) float32, 1 = protected, 0 = safe to recolor

final_mask = apply_protection(clean_mask, protection_mask)
# returns: M_wall * (1 - M_protect), ready for blending
```

### Debug visualisation

```bash
python protect.py room.jpg
# runs stages 1–5, saves <stem>_protection.png
# 5-panel figure: wall mask | object mask | protection | final | overlay
```

---

## Stage 6 — Wall Color Detection (`color_detect.py`)

Extracts the **dominant wall color** from the masked region using K-Means
clustering, independently of lighting, shadows, and compression noise.

### Why not just average the wall pixels?

A simple mean of all wall pixels gives a muddy result because lighting
gradients (brighter near windows, darker in corners) and soft shadows each
shift the average away from the true paint color. K-Means separates these
into distinct clusters so you can pick just the dominant one.

### How K-Means works here

Only pixels where `M_final > 0.7` are used — the most confident wall pixels.

```
Wall pixels S = { I(x,y) | M_final(x,y) > 0.7 }

K-Means objective:
    minimize  Σ_{i=1}^{N} || x_i - μ_{c(i)} ||²

Where:
    x_i      = [R, G, B] color vector of pixel i
    μ_{c(i)} = centroid of the cluster pixel i is assigned to
    || · ||² = squared Euclidean distance in 3D RGB space
    c(i)     = cluster assignment for pixel i

Each term measures how far a pixel is from its cluster center.
Minimizing the total drives each centroid to the mean color of its group.
```

Typical cluster layout in a room photo:

| Cluster | What it captures | Why it's smaller |
|---|---|---|
| **Largest** | True base paint color | Most wall area is evenly lit |
| Medium | Highlight zone (near window) | Only affects bright patches |
| Small | Shadow / corner zone | Affects limited area |

The dominant cluster (largest by pixel count) is the true base color.

### What this feeds into

- **Stage 7 (adaptive recoloring)** — measures the LAB distance between
  the current wall color and the target color to scale the blend adaptively.
  Dark-to-light and light-to-dark transitions need different treatment.
- **Stage 9 (color consistency metric)** — compares the measured wall
  color after recoloring against the target to verify accuracy.

### Tunable parameters

| Parameter | Default | What it controls |
|---|---|---|
| `K_CLUSTERS` | `3` | Number of clusters. 3 = base + highlight + shadow |
| `MASK_THRESHOLD` | `0.7` | Min mask confidence to include a pixel |
| `KMEANS_ATTEMPTS` | `10` | Independent runs — best result kept |
| `KMEANS_MAX_ITER` | `100` | Max iterations per run |

### Public API

```python
from color_detect import extract_wall_color

result = extract_wall_color(image_rgb, final_mask)

result.dominant_color   # (3,) uint8 [R, G, B]
result.dominant_lab     # (3,) float32 [L, A, B] — ready for Stage 7
result.clusters         # (K, 3) uint8 — all centroids, largest first
result.counts           # (K,) int — pixels per cluster
result.fractions        # (K,) float — fraction of wall area
```

### Debug visualisation

```bash
python color_detect.py room.jpg
# runs stages 1–6, saves <stem>_wall_colors.png
# 3-panel figure: wall overlay | cluster swatches | dominant color swatch
```

---

## Stage 7 — Color Layer Generation (`color_layer.py`)

Builds the **target reflectance layer** R(x,y) — a full-size (H×W×3) image
where every pixel is set to the desired paint color, ready to be fed into
the Stage 8 blending formula.

### The image formation model

Every pixel in a photograph is the product of two signals:

```
I(x,y) = R_true(x,y) * S(x,y)

  I(x,y)      = observed pixel (what the camera captured)
  R_true(x,y) = reflectance — the paint color, independent of lighting
  S(x,y)      = shading — local lighting intensity, shadows, ambient
```

This is why the same wall looks brighter near a window and darker in a
corner: `R_true` is the same everywhere, `S(x,y)` varies.

We want to replace `R_true` with `R_target = C_target` while keeping
`S` untouched. Stage 7 builds `R_target`; Stage 8 does the actual swap.

### Why a full-size image (not just masked pixels)?

Stage 8 runs a vectorized element-wise operation:

```
O(x,y) = M_final(x,y) * R(x,y) + (1 - M_final(x,y)) * I(x,y)
```

`R` must be the same shape as `I` and `M_final` so the entire image
can be computed in one NumPy broadcast — no loops, no indexing tricks.
The mask decides which pixels matter; `R` just needs to be present everywhere.

### Why masking is NOT applied here

Separation of concerns. This stage only answers **"what color should the
wall be?"** The **when and where** is entirely Stage 8's responsibility.
This means:
- You can swap the target color without touching any mask logic.
- You can inspect `R(x,y)` independently before running the full pipeline.
- Reusing the same `R` with a different mask costs zero extra compute.

### Optional: brightness normalization

When `normalize=True`, the target color is scaled to match the original
wall's brightness:

```
scale      = ||C_wall|| / ||C_target||      (vector magnitudes)
R_adjusted = C_target * scale
```

Useful for dark-to-light or light-to-dark paint swaps where the raw
target color would otherwise look over- or under-exposed.

### Tunable parameters

| Parameter | Default | What it controls |
|---|---|---|
| `BRIGHTNESS_FACTOR` | `1.0` | Multiply L channel — 0.9 = 10% darker |
| `SATURATION_SCALE` | `1.0` | Scale A/B channels — 1.15 = 15% more vivid |
| `normalize` (arg) | `False` | Brightness-match target to original wall |

### Public API

```python
from color_layer import generate_color_layer

layer = generate_color_layer(
    image,
    target_color=(115, 70, 130),    # Amethyst Ice RGB
    dominant_color=(200, 198, 190), # from Stage 6
    normalize=False,
)
# returns: np.ndarray (H, W, 3) uint8 — ready for Stage 8
```

### Debug visualisation

```bash
python color_layer.py room.jpg 8001-1G
# runs stages 1–7, saves <stem>_<code>_color_layer.png
# 3-panel: original | color layer | preview blend with current mask
```

---

## Stage 4 — Wall Recoloring (`recolor.py`)

Takes the refined mask from Stage 3 and repaints the wall region to any
color in the official Valspar database.

### How the recoloring works

We stay in **LAB color space** for the same reason as Stage 1:
- `L` channel = lightness → we leave this completely untouched
- `A` and `B` channels = color → we shift these toward the target paint color

For every pixel in the wall mask:
```
new_A = original_A × (1 - weight) + target_A × weight
new_B = original_B × (1 - weight) + target_B × weight

where weight = mask_probability × blend_strength
```

This means:
- Pixels deep inside the wall (mask ≈ 1.0) get the full target color
- Edge pixels (mask ≈ 0.3) get a partial shift — smooth, natural feathering
- Shadows and highlights are fully preserved — the wall looks lit by the same light

### Run it

```bash
python recolor.py room.jpg "Lucy Blue"
python recolor.py room.jpg 5001-5C
python recolor.py room.jpg "warm gray" --no-sam
```

Output files are saved next to the source image, named after the color code:
```
room_5001-5C_recolored.jpg    ← full resolution result
room_5001-5C_comparison.jpg   ← 3-panel: original | mask | recolored
```

---

## Color Database (`colors.py` + `colors_valspar.csv`)

1,596 official Valspar / Sherwin-Williams paint colors with full RGB values,
loaded from the official Lowe's Digital Data 2025 dataset.

### Look up a color

```python
from colors import get_color, search_colors, list_colors, list_families

# Exact lookup — by code, name, or hex (any of the three works)
color = get_color("5001-5C")          # by code
color = get_color("Lucy Blue")        # by name (case-insensitive)
color = get_color("#81A9B2")          # by hex

# color.name   → "Lucy Blue"
# color.code   → "5001-5C"
# color.rgb    → (129, 169, 178)
# color.hex    → "81A9B2"
# color.family → "Blues"
# color.lrv    → 36.0

# Fuzzy search — great for partial names or unsure spelling
results = search_colors("dusty teal", n=5)

# Browse all colors in a family
grays = list_colors("Grays")

# See all available families
list_families()
# → ['Blacks', 'Blues', 'Browns', 'Grays', 'Greens',
#    'Neutrals', 'Oranges', 'Pinks', 'Purples',
#    'Reds', 'Teals', 'Whites', 'Yellows']
```

### Browse from the terminal

```bash
python colors.py                  # shows sample colors from each family
python colors.py "Lucy Blue"      # exact + fuzzy results for any query
python colors.py warm beige       # multi-word queries work too
```

---

## Setup

```bash
pip install opencv-python numpy torch torchvision matplotlib pillow
# For Stage 3 also:
pip install git+https://github.com/facebookresearch/segment-anything.git
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
