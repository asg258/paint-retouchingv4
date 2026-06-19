"""
wall_enhance.py — Advanced color analysis and luminance-aware recoloring.

Solves three problems that the base pipeline struggles with:

PROBLEM 1 — SAM gets zero background hints in rooms where DeepLab assigns
  high wall-probability to EVERYTHING (e.g. bathrooms where the background
  class captures warm wood cabinets, tile, AND the painted wall equally).
  Fix: analyse the image's own color distribution to locate pixels that are
  clearly NOT the painted wall, regardless of what DeepLab thought.

PROBLEM 2 — Recoloring dark target colors (LRV < 25) onto bright walls
  looks washed-out because the current pipeline keeps the L (lightness)
  channel completely unchanged. A wall originally at L=180 repainted
  to Cosmic Berry (LRV=11, L≈55) must get darker — the paint really IS dark.
  Fix: Retinex-inspired luminance transfer that scales L proportionally to
  the target lightness while preserving per-pixel shading variation.

PROBLEM 3 — The soft erosion/protection pipeline sometimes still lets low-
  confidence wall probability leak onto dark furniture (V < 80 in HSV).
  Fix: build an additional luminance-based protection mask that explicitly
  suppresses very dark and very desaturated pixels.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

# Hue tolerance (in OpenCV HSV, H is 0–180).
# Pixels whose hue differs from the dominant wall hue by more than this
# are treated as background candidates.  30° ≈ an eighth of the color wheel.
HUE_TOLERANCE: int = 28

# Saturation below this (0–255) → too grey/neutral to be a painted wall.
# Catches mirrors, tiles, and white trim.
SAT_MIN_WALL: int = 45

# Value (brightness) thresholds for background detection.
# Very dark pixels are furniture/cabinets; very bright are blown highlights.
LUMA_DARK_THRESH:  int = 55    # V < this  → definitely furniture (lowered from 75 to avoid
                               # catching wall shadows which can score V ≈ 60-80)
LUMA_BRIGHT_THRESH: int = 230  # V > this  → blown highlight / window

# How many background prompt points to pull from each color-analysis region.
COLOR_BG_POINTS: int = 6

# Retinex L-transfer: how strongly to blend toward the target L value.
# 0.0 = keep original L entirely (old behaviour, washed-out for dark colors)
# 1.0 = fully adopt target L (correct physically, may look too flat)
# 0.6 = good balance — preserves shading variation while reaching target depth
L_BLEND_STRENGTH: float = 0.60

# For target colors with LRV above this, don't shift L at all.
L_SHIFT_MAX_LRV: float = 40.0

# Maximum allowed L-channel shift per pixel (cv2 LAB scale, 0-255).
# Without this cap, the Retinex formula O_L = target_L × (I_L / mean_wall_L)
# overshoots in bright narrow strips (e.g. wall above shower frame) where
# I_L >> mean_wall_L, producing hyperpigmentation artifacts.
# 35 LAB units ≈ a perceptible but not jarring brightness difference.
MAX_L_SHIFT: float = 35.0

# ---------------------------------------------------------------------------
# Texture analysis parameters
# ---------------------------------------------------------------------------

# Laplacian blur radius for computing local texture energy.
# Larger = smoother texture map, less sensitive to noise but may blur
# the boundary between smooth wall and textured tile.
TEXTURE_BLUR_SIGMA:  float = 8.0
TEXTURE_BLUR_KERNEL: int   = 31   # must be odd

# Pixels with normalised texture energy above this are considered
# textured surfaces (tile, wood grain, fabric) — not painted wall.
# Range [0,1].  Lower = more aggressive tile exclusion.
TEXTURE_BG_THRESH: float = 0.20

# Pixels with texture below this are eligible as foreground (wall) prompts.
# Keeps SAM from sampling ambiguous mid-texture areas as wall anchors.
TEXTURE_FG_THRESH: float = 0.12

# How many background prompt points to take from high-texture regions.
TEXTURE_BG_POINTS: int = 12

# Spatial prior: ignore the bottom fraction of the image for FG sampling.
# Painted walls are rarely on the floor — excluding the bottom portion
# prevents floor tiles from being sampled as foreground wall points.
WALL_BOTTOM_EXCLUSION: float = 0.25   # ignore bottom 25% for FG prompts


# ---------------------------------------------------------------------------
# 1. WALL COLOR STATISTICS
# ---------------------------------------------------------------------------

def compute_wall_color_stats(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    threshold: float = 0.65,
) -> dict:
    """
    Compute mean hue, saturation, and luminance of the confident wall region.

    Used to characterise "what the wall actually looks like" so we can
    identify pixels that differ enough to be considered background.

    Args:
        image_rgb:  (H, W, 3) uint8 RGB.
        mask:       (H, W) float32 from DeepLab / SAM.
        threshold:  Pixels above this are used as the wall sample.

    Returns:
        Dict with keys: mean_h, mean_s, mean_v, std_h, std_s, std_v,
                        mean_L, mean_A, mean_B (LAB), n_pixels.
    """
    wall_px = mask > threshold
    if wall_px.sum() < 100:
        # Fall back to full image if the mask is too sparse.
        wall_px = np.ones_like(mask, dtype=bool)

    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    def _stats(arr):
        s = arr[wall_px]
        return float(s.mean()), float(s.std())

    mh, sh = _stats(hsv[:, :, 0])
    ms, ss = _stats(hsv[:, :, 1])
    mv, sv = _stats(hsv[:, :, 2])
    mL, _  = _stats(lab[:, :, 0])
    mA, _  = _stats(lab[:, :, 1])
    mB, _  = _stats(lab[:, :, 2])

    return dict(
        mean_h=mh, std_h=sh,
        mean_s=ms, std_s=ss,
        mean_v=mv, std_v=sv,
        mean_L=mL, mean_A=mA, mean_B=mB,
        n_pixels=int(wall_px.sum()),
    )


# ---------------------------------------------------------------------------
# 2. COLOR-BASED BACKGROUND POINT DETECTION
# ---------------------------------------------------------------------------

def color_based_background_points(
    image_rgb:   np.ndarray,
    coarse_mask: np.ndarray,
    wall_stats:  dict | None = None,
    n_points:    int = COLOR_BG_POINTS,
    n_zones:     int = 4,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Find background (non-wall) SAM prompt points purely from image color.

    Used as a FALLBACK when the mask-based approach yields zero background
    points (common in rooms where DeepLab assigns high background probability
    to everything — bathroom, kitchen, open-plan spaces).

    Strategy:
        1. Build a "definitely NOT wall" mask using three criteria:
           a. Dark pixels (low V in HSV) → furniture, cabinets, vanity
           b. Desaturated pixels (low S) → tile grout, mirrors, chrome
           c. Hue-deviant pixels → any surface whose hue differs strongly
              from the dominant wall hue (if wall stats are provided)
        2. Divide image into n_zones x n_zones cells.
        3. From each cell that contains background pixels, sample one point.

    Args:
        image_rgb:   (H, W, 3) uint8 RGB.
        coarse_mask: (H, W) float32 — not used for content but for spatial context.
        wall_stats:  Dict from compute_wall_color_stats(). If None, only dark/
                     desaturated criteria are used.
        n_points:    Target number of background points.
        n_zones:     Spatial grid divisions per axis.

    Returns:
        (coords, labels) arrays for SAM, or (None, None) if nothing found.
        coords shape: (N, 2) float32, x-y pairs.
        labels shape: (N,) int, all zeros (background).
    """
    h, w = image_rgb.shape[:2]
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    H_ch = hsv[:, :, 0]
    S_ch = hsv[:, :, 1]
    V_ch = hsv[:, :, 2]

    # Criterion A: dark pixels → very likely furniture / non-wall
    dark_mask = V_ch < LUMA_DARK_THRESH

    # Criterion B: desaturated pixels → tiles, mirrors, chrome
    desat_mask = S_ch < SAT_MIN_WALL

    # Criterion C: hue deviation from dominant wall hue
    hue_dev_mask = np.zeros((h, w), dtype=bool)
    if wall_stats is not None:
        mh = wall_stats['mean_h']
        # Circular hue distance — handle 0°/180° wraparound.
        raw_diff = np.abs(H_ch - mh)
        hue_diff = np.minimum(raw_diff, 180.0 - raw_diff)
        # Only mark as deviant if also not dark (dark pixels are already caught)
        hue_dev_mask = (hue_diff > HUE_TOLERANCE) & (V_ch > LUMA_DARK_THRESH)

    bg_candidates = dark_mask | desat_mask | hue_dev_mask

    # Sample spatially across zones so points cover the whole image.
    zone_h = max(1, h // n_zones)
    zone_w = max(1, w // n_zones)
    rng = np.random.default_rng(seed=99)
    pts_per_zone = max(1, n_points // (n_zones * n_zones))

    bg_list: list[tuple[int, int]] = []
    for zi in range(n_zones):
        for zj in range(n_zones):
            y0, y1 = zi * zone_h, min(h, (zi + 1) * zone_h)
            x0, x1 = zj * zone_w, min(w, (zj + 1) * zone_w)
            zone_bg = bg_candidates[y0:y1, x0:x1]
            by, bx  = np.where(zone_bg)
            if len(by) > 0:
                k = min(pts_per_zone, len(by))
                idx = rng.choice(len(by), k, replace=False)
                for i in idx:
                    bg_list.append((int(x0 + bx[i]), int(y0 + by[i])))

    if not bg_list:
        return None, None

    bg_arr    = np.array(bg_list, dtype=np.float32)
    bg_labels = np.zeros(len(bg_arr), dtype=int)
    return bg_arr, bg_labels


# ---------------------------------------------------------------------------
# 3. COLOR-BASED MASK REFINEMENT
# ---------------------------------------------------------------------------

def refine_mask_by_color(
    image_rgb:   np.ndarray,
    coarse_mask: np.ndarray,
    wall_stats:  dict | None = None,
) -> np.ndarray:
    """
    Suppress mask values in pixels that are clearly not the painted wall.

    Multiplies the coarse mask by a per-pixel "wall likelihood" score based
    on how close each pixel's color is to the dominant wall hue/saturation.
    This reduces leakage onto wood cabinets, tile, and mirrors before the
    mask even reaches SAM — giving SAM a cleaner starting point.

    Suppression map M_color ∈ [0,1]:
        - Pixels matching wall hue/sat → M_color ≈ 1 (keep)
        - Dark pixels (V < thresh)     → M_color = 0 (suppress)
        - Desaturated pixels           → M_color reduced
        - Hue-deviant pixels           → M_color reduced

    Args:
        image_rgb:   (H, W, 3) uint8 RGB.
        coarse_mask: (H, W) float32 — mask to refine.
        wall_stats:  Dict from compute_wall_color_stats().

    Returns:
        refined_mask: (H, W) float32, values in [0,1].
    """
    if wall_stats is None:
        wall_stats = compute_wall_color_stats(image_rgb, coarse_mask)

    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    H_ch = hsv[:, :, 0]
    S_ch = hsv[:, :, 1]
    V_ch = hsv[:, :, 2]

    # --- Luminance-based suppression ---
    # Dark pixels get multiplier 0; bright wall pixels stay at 1.
    luma_score = np.clip((V_ch - LUMA_DARK_THRESH) / (200.0 - LUMA_DARK_THRESH), 0, 1)

    # --- Saturation-based suppression ---
    # Unsaturated pixels (mirrors, chrome) lose confidence.
    sat_score = np.clip((S_ch - SAT_MIN_WALL) / (180.0 - SAT_MIN_WALL), 0, 1)

    # --- Hue-based suppression ---
    mh  = wall_stats['mean_h']
    raw_diff = np.abs(H_ch - mh)
    hue_diff = np.minimum(raw_diff, 180.0 - raw_diff).astype(np.float32)
    # Full confidence within HUE_TOLERANCE; ramps down to 0 at 2× tolerance.
    hue_score = np.clip(1.0 - (hue_diff - HUE_TOLERANCE) / HUE_TOLERANCE, 0, 1)

    # Combined color wall-likelihood.
    # Luma and hue scores are most important; sat score is a softer signal.
    color_confidence = luma_score * hue_score * (0.5 + 0.5 * sat_score)

    # Blend with original mask: keep the mask where it agrees with the color
    # analysis, reduce it where the color says "this isn't wall".
    refined = coarse_mask.astype(np.float32) * color_confidence

    return np.clip(refined, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# 4. LUMINANCE-AWARE (RETINEX) RECOLORING
# ---------------------------------------------------------------------------

def luminance_aware_recolor(
    image_rgb:        np.ndarray,
    mask:             np.ndarray,
    target_rgb:       tuple[int, int, int],
    wall_stats:       dict | None = None,
    l_blend_strength: float = L_BLEND_STRENGTH,
    target_lrv:       float | None = None,
) -> np.ndarray:
    """
    Retinex-inspired wall recoloring that correctly handles dark target colors.

    THE PHYSICS
    -----------
    Under the I = R * S model (image = reflectance × shading):
        S(x,y) = I_L(x,y) / mean_wall_L   (normalized per-pixel shading)

    The target pixel should look like:
        O_L(x,y) = target_L * S(x,y)
                 = target_L * (I_L(x,y) / mean_wall_L)

    This preserves the relative brightness variation of the original shading
    (shadows stay dark, highlights stay bright) while shifting the overall
    brightness toward the target paint's lightness.

    Without this, a wall at L=180 painted with Cosmic Berry (L≈55) would
    output L=180 with purple hue — a pastel lilac, not a deep berry.

    BLENDING
    --------
    We blend between the pure Retinex prediction and the original L:
        O_L = (1 - α) * I_L  +  α * (target_L * I_L / mean_wall_L)
    where α = l_blend_strength * mask_value

    This keeps full shading when α=0 and full Retinex when α=1.
    α is automatically set to 0 when the target is light (LRV > L_SHIFT_MAX_LRV)
    since light colors don't need the L correction.

    Args:
        image_rgb:        (H, W, 3) uint8 RGB.
        mask:             (H, W) float32 M_final.
        target_rgb:       (R, G, B) target paint color.
        wall_stats:       Optional dict from compute_wall_color_stats().
        l_blend_strength: How strongly to apply the L correction (0–1).
        target_lrv:       LRV of the target color (0–100). If <= L_SHIFT_MAX_LRV,
                          the L correction is applied. If None, always applies.

    Returns:
        recolored: (H, W, 3) uint8 RGB.
    """
    # Decide whether to apply L correction based on target LRV.
    apply_l_shift = (target_lrv is None) or (target_lrv <= L_SHIFT_MAX_LRV)
    if not apply_l_shift:
        l_blend_strength = 0.0

    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Convert target color to LAB.
    t_bgr = np.array([[[target_rgb[2], target_rgb[1], target_rgb[0]]]], dtype=np.uint8)
    t_lab = cv2.cvtColor(t_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_L = float(t_lab[0, 0, 0])
    target_A = float(t_lab[0, 0, 1])
    target_B = float(t_lab[0, 0, 2])

    # Compute mean wall L from high-confidence pixels.
    if wall_stats is None:
        wall_stats = compute_wall_color_stats(image_rgb, mask)
    mean_wall_L = max(wall_stats['mean_L'], 1.0)   # avoid div by zero

    I_L = lab[:, :, 0]   # original lightness
    M   = np.clip(mask, 0.0, 1.0)

    # ── L channel: Retinex shading-preserving transfer ──────────────────
    # Shading factor: S(x,y) = I_L(x,y) / mean_wall_L
    # Retinex target:          R_L(x,y) = target_L * S(x,y)
    # Blended:                 O_L = (1-α*M) * I_L + α*M * R_L
    if apply_l_shift and l_blend_strength > 0:
        retinex_L = target_L * (I_L / mean_wall_L)
        alpha     = l_blend_strength * M
        O_L       = (1.0 - alpha) * I_L + alpha * retinex_L
        # Clamp the L shift to ±MAX_L_SHIFT.
        # Without this, bright strips (I_L >> mean_wall_L, e.g. wall above
        # a shower frame) produce retinex_L values far outside the wall's
        # normal range, causing the hyperpigmentation artifacts visible above
        # the shower enclosure.
        O_L = np.clip(O_L, I_L - MAX_L_SHIFT, I_L + MAX_L_SHIFT)
    else:
        O_L = I_L.copy()

    # ── A/B channels: shift toward target hue ───────────────────────────
    # We use a weighted blend:  O_c = (1-M)*I_c + M*target_c
    # Full replacement at M=1 (deep wall); tapering at edges.
    O_A = (1.0 - M) * lab[:, :, 1] + M * target_A
    O_B = (1.0 - M) * lab[:, :, 2] + M * target_B

    # Reconstruct LAB → RGB.
    lab_out = np.stack([O_L, O_A, O_B], axis=2)
    lab_out = np.clip(lab_out, 0, 255).astype(np.uint8)
    bgr_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
    rgb_out = cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB)

    return rgb_out


# ---------------------------------------------------------------------------
# 5. LUMINANCE-BASED PROTECTION BOOST
# ---------------------------------------------------------------------------

def luminance_protection_mask(
    image_rgb:       np.ndarray,
    existing_protect: np.ndarray,
    dark_threshold:   int   = LUMA_DARK_THRESH,
    blend_sigma:      float = 3.0,
) -> np.ndarray:
    """
    Supplement the existing protection mask with luminance-based suppression.

    Dark pixels (V < dark_threshold in HSV) are almost certainly furniture,
    cabinets, or vanity units — not painted wall.  Force their protection
    value to 1 so Stage 8 blending never touches them.

    This is purely additive: it can only INCREASE the protection, never
    reduce it, so it can't break the mask in wall areas.

    Args:
        image_rgb:        (H, W, 3) uint8 RGB.
        existing_protect: (H, W) float32 from Stage 5 protect.py.
        dark_threshold:   V values below this are fully protected.
        blend_sigma:      Gaussian sigma to smooth the dark-pixel boundary.

    Returns:
        boosted_protect: (H, W) float32, values in [0,1].
    """
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    V   = hsv[:, :, 2]

    # Binary dark-pixel mask.
    dark = (V < dark_threshold).astype(np.float32)

    # Smooth edges so the transition isn't a hard step.
    k = 15
    dark_smooth = cv2.GaussianBlur(dark, (k, k), blend_sigma)

    # Cap the luminance-based signal before merging.
    # We don't want wall shadows (which can have low V) to be mistakenly
    # treated as "fully protected furniture".  Capping at 0.70 means the
    # luminance signal can boost protection but cannot drive it to 1.0 on
    # its own — it still needs the wall mask to be low for M_final to be low.
    dark_smooth = np.clip(dark_smooth, 0.0, 0.70)

    # Take element-wise max: existing protection OR luminance-based protection.
    boosted = np.maximum(existing_protect.astype(np.float32), dark_smooth)
    return np.clip(boosted, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 6. TEXTURE-BASED MATERIAL SEPARATION
# ---------------------------------------------------------------------------

def compute_texture_map(
    image_rgb:   np.ndarray,
    blur_sigma:  float = TEXTURE_BLUR_SIGMA,
    blur_kernel: int   = TEXTURE_BLUR_KERNEL,
) -> np.ndarray:
    """
    Compute a per-pixel texture energy map using the Laplacian operator.

    WHY THIS WORKS FOR SEPARATING WALL FROM TILE/WOOD
    --------------------------------------------------
    The Laplacian ∇²I measures the second spatial derivative of intensity:
        ∇²I(x,y) = ∂²I/∂x² + ∂²I/∂y²

    It produces a large response at edges and fine texture patterns.
    Painted drywall is spectrally smooth — very low Laplacian magnitude.
    Tile grout lines, wood grain, and fabric all produce high Laplacian
    magnitude because they contain repeated high-frequency edge patterns.

    By blurring the absolute Laplacian with a wide Gaussian, we get a
    smooth map of local texture energy:

        T(x,y) = GaussianBlur(|∇²I(x,y)|, σ)

    Low T → smooth surface → likely painted wall
    High T → textured surface → likely tile, wood, fabric, or trim

    Args:
        image_rgb:   (H, W, 3) uint8 RGB.
        blur_sigma:  Gaussian sigma for local texture energy pooling.
        blur_kernel: Gaussian kernel size (must be odd).

    Returns:
        texture_map: (H, W) float32, normalised to [0, 1].
                     0 = very smooth (likely painted wall)
                     1 = highly textured (likely tile / wood)
    """
    # Work on luminance (grayscale) to avoid color variation contaminating
    # the texture signal — a yellow wall and beige tile can look similar in
    # color but differ sharply in texture.
    bgr  = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Laplacian: second derivative → high at edges and fine detail.
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)

    # Absolute value: we care about magnitude, not sign.
    lap_abs = np.abs(lap)

    # Pool locally with a wide Gaussian so individual edge pixels don't
    # dominate — we want the average texture density in a neighbourhood.
    k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
    texture_energy = cv2.GaussianBlur(lap_abs, (k, k), blur_sigma)

    # Normalise to [0, 1].  Use 99th percentile to avoid outliers
    # (e.g. a very bright specular highlight) compressing the range.
    p99 = float(np.percentile(texture_energy, 99))
    if p99 < 1e-6:
        return np.zeros_like(texture_energy)
    texture_map = np.clip(texture_energy / p99, 0.0, 1.0).astype(np.float32)

    return texture_map


def texture_based_background_points(
    image_rgb:    np.ndarray,
    texture_map:  np.ndarray,
    n_points:     int   = TEXTURE_BG_POINTS,
    n_zones:      int   = 4,
    bg_threshold: float = TEXTURE_BG_THRESH,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Sample background SAM prompt points from high-texture image regions.

    High-texture pixels (tile grout lines, wood grain) are almost certainly
    NOT painted wall. Providing them as SAM background anchors gives SAM
    the discriminative signal it needs to exclude those surfaces from the
    wall mask — even when the DeepLab background class covers everything.

    Args:
        image_rgb:    (H, W, 3) uint8 RGB (used for zone dimensions).
        texture_map:  (H, W) float32 from compute_texture_map().
        n_points:     Target number of background prompt points.
        n_zones:      Spatial grid divisions per axis.
        bg_threshold: Texture value above which a pixel is a BG candidate.

    Returns:
        (coords, labels) or (None, None) if no candidates found.
    """
    h, w = image_rgb.shape[:2]
    rng  = np.random.default_rng(seed=77)
    pts_per_zone = max(1, n_points // (n_zones * n_zones))
    zone_h = max(1, h // n_zones)
    zone_w = max(1, w // n_zones)

    bg_list: list[tuple[int, int]] = []
    for zi in range(n_zones):
        for zj in range(n_zones):
            y0, y1 = zi * zone_h, min(h, (zi + 1) * zone_h)
            x0, x1 = zj * zone_w, min(w, (zj + 1) * zone_w)
            zone_t  = texture_map[y0:y1, x0:x1]
            ty, tx  = np.where(zone_t >= bg_threshold)
            if len(ty) > 0:
                k = min(pts_per_zone, len(ty))
                idx = rng.choice(len(ty), k, replace=False)
                for i in idx:
                    bg_list.append((int(x0 + tx[i]), int(y0 + ty[i])))

    if not bg_list:
        return None, None
    bg_arr    = np.array(bg_list, dtype=np.float32)
    bg_labels = np.zeros(len(bg_arr), dtype=int)
    return bg_arr, bg_labels


def texture_suppressed_mask(
    mask:        np.ndarray,
    texture_map: np.ndarray,
    suppress_threshold: float = TEXTURE_BG_THRESH,
    suppress_strength:  float = 0.85,
) -> np.ndarray:
    """
    Suppress mask values in high-texture regions after SAM refinement.

    Even after SAM, the soft mask (M_refined = SAM_binary × M₀) retains
    high values in tile areas because M₀ was high everywhere (DeepLab
    background class). Multiplying by a texture suppression factor reduces
    the mask weight in textured regions so the blending formula
    O = M*R + (1-M)*I gives tile its original color back.

    Formula:
        suppress_weight(x,y) = 1 - suppress_strength × clip(T(x,y)/1, 0, 1)
        M_out(x,y) = M(x,y) × suppress_weight(x,y)

    Args:
        mask:               (H, W) float32 wall mask.
        texture_map:        (H, W) float32 from compute_texture_map().
        suppress_threshold: Texture above this starts suppressing the mask.
        suppress_strength:  Maximum suppression factor (0=no effect, 1=full).

    Returns:
        M_suppressed: (H, W) float32, values in [0, 1].
    """
    # How much to suppress, scaled by how far above threshold the texture is.
    excess   = np.clip((texture_map - suppress_threshold) / (1.0 - suppress_threshold + 1e-6), 0.0, 1.0)
    suppress = 1.0 - suppress_strength * excess
    return np.clip(mask.astype(np.float32) * suppress, 0.0, 1.0)
