"""
mask_pipeline.py — Unified, material-aware wall mask generation.

ARCHITECTURAL RATIONALE
-----------------------
The previous pipeline had four independent stages that each suppressed the mask
(color refinement, erosion, dilation-protection, luminance boost). Each was
correct in isolation but they composed non-linearly on a nearly-uniform input
(DeepLab background scores 0.60–1.0 everywhere), causing fragmentation.

This module replaces those stages with ONE clean combination formula:

    M_final(x,y) = M_sam(x,y) × W_material(x,y)

Where:
  M_sam       = SAM binary mask × M_coarse  (boundary accuracy from SAM)
  W_material  = combined material weight    (semantic signal the models lack)

W_material is composed of three independent, interpretable signals:

  W_texture(x,y)  — walls are smooth; tile/wood are textured
  W_color(x,y)    — walls match the dominant wall color; cabinets/tile deviate
  W_gradient(x,y) — walls have slow spatial gradients; high-gradient = edge/texture

These are multiplied:
    W_material = W_texture × W_color × W_gradient

All three signals are computed directly from the image pixels — no model
inference needed. They are calibrated against the dominant wall color extracted
from the high-confidence centre of M_coarse.

MATHEMATICAL FORMULATION
------------------------
1. Local variance (texture):
        Var(x,y) = E[I²] - (E[I])²      over an N×N local window
        W_texture = exp(−λ_t × Var_norm)

2. Color distance (LAB space):
        d(x,y) = ||I_LAB(x,y) − C_wall_LAB||₂
        W_color = exp(−d(x,y) / σ_c)

3. Gradient magnitude (planarity):
        G(x,y) = √(Gx² + Gy²)   (Sobel)
        G_local = GaussianBlur(G, σ_g)
        W_gradient = exp(−λ_g × G_local_norm)

4. Grid-based normalisation:
        The image is divided into P×P patches. Within each patch, M_final
        values are normalised so locally-dominant wall regions maintain
        connectivity even when global thresholds would fragment them.

5. Final mask combination:
        M_material = clip(W_texture × W_color × W_gradient, 0, 1)
        M_final    = M_sam × M_material
        M_final    = GaussianBlur(M_final, σ_smooth)   [feather edges]
"""

from __future__ import annotations

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

# --- Texture (local variance) ---
VAR_WINDOW_SIZE:  int   = 15
# Reduced to 0.5: walls are not perfectly flat — mild bumps, paint texture,
# and shadow gradients all add variance that should not suppress valid wall pixels.
VAR_LAMBDA:       float = 0.5

# --- Color distance (LAB space) ---
COLOR_SIGMA:      float = 80.0

# --- Gradient planarity ---
GRAD_SIGMA:       float = 10.0
GRAD_LAMBDA:      float = 1.0

# --- Additive combination weights ---
# Replaced multiplicative (W_tex × W_col × W_grad) with additive formula.
# Multiplication amplifies small errors — if any one signal is weak, the
# product collapses even when the other two are strong. Additive weighting
# gives smooth influence without catastrophic suppression.
#
# W_material = 0.5 + 0.5 × (w_t×W_tex + w_c×W_col + w_g×W_grad)
#
# This guarantees W_material ∈ [0.5, 1.0] — the floor is structural,
# not a parameter. Any pixel gets at minimum 50% of its ADE20K mask value.
W_TEX_WEIGHT:  float = 0.4   # texture is most discriminative
W_COL_WEIGHT:  float = 0.3   # color helpful but less reliable in shadows
W_GRAD_WEIGHT: float = 0.3   # gradient complements texture

# --- Strong-wall preservation threshold ---
STRONG_WALL_THRESH: float = 0.70
STRONG_WALL_MIN:    float = 0.60

# --- Periodic texture penalty (tile grout detection) ---
# Tile surfaces have structured, repeating Laplacian patterns.
# We detect this via the variance of the Laplacian (variance-of-edges),
# which is high for periodic grout-line patterns and low for smooth paint.
TILE_VAR_WINDOW: int   = 15     # local window for Laplacian variance
TILE_LAMBDA:     float = 3.0    # suppression strength (higher = harder cut)
TILE_VAR_THRESH: float = 0.15   # normalised Laplacian variance above which
                                 # the tile penalty activates

# --- Reflection / specular penalty (mirror / glass detection) ---
# Mirrors and polished glass appear as regions with high per-pixel RGB channel
# variance (reflected scene has different hue ratios than a painted wall) plus
# high local hue variation (the reflected scene contains many different colors).
REFLECT_RGB_SIGMA:  float = 25.0  # suppression strength for inter-channel variance
REFLECT_HUE_WINDOW: int   = 21    # window for local hue std-dev computation
REFLECT_HUE_THRESH: float = 0.20  # normalised hue-std above which reflection
                                   # penalty activates

# --- Gaussian color distance (Step 3) ---
# Replaced exp(-d/σ) with exp(-d²/2σ²) for sharper discrimination:
# similar colors remain near 1, different materials drop faster at large d.
COLOR_SIGMA_GAUSS: float = 40.0   # Gaussian σ in LAB units (replaces COLOR_SIGMA)

# --- Connected-component filtering (Step 4) ---
# Remove isolated mask regions that are too small to be real wall planes.
MIN_COMPONENT_PX: int   = 800    # components smaller than this are zeroed
COMPONENT_THRESH: float = 0.35   # binarise threshold for component analysis

# --- Final smoothing ---
SMOOTH_SIGMA:     float = 2.0
SMOOTH_KERNEL:    int   = 15

# --- Noise cleanup ---
NOISE_THRESHOLD:  float = 0.05


# ---------------------------------------------------------------------------
# 1. Local Variance Map (Texture)
# ---------------------------------------------------------------------------

def compute_local_variance(
    image_rgb:   np.ndarray,
    window_size: int   = VAR_WINDOW_SIZE,
    lam:         float = VAR_LAMBDA,
) -> np.ndarray:
    """
    Compute a per-pixel texture weight using local intensity variance.

    Formula:
        E[I](x,y)   = boxFilter(I_gray, window_size)
        E[I²](x,y)  = boxFilter(I_gray², window_size)
        Var(x,y)    = E[I²] − (E[I])²          [local variance]
        Var_norm    = Var / percentile95(Var)   [normalised to ~[0,1]]
        W_tex(x,y)  = exp(−λ × Var_norm)       [exponential suppression]

    boxFilter is O(1) per pixel regardless of window size — much faster than
    a naive NxN convolution and numerically stable.

    High variance → textured surface (tile grout, wood grain) → W_tex ≈ 0
    Low variance  → smooth surface   (painted wall)            → W_tex ≈ 1

    Args:
        image_rgb:   (H, W, 3) uint8 RGB.
        window_size: Local window size for variance computation.
        lam:         Exponential decay constant.

    Returns:
        W_texture: (H, W) float32, values in [0, 1].
    """
    bgr  = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    k = (window_size, window_size)

    # E[I] and E[I²] using separable box filter (efficient O(1)/pixel)
    mean_I   = cv2.boxFilter(gray,    cv2.CV_32F, k, normalize=True)
    mean_I2  = cv2.boxFilter(gray**2, cv2.CV_32F, k, normalize=True)
    variance = np.clip(mean_I2 - mean_I**2, 0, None)

    # Normalise using the 95th percentile to avoid outliers compressing the scale
    p95 = float(np.percentile(variance, 95))
    if p95 < 1e-6:
        return np.ones_like(variance)
    var_norm = np.clip(variance / p95, 0.0, 1.0)

    # Exponential suppression: high-variance pixels get weight near 0
    return np.exp(-lam * var_norm).astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Color Distance Map (LAB)
# ---------------------------------------------------------------------------

def compute_color_distance_weight(
    image_rgb:  np.ndarray,
    wall_color_rgb: tuple[int, int, int],
    sigma:      float = COLOR_SIGMA_GAUSS,   # now uses Gaussian formula
) -> np.ndarray:
    """
    Compute a per-pixel color weight based on LAB distance from wall color.

    Formula:
        I_LAB(x,y) = RGB_to_LAB(I(x,y))
        C_wall_LAB = RGB_to_LAB(C_wall)
        d(x,y)     = ||I_LAB(x,y) − C_wall_LAB||₂  (Euclidean in LAB)
        W_color    = exp(−d / σ)

    LAB is used because distances are perceptually uniform — a delta of 10
    always represents the same perceptual difference regardless of which color.
    This avoids the RGB bias where blue distances are compressed relative to red.

    σ controls the acceptance bandwidth:
        σ = 15 → only very similar colors survive
        σ = 28 → accepts colors within one "color family"
        σ = 50 → very permissive

    Painted walls tend to be within d ≈ 20–30 of each other (same paint +
    lighting variation). Tile grout (d ≈ 40+), wood (d ≈ 35+), and granite
    (d ≈ 30+) are systematically further away.

    Args:
        image_rgb:       (H, W, 3) uint8 RGB.
        wall_color_rgb:  (R, G, B) dominant wall color from Stage 6.
        sigma:           LAB distance bandwidth.

    Returns:
        W_color: (H, W) float32, values in (0, 1].
    """
    bgr      = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    lab_img  = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Convert the wall color reference point to LAB
    wall_bgr = np.array([[[wall_color_rgb[2], wall_color_rgb[1], wall_color_rgb[0]]]],
                         dtype=np.uint8)
    wall_lab = cv2.cvtColor(wall_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]

    # Per-pixel Euclidean distance in LAB space
    diff     = lab_img - wall_lab[np.newaxis, np.newaxis, :]  # (H, W, 3)
    dist     = np.sqrt(np.sum(diff**2, axis=2))               # (H, W)

    # Gaussian formula: exp(−d²/2σ²) gives sharper discrimination than
    # the linear-decay exp(−d/σ). At d=σ, both give similar values,
    # but at d=2σ the Gaussian drops faster — different materials (tile,
    # wood, granite) at d≈50–80 LAB units are more strongly suppressed.
    return np.exp(-(dist ** 2) / (2.0 * sigma ** 2)).astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Gradient Planarity Map
# ---------------------------------------------------------------------------

def compute_gradient_weight(
    image_rgb: np.ndarray,
    pool_sigma: float = GRAD_SIGMA,
    lam:        float = GRAD_LAMBDA,
) -> np.ndarray:
    """
    Compute a per-pixel planarity weight based on local gradient magnitude.

    Formula:
        Gx, Gy      = Sobel(I_gray, dx=1), Sobel(I_gray, dy=1)
        G(x,y)      = √(Gx² + Gy²)              [edge magnitude]
        G_local     = GaussianBlur(G, σ_g)       [local average edge density]
        G_norm      = G_local / percentile95     [normalised]
        W_grad      = exp(−λ × G_norm)

    Painted walls are planar with slow lighting gradients → low edge density.
    Tile surfaces have dense regular edge patterns (grout lines) → high density.
    Wood grain has directional high-frequency edges → high density.
    Object boundaries have isolated strong edges → high density.

    This differs from the Laplacian texture map in that Sobel responds to
    oriented edges (grout lines are strongly directional), whereas Laplacian
    responds to all-direction edge density. Both are useful; gradient is more
    discriminative for tile patterns.

    Args:
        image_rgb:  (H, W, 3) uint8 RGB.
        pool_sigma: Gaussian pooling sigma for local edge density.
        lam:        Exponential decay constant.

    Returns:
        W_gradient: (H, W) float32, values in (0, 1].
    """
    bgr  = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    G  = np.sqrt(gx**2 + gy**2)

    # Pool locally so individual edge pixels don't dominate
    k_size = int(6 * pool_sigma + 1) | 1   # ensure odd
    G_pool = cv2.GaussianBlur(G, (k_size, k_size), pool_sigma)

    p95 = float(np.percentile(G_pool, 95))
    if p95 < 1e-6:
        return np.ones_like(G_pool)
    G_norm = np.clip(G_pool / p95, 0.0, 1.0)

    return np.exp(-lam * G_norm).astype(np.float32)


# ---------------------------------------------------------------------------
# NEW — Periodic Texture Penalty (tile grout detection)
# ---------------------------------------------------------------------------

def compute_periodic_texture_penalty(
    image_rgb:   np.ndarray,
    window:      int   = TILE_VAR_WINDOW,
    lam:         float = TILE_LAMBDA,
    var_thresh:  float = TILE_VAR_THRESH,
) -> np.ndarray:
    """
    Detect periodic, structured texture (tile grout) using variance-of-Laplacian.

    WHY: The Laplacian |∇²I| has high magnitude at every edge. For smooth paint,
    edges are sparse and the local variance of the Laplacian is LOW. For tile,
    the regular grout-line pattern produces a dense, periodically-repeating
    Laplacian signal whose local variance is HIGH. This distinguishes tile from
    paint better than simple gradient magnitude because it responds to
    REGULARITY, not just the presence of edges.

    Formula:
        L(x,y)      = |∇² I_gray|
        μ_L         = boxFilter(L, window)
        μ_L²        = boxFilter(L², window)
        VarL(x,y)   = μ_L² − (μ_L)²            [local variance of Laplacian]
        VarL_norm   = clip(VarL / percentile₉₅, 0, 1)
        W_tile      = exp(−λ × max(0, VarL_norm − thresh))

    The threshold means the penalty only activates above a minimum variance
    level, so smooth-but-textured materials (e.g. slightly rough plaster) are
    not incorrectly penalised.

    Returns:
        W_tile: (H, W) float32, values ∈ (0, 1]. Low = tile-like. High = smooth.
    """
    bgr  = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    lap  = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))

    k = (window, window)
    mu_L  = cv2.boxFilter(lap,    cv2.CV_32F, k, normalize=True)
    mu_L2 = cv2.boxFilter(lap**2, cv2.CV_32F, k, normalize=True)
    var_L = np.clip(mu_L2 - mu_L**2, 0, None)

    p95      = float(np.percentile(var_L, 95)) + 1e-6
    var_norm = np.clip(var_L / p95, 0.0, 1.0)

    # Penalty activates only above the threshold to avoid false positives on
    # rough plaster or embossed wallpaper that isn't actually tile.
    excess   = np.maximum(var_norm - var_thresh, 0.0)
    W_tile   = np.exp(-lam * excess).astype(np.float32)

    return W_tile


# ---------------------------------------------------------------------------
# NEW — Reflection / Specular Penalty (mirror and glass detection)
# ---------------------------------------------------------------------------

def compute_reflection_penalty(
    image_rgb:   np.ndarray,
    rgb_sigma:   float = REFLECT_RGB_SIGMA,
    hue_window:  int   = REFLECT_HUE_WINDOW,
    hue_thresh:  float = REFLECT_HUE_THRESH,
) -> np.ndarray:
    """
    Suppress mask on mirror and reflective glass surfaces.

    WHY: A painted wall has a consistent single color across the surface —
    per-pixel variance across the R, G, B channels is LOW (all three
    channels track the same painted tone). A mirror reflects diverse scene
    content (furniture, towels, sky through a window), so the R, G, B values
    at adjacent pixels change independently → high inter-channel variance
    AND high local hue variation.

    Two signals combined:
    1. Inter-channel RGB variance:
           var_rgb(x,y) = Var([R,G,B]) per pixel
           W_rgb        = exp(−var_rgb² / (2·rgb_sigma²))
       Low on uniform painted surfaces. High on color-diverse reflections.

    2. Local circular hue standard deviation:
           Hue_local_std = std(H in local window), using circular statistics
           W_hue         = 1 if std < hue_thresh, ramps to 0 above
       A wall has nearly constant hue. A mirror reflection contains many hues.

    Final: W_reflect = W_rgb × W_hue  (both must be clean to pass)

    Returns:
        W_reflect: (H, W) float32, values ∈ (0, 1]. Low = likely reflection.
    """
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    # --- Signal 1: per-pixel inter-channel variance ---
    img_f   = image_rgb.astype(np.float32)
    mean_ch = img_f.mean(axis=2, keepdims=True)
    var_rgb = np.mean((img_f - mean_ch) ** 2, axis=2)   # (H, W)
    p95_rgb = float(np.percentile(var_rgb, 95)) + 1e-6
    var_rgb_norm = np.clip(var_rgb / p95_rgb, 0.0, 1.0)
    W_rgb   = np.exp(-(var_rgb_norm ** 2) / (2 * (rgb_sigma / 100.0) ** 2))

    # --- Signal 2: local hue circular std-dev ---
    # Hue in OpenCV: 0–180. Convert to unit circle for circular statistics.
    H    = hsv[:, :, 0]
    theta = H * (np.pi / 90.0)   # map [0,180] → [0, 2π]
    sin_t = cv2.boxFilter(np.sin(theta).astype(np.float32), cv2.CV_32F,
                          (hue_window, hue_window), normalize=True)
    cos_t = cv2.boxFilter(np.cos(theta).astype(np.float32), cv2.CV_32F,
                          (hue_window, hue_window), normalize=True)
    R_bar     = np.sqrt(sin_t**2 + cos_t**2)          # circular mean length
    hue_std   = np.sqrt(np.clip(1.0 - R_bar, 0, 1))   # circular std ∈ [0,1]
    # Suppress where local hue std exceeds the threshold.
    hue_excess = np.maximum(hue_std - hue_thresh, 0.0)
    W_hue      = np.exp(-5.0 * hue_excess).astype(np.float32)

    return (W_rgb * W_hue).astype(np.float32)


# ---------------------------------------------------------------------------
# NEW — Small Component Filter (Step 4)
# ---------------------------------------------------------------------------

def filter_small_components(
    mask:       np.ndarray,
    threshold:  float = COMPONENT_THRESH,
    min_pixels: int   = MIN_COMPONENT_PX,
) -> np.ndarray:
    """
    Remove small isolated mask regions that cannot be real wall planes.

    Real painted wall surfaces are large and continuous. Isolated blobs of
    a few hundred pixels are almost certainly:
        - Texture artifacts (grout patches that slipped through W_tile)
        - Reflection fragments (mirror edge leakage)
        - Segmentation noise

    Method: binarise at threshold, compute connected components, zero out
    components below min_pixels, apply as a multiplicative gate on the
    soft mask to preserve the original probability values in kept regions.

    Args:
        mask:       (H, W) float32 soft mask.
        threshold:  Binarisation level for component analysis.
        min_pixels: Components smaller than this are removed.

    Returns:
        filtered: (H, W) float32 — small components zeroed, others unchanged.
    """
    binary    = (mask > threshold).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

    keep = np.zeros_like(binary)
    for i in range(1, n_labels):   # skip label 0 = background
        if stats[i, cv2.CC_STAT_AREA] >= min_pixels:
            keep[labels == i] = 1

    return (mask * keep.astype(np.float32))


# ---------------------------------------------------------------------------
# 5. Combined Material-Aware Refinement
# ---------------------------------------------------------------------------

def material_aware_mask(
    image_rgb:       np.ndarray,
    sam_mask:        np.ndarray,
    wall_color_rgb:  tuple[int, int, int] | None = None,
    var_window:      int   = VAR_WINDOW_SIZE,
    var_lambda:      float = VAR_LAMBDA,
    color_sigma:     float = COLOR_SIGMA,
    grad_lambda:     float = GRAD_LAMBDA,
    smooth_sigma:    float = SMOOTH_SIGMA,
    noise_threshold: float = NOISE_THRESHOLD,
) -> tuple[np.ndarray, dict]:
    """
    Apply material-aware refinement to a SAM-refined wall mask.

    Replaces the previous fragmented approach (color refinement, protection,
    luminance boost, alpha scaling) with one principled combination:

        W_material = W_texture × W_color × W_gradient
        M_refined  = M_sam × W_material
        M_final    = GridNorm(M_refined)
        M_final    = GaussianBlur(M_final)

    If wall_color_rgb is None, the dominant wall color is estimated from
    the high-confidence centre of the SAM mask (M > 0.7).

    Args:
        image_rgb:       (H, W, 3) uint8 RGB image.
        sam_mask:        (H, W) float32 SAM-refined mask, values in [0, 1].
        wall_color_rgb:  (R, G, B) reference wall color. Auto-detected if None.
        var_window:      Local variance window size.
        var_lambda:      Texture suppression strength.
        color_sigma:     Color distance bandwidth (LAB units).
        grad_lambda:     Gradient suppression strength.
        patch_size:      Grid patch size for local normalisation.
        smooth_sigma:    Final Gaussian feathering sigma.
        noise_threshold: Final noise floor.

    Returns:
        M_final:   (H, W) float32, values in [0, 1].
        debug:     Dict of intermediate maps for visualisation.
    """
    # --- Estimate wall color if not provided ---
    if wall_color_rgb is None:
        wall_color_rgb = _estimate_wall_color(image_rgb, sam_mask)

    # --- Compute base material weight maps ---
    W_tex  = compute_local_variance(image_rgb, window_size=var_window, lam=var_lambda)
    W_col  = compute_color_distance_weight(image_rgb, wall_color_rgb, sigma=color_sigma)
    W_grad = compute_gradient_weight(image_rgb, lam=grad_lambda)

    # --- Additive base material weight (preserves large wall planes) ---
    W_material = 0.5 + 0.5 * (
        W_TEX_WEIGHT  * W_tex  +
        W_COL_WEIGHT  * W_col  +
        W_GRAD_WEIGHT * W_grad
    )
    W_material = np.clip(W_material, 0.0, 1.0).astype(np.float32)

    # --- Targeted suppressions (multiplicative, applied AFTER the additive base) ---
    # These are designed for specific failure modes: tile and mirrors.
    # They are multiplicative because they represent hard physical evidence
    # ("this IS tile", "this IS a mirror") that should override the base weight.

    # Periodic texture penalty — suppresses tile grout patterns.
    W_tile    = compute_periodic_texture_penalty(image_rgb)
    # Reflection penalty — suppresses mirror and polished glass.
    W_reflect = compute_reflection_penalty(image_rgb)

    # Apply targeted suppressions to the base weight.
    W_combined = W_material * W_tile * W_reflect
    W_combined = np.clip(W_combined, 0.0, 1.0).astype(np.float32)

    # --- Apply to SAM mask ---
    M_material = sam_mask.astype(np.float32) * W_combined

    # --- Strong-wall preservation ---
    # Where ADE20K+SAM was very confident, the combined filter cannot override it.
    # This is applied AFTER the targeted suppressions so tile/mirror suppression
    # still works even in high-confidence regions.
    strong_wall = sam_mask > STRONG_WALL_THRESH
    M_material  = np.where(
        strong_wall,
        np.maximum(M_material, STRONG_WALL_MIN),
        M_material,
    ).astype(np.float32)

    # --- Morphological opening (Step 5): remove thin streaks and noise ---
    # A small 3×3 open removes isolated pixel noise and thin vertical/horizontal
    # artifacts without affecting large continuous wall regions.
    open_k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    M_opened  = cv2.morphologyEx(M_material, cv2.MORPH_OPEN, open_k)

    # --- Connected component filtering (Step 4): remove isolated blobs ---
    # Real wall planes are large. Anything under MIN_COMPONENT_PX pixels
    # is a texture artifact, reflection fragment, or segmentation noise.
    M_filtered = filter_small_components(M_opened)

    # --- Final Gaussian feathering ---
    k = int(smooth_sigma * 6 + 1) | 1
    M_smooth = cv2.GaussianBlur(M_filtered, (k, k), smooth_sigma)

    # --- Noise floor ---
    M_smooth[M_smooth < noise_threshold] = 0.0
    M_final = np.clip(M_smooth, 0.0, 1.0).astype(np.float32)

    # --- Mask distribution report ---
    print(f"[mask_pipeline] M_final — "
          f"mean={M_final.mean():.3f}  "
          f"min={M_final.min():.3f}  "
          f"max={M_final.max():.3f}  "
          f"coverage={(M_final > 0.3).mean()*100:.1f}%")

    debug = {
        "W_texture":       W_tex,
        "W_color":         W_col,
        "W_gradient":      W_grad,
        "W_tile":          W_tile,
        "W_reflect":       W_reflect,
        "W_material_base": W_material,
        "W_combined":      W_combined,
        "M_sam_raw":       sam_mask,
        "M_after_mat":     M_material,
        "M_final":         M_final,
        "wall_color":      wall_color_rgb,
    }
    return M_final, debug


# ---------------------------------------------------------------------------
# 6. Mask Quality Metrics
# ---------------------------------------------------------------------------

def compute_mask_metrics(
    mask:      np.ndarray,
    image_rgb: np.ndarray,
) -> dict:
    """
    Quantify mask quality along three axes.

    Metrics:
        edge_alignment     — do mask boundaries follow image edges?
        region_continuity  — is the mask one connected region or many islands?
        material_variance  — how uniform is the image inside the mask?

    1. Edge alignment (Dice-like overlap of gradients):
            ∇M = |Sobel(M)|
            ∇I = |Sobel(I_gray)|
            edge_alignment = 2 × sum(∇M × ∇I) / (sum(∇M) + sum(∇I) + ε)
       Value ∈ [0,1]. Higher = mask boundaries coincide with image edges.

    2. Region continuity (inverse fragmentation):
            n_components = number of connected components in M > 0.5
            continuity = 1 / (1 + log(n_components))
       Value ∈ (0,1]. Higher = fewer, more connected regions.

    3. Material variance (colour consistency inside mask):
            pixels_in_mask = image[M > 0.5]
            material_variance = mean(var(pixels, axis=0))   over RGB channels
       Lower = more uniform colour (more likely a single painted surface).
    """
    M   = mask.astype(np.float32)
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 1. Edge alignment
    gx_M = cv2.Sobel(M,    cv2.CV_32F, 1, 0, ksize=3)
    gy_M = cv2.Sobel(M,    cv2.CV_32F, 0, 1, ksize=3)
    gx_I = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy_I = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_M = np.sqrt(gx_M**2 + gy_M**2)
    grad_I = np.sqrt(gx_I**2 + gy_I**2)
    edge_alignment = float(
        2.0 * (grad_M * grad_I).sum() /
        (grad_M.sum() + grad_I.sum() + 1e-6)
    )

    # 2. Region continuity
    binary = (M > 0.5).astype(np.uint8)
    n_comp, _ = cv2.connectedComponents(binary)
    n_comp = max(1, n_comp - 1)   # subtract background component
    continuity = float(1.0 / (1.0 + np.log(n_comp)))

    # 3. Material variance
    in_mask = image_rgb[M > 0.5].astype(np.float32)
    material_variance = float(np.mean(np.var(in_mask, axis=0))) if len(in_mask) > 0 else 0.0

    return {
        "edge_alignment":    round(edge_alignment, 4),
        "region_continuity": round(continuity, 4),
        "material_variance": round(material_variance, 2),
        "n_components":      n_comp,
        "wall_coverage_pct": round(float((M > 0.5).mean() * 100), 1),
    }


# ---------------------------------------------------------------------------
# 7. Visualisation
# ---------------------------------------------------------------------------

def visualize_material_pipeline(
    image_rgb:   np.ndarray,
    debug:       dict,
    metrics:     dict | None = None,
    save_path:   str | Path | None = None,
) -> None:
    """
    Six-panel diagnostic figure showing all material weight maps.

    Panels:
        [0] Original image
        [1] W_texture     — smooth=bright (wall), textured=dark (tile/wood)
        [2] W_color       — on-color=bright, off-color=dark
        [3] W_gradient    — planar=bright, edgy=dark
        [4] W_material    — combined weight (product of above three)
        [5] M_final       — final mask after all processing
    """
    panels = [
        (image_rgb,             "Original image",         None),
        (debug["W_texture"],    "W_texture\n(smooth=wall, textured=tile/wood)", "RdYlGn"),
        (debug["W_color"],      "W_color\n(on-wall-color=bright)",              "RdYlGn"),
        (debug["W_gradient"],   "W_gradient\n(planar=bright, edgy=dark)",       "RdYlGn"),
        (debug["W_material"],   "W_material = W_tex × W_col × W_grad",         "RdYlGn"),
        (debug["M_final"],      "M_final (wall mask)",                           "hot"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    title = "Material-Aware Mask Pipeline"
    if metrics:
        title += (f"\n  edge_align={metrics['edge_alignment']:.3f}  "
                  f"continuity={metrics['region_continuity']:.3f}  "
                  f"mat_var={metrics['material_variance']:.1f}  "
                  f"coverage={metrics['wall_coverage_pct']}%")
    fig.suptitle(title, fontsize=11)

    for ax, (data, label, cmap) in zip(axes, panels):
        if cmap is None:
            ax.imshow(data)
        else:
            ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(label, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(str(save_path), dpi=140, bbox_inches="tight")
        print(f"[mask_pipeline] Diagnostic saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _estimate_wall_color(
    image_rgb: np.ndarray,
    mask:      np.ndarray,
    threshold: float = 0.40,   # lowered from 0.65: ADE20K mean ≈ 0.35, 0.65 found too few pixels
) -> tuple[int, int, int]:
    """
    Estimate dominant wall color from high-confidence mask pixels using K-means.
    Falls back to full-image median if the mask is too sparse.
    """
    wall_px = mask > threshold
    if wall_px.sum() < 200:
        wall_px = mask > 0.4
    if wall_px.sum() < 200:
        # Last resort: use centre crop (upper centre is usually wall)
        h, w   = mask.shape
        cy, cx = h // 2, w // 2
        crop   = image_rgb[max(0,cy-60):cy, max(0,cx-60):cx+60]
        return tuple(int(v) for v in np.median(crop.reshape(-1, 3), axis=0))

    samples = image_rgb[wall_px].astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _, labels, centers = cv2.kmeans(samples, 3, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    labels  = labels.ravel()
    counts  = np.bincount(labels, minlength=3)
    dom_rgb = centers[np.argmax(counts)].astype(int)
    return (int(dom_rgb[0]), int(dom_rgb[1]), int(dom_rgb[2]))
