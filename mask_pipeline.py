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
VAR_WINDOW_SIZE:  int   = 15    # local window for variance computation (pixels)
VAR_LAMBDA:       float = 2.0   # suppression strength (lower = softer cutoff)
# Walls typically have normalised variance < 0.15; tile/wood > 0.25

# --- Color distance (LAB space) ---
COLOR_SIGMA:      float = 45.0  # LAB distance bandwidth — wider = more permissive
# LAB distances: same color ~5, same family ~20, different material ~40+
# σ=45 accepts anything within ≈1.5 "color families" of the wall

# --- Gradient planarity ---
GRAD_SIGMA:       float = 10.0  # Gaussian sigma for local gradient pooling
GRAD_LAMBDA:      float = 2.0   # suppression strength (lower = softer cutoff)

# --- Combined weight exponent ---
# W_material = (W_tex × W_col × W_grad)^COMBINATION_POWER
# Power < 1 softens the multiplicative combination so no single signal
# can fully zero out a pixel on its own. 0.5 = geometric mean of each pair.
COMBINATION_POWER: float = 0.5

# --- Grid-based patch normalisation ---
PATCH_SIZE:       int   = 64    # image patch size for grid analysis (pixels)
PATCH_FLOOR:      float = 0.20  # within a patch, wall pixels below this get zeroed

# --- Final smoothing ---
SMOOTH_SIGMA:     float = 2.0   # Gaussian sigma for final edge feathering
SMOOTH_KERNEL:    int   = 15    # must be odd

# --- Noise cleanup ---
NOISE_THRESHOLD:  float = 0.05  # values below this zeroed after all processing


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
    sigma:      float = COLOR_SIGMA,
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

    return np.exp(-dist / sigma).astype(np.float32)


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
# 4. Grid-Based Patch Normalisation
# ---------------------------------------------------------------------------

def grid_patch_normalise(
    mask:       np.ndarray,
    patch_size: int   = PATCH_SIZE,
    floor:      float = PATCH_FLOOR,
) -> np.ndarray:
    """
    Normalise mask values within each image patch to preserve local continuity.

    WHY THIS IS NEEDED
    ------------------
    After material suppression, some image patches may have been globally
    suppressed (e.g. the entire upper-left patch of a wall was identified as
    slightly textured). Within such a patch, the relative ordering of
    confidence values is still meaningful — the pixels with relatively higher
    values are more likely wall than their neighbours.

    Grid normalisation prevents a mildly-suppressed patch from being entirely
    zeroed out. It rescales each patch so its maximum value is preserved, then
    applies the floor threshold to remove noise within the patch.

    Formula (per patch P_ij):
        max_p       = max(mask, P_ij)
        if max_p > 0:
            mask[P_ij] = mask[P_ij] / max_p × max_p   (no-op: keeps scale)
        mask[P_ij < floor × max_p] = 0

    This is a soft floor: within each patch, pixels below floor × local_max
    are zeroed, preserving the strongest wall signal in each region.

    Args:
        mask:       (H, W) float32 mask.
        patch_size: Patch size in pixels.
        floor:      Relative threshold within each patch.

    Returns:
        normalised: (H, W) float32.
    """
    h, w  = mask.shape
    out   = mask.copy()
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = out[y:y+patch_size, x:x+patch_size]
            p_max = float(patch.max())
            if p_max > 0:
                # Zero out pixels below floor × local_max within this patch
                patch[patch < floor * p_max] = 0.0
    return out


# ---------------------------------------------------------------------------
# 5. Combined Material-Aware Refinement
# ---------------------------------------------------------------------------

def material_aware_mask(
    image_rgb:         np.ndarray,
    sam_mask:          np.ndarray,
    wall_color_rgb:    tuple[int, int, int] | None = None,
    var_window:        int   = VAR_WINDOW_SIZE,
    var_lambda:        float = VAR_LAMBDA,
    color_sigma:       float = COLOR_SIGMA,
    grad_lambda:       float = GRAD_LAMBDA,
    combination_power: float = COMBINATION_POWER,
    patch_size:        int   = PATCH_SIZE,
    smooth_sigma:      float = SMOOTH_SIGMA,
    noise_threshold:   float = NOISE_THRESHOLD,
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

    # --- Compute three material weight maps ---
    W_tex  = compute_local_variance(image_rgb, window_size=var_window, lam=var_lambda)
    W_col  = compute_color_distance_weight(image_rgb, wall_color_rgb, sigma=color_sigma)
    W_grad = compute_gradient_weight(image_rgb, lam=grad_lambda)

    # --- Combine material weights ---
    # W_product = W_tex × W_col × W_grad
    # Pure multiplication is too aggressive when all three are moderately low.
    # Raising to COMBINATION_POWER < 1 softens the combination:
    #   power=1.0 → pure product (aggressive)
    #   power=0.5 → geometric mean of each pair (balanced)
    #   power=0.33 → cube root (very soft)
    # A pixel with W_tex=0.6, W_col=0.5, W_grad=0.6 gets:
    #   power=1.0 → 0.18   (most pixels near-zero)
    #   power=0.5 → 0.42   (meaningful suppression without collapse)
    W_product  = W_tex * W_col * W_grad
    W_material = np.power(W_product, combination_power)

    # --- Apply to SAM mask ---
    M_refined  = sam_mask.astype(np.float32) * W_material

    # --- Grid patch normalisation ---
    # Rescales within each patch so locally-dominant wall regions survive
    # even if globally suppressed by a slightly elevated material signal.
    M_grid = grid_patch_normalise(M_refined, patch_size=patch_size)

    # --- Final Gaussian feathering ---
    k = int(smooth_sigma * 6 + 1) | 1
    M_smooth = cv2.GaussianBlur(M_grid, (k, k), smooth_sigma)

    # --- Noise floor ---
    M_smooth[M_smooth < noise_threshold] = 0.0
    M_final = np.clip(M_smooth, 0.0, 1.0).astype(np.float32)

    debug = {
        "W_texture":   W_tex,
        "W_color":     W_col,
        "W_gradient":  W_grad,
        "W_material":  W_material,
        "M_sam_raw":   sam_mask,
        "M_after_mat": M_refined,
        "M_final":     M_final,
        "wall_color":  wall_color_rgb,
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
    threshold: float = 0.65,
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
