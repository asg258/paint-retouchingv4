"""
color_detect.py — Stage 6 of the wall recoloring pipeline.

Extracts the dominant wall color from the masked wall region using
K-Means clustering.  The result is used by later stages to:

  - Stage 7 (adaptive recoloring) — measure how far the target color
    departs from the current wall color and scale the blend accordingly.
  - Stage 9 (color consistency metric) — compare the expected output
    against a reference measurement of what the wall should look like.

WHY WE NEED THIS STEP
----------------------
The wall in a photo is not one flat color.  It contains:
  - Lighting gradients (brighter near windows, darker in corners)
  - Soft shadows cast by furniture
  - Specular highlights on matte/semi-gloss surfaces
  - Compression artifacts

If you just average all wall pixels you get a gray muddy result.
K-Means finds the natural clusters in the pixel distribution and lets you
pick the LARGEST cluster, which corresponds to the true base color of the
paint — the dominant signal, with lighting and shadow pushed into
smaller satellite clusters.
"""

from __future__ import annotations

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------

# Number of color clusters to find inside the wall region.
# 3–5 is the sweet spot:
#   2 = often splits highlight vs shadow without finding the true base color
#   3 = base color + highlight + shadow  ← good default
#   5 = more granular, useful for multi-tone walls or heavy shadow gradients
K_CLUSTERS: int = 3

# Only pixels where M_final > this value are included in clustering.
# Using a high threshold (0.7) means we only sample pixels the pipeline
# is very confident belong to the wall — not edge pixels, not the protection
# buffer zone.  This keeps the cluster centers clean.
MASK_THRESHOLD: float = 0.7

# K-Means stopping criteria:
#   MAX_ITER  — stop after this many iterations even if not converged
#   EPSILON   — stop when cluster centroids move less than this (in pixel space)
KMEANS_MAX_ITER: int   = 100
KMEANS_EPSILON: float  = 0.2

# How many times to re-run K-Means with different random seeds and keep
# the result with the lowest inertia.  More attempts = more stable result
# but slower.  10 is a good balance.
KMEANS_ATTEMPTS: int = 10


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class WallColorResult:
    """
    All outputs from the K-Means wall color extraction.

    Attributes:
        dominant_color:  RGB triplet of the largest cluster centroid.
                         This is the "true base color" of the wall.
        dominant_lab:    Same color in LAB space (useful for Stage 7).
        clusters:        (K, 3) uint8 array of all centroid colors (RGB).
        counts:          (K,) int array — number of wall pixels in each cluster.
        fractions:       (K,) float array — fraction of wall pixels per cluster.
        dominant_index:  Which cluster index is the dominant one.
        n_wall_pixels:   Total number of wall pixels sampled.
    """
    dominant_color:  np.ndarray     # (3,) uint8 [R,G,B]
    dominant_lab:    np.ndarray     # (3,) float32 [L,A,B] in cv2 scale
    clusters:        np.ndarray     # (K, 3) uint8 [R,G,B]
    counts:          np.ndarray     # (K,) int
    fractions:       np.ndarray     # (K,) float
    dominant_index:  int
    n_wall_pixels:   int

    def as_dict(self) -> dict:
        """Return a plain-dict version suitable for JSON serialisation."""
        return {
            "dominant_color": self.dominant_color.tolist(),
            "dominant_lab":   self.dominant_lab.tolist(),
            "clusters":       self.clusters.tolist(),
            "counts":         self.counts.tolist(),
            "fractions":      self.fractions.tolist(),
            "dominant_index": self.dominant_index,
            "n_wall_pixels":  self.n_wall_pixels,
        }

    def __str__(self) -> str:
        r, g, b = self.dominant_color
        pct = self.fractions[self.dominant_index] * 100
        return (
            f"Dominant wall color: RGB({r},{g},{b})  "
            f"#{r:02X}{g:02X}{b:02X}  "
            f"({pct:.1f}% of sampled wall pixels)"
        )


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def extract_wall_color(
    image: np.ndarray,
    mask:  np.ndarray,
    k:                int   = K_CLUSTERS,
    mask_threshold:   float = MASK_THRESHOLD,
    kmeans_attempts:  int   = KMEANS_ATTEMPTS,
) -> WallColorResult:
    """
    Extract the dominant wall color by clustering high-confidence wall pixels.

    WHY RESTRICT TO WALL PIXELS USING THE MASK?
    We only want to analyse pixels that actually belong to the wall.
    If we ran K-Means on the full image, sofa browns, curtain beiges, and
    floor grays would all pull the centroids away from the true wall color.
    The mask (M_final from Stage 5) tells us which pixels are confident wall —
    threshold at 0.7 to keep only the most reliable samples.

    WHY K-MEANS FOR COLOR CLUSTERING?
    K-Means groups pixels by minimizing within-cluster variance:

        Objective: minimize Σ_{i=1}^{N} || x_i - μ_{c(i)} ||²

    Where:
        x_i           = the [R, G, B] color vector of pixel i
        μ_{c(i)}      = the centroid (mean) of the cluster pixel i belongs to
        || · ||²       = squared Euclidean distance in 3D color space
        c(i)           = the cluster assignment for pixel i

    Each term in the sum measures how "far" a pixel is from its cluster center.
    Minimizing the total means each centroid ends up at the geometric center of
    its pixel group — effectively the average color of that group.

    In a painted room, pixels naturally form a few tight clusters:
        - The largest  → true base wall color (even illumination across the wall)
        - Smaller ones → highlights (near window), shadows (corners, behind sofa)
        - Tiny ones    → artifacts, noise, wall-adjacent object pixels that leaked

    WHY IS THE DOMINANT CLUSTER THE TRUE BASE COLOR?
    Paint manufacturers design paint to look uniform across a wall under
    standard lighting. The largest cluster captures the majority of the wall
    area where lighting is close to normal. Highlights and shadows each affect
    a much smaller portion of the total wall surface. So the dominant cluster
    (by pixel count) is always the "as designed" wall color.

    Args:
        image:           (H, W, 3) uint8 RGB image.
        mask:            (H, W) float32 wall mask from Stage 5.
        k:               Number of clusters.
        mask_threshold:  Only pixels with mask > this are used.
        kmeans_attempts: Number of independent K-Means runs.

    Returns:
        WallColorResult — see class definition above.

    Raises:
        ValueError: if fewer than k pixels pass the mask threshold.
    """
    # ------------------------------------------------------------------
    # Step 1 — Extract masked wall pixels
    # ------------------------------------------------------------------
    # Boolean index: True wherever the mask is confident wall.
    # This collapses the (H,W) spatial grid into a 1D list of pixel vectors.
    wall_px_mask = mask > mask_threshold          # (H, W) bool
    wall_pixels  = image[wall_px_mask]            # (N, 3) uint8
    n            = len(wall_pixels)

    if n < k:
        raise ValueError(
            f"Only {n} pixels exceeded mask threshold {mask_threshold} — "
            f"need at least {k} for {k}-means clustering. "
            "Try lowering mask_threshold."
        )

    print(f"[color_detect] Clustering {n:,} wall pixels with k={k} ...")

    # ------------------------------------------------------------------
    # Step 2 — Run K-Means with cv2.kmeans
    # ------------------------------------------------------------------
    # cv2.kmeans expects a float32 array of shape (N, D).
    # We use KMEANS_PP_CENTERS (K-Means++ initialization) which picks
    # starting centroids that are spread out — much faster convergence
    # and more stable results than random initialization.
    samples = wall_pixels.astype(np.float32)  # (N, 3)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        KMEANS_MAX_ITER,
        KMEANS_EPSILON,
    )

    _compactness, labels, centers = cv2.kmeans(
        samples,
        k,
        None,
        criteria,
        kmeans_attempts,
        cv2.KMEANS_PP_CENTERS,   # K-Means++ for stable init
    )
    # labels:  (N, 1) int32  — which cluster each pixel belongs to
    # centers: (k, 3) float32 — RGB centroid of each cluster

    labels = labels.ravel()    # flatten to (N,)

    # ------------------------------------------------------------------
    # Step 3 — Count pixels per cluster
    # ------------------------------------------------------------------
    counts = np.bincount(labels, minlength=k)   # (k,) — pixels per cluster
    fractions = counts / counts.sum()           # fraction of wall area

    # The dominant cluster is the one with the most pixels.
    # This is the true base wall color (see docstring above).
    dominant_idx = int(np.argmax(counts))
    dominant_rgb = centers[dominant_idx].astype(np.uint8)   # (3,) uint8

    # ------------------------------------------------------------------
    # Step 4 — Convert dominant color to LAB
    # ------------------------------------------------------------------
    # Stage 7 (adaptive recoloring) works in LAB space, so pre-compute
    # the LAB representation of the detected wall color here to avoid
    # repeating the conversion later.
    bgr_pixel = np.array(
        [[[int(dominant_rgb[2]), int(dominant_rgb[1]), int(dominant_rgb[0])]]],
        dtype=np.uint8,
    )
    lab_pixel  = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2LAB).astype(np.float32)
    dominant_lab = lab_pixel[0, 0]   # (3,) float32 [L, A, B]

    # Sort clusters from largest to smallest for readability.
    sort_order = np.argsort(counts)[::-1]
    sorted_centers    = centers[sort_order].astype(np.uint8)
    sorted_counts     = counts[sort_order]
    sorted_fractions  = fractions[sort_order]
    # The dominant cluster is now always at index 0.
    dominant_idx_sorted = 0

    result = WallColorResult(
        dominant_color  = sorted_centers[0],
        dominant_lab    = dominant_lab,
        clusters        = sorted_centers,
        counts          = sorted_counts,
        fractions       = sorted_fractions,
        dominant_index  = dominant_idx_sorted,
        n_wall_pixels   = n,
    )

    print(f"[color_detect] {result}")
    return result


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_clusters(
    image_rgb:   np.ndarray,
    mask:        np.ndarray,
    result:      WallColorResult,
    save_path:   str | Path | None = None,
) -> None:
    """
    Three-panel figure:
        1. Original image with wall region highlighted
        2. Color swatches for each cluster, largest first
        3. Dominant wall color swatch with RGB + hex label

    Args:
        image_rgb:  (H, W, 3) uint8 RGB original image.
        mask:       (H, W) float32 wall mask.
        result:     WallColorResult from extract_wall_color().
        save_path:  If set, saves figure here.
    """
    k = len(result.clusters)

    fig = plt.figure(figsize=(18, 5))
    fig.suptitle("Stage 6 — Wall Color Detection (K-Means clustering)", fontsize=13)

    # --- Panel 1: image with wall overlay ---
    ax1 = fig.add_subplot(1, 3, 1)
    overlay = image_rgb.copy()
    wall_highlight = np.zeros_like(image_rgb)
    wall_highlight[:, :, 1] = 80   # green tint on wall pixels
    alpha = np.clip(mask, 0, 1)[:, :, np.newaxis]
    overlay = (overlay.astype(float) + wall_highlight * alpha).clip(0, 255).astype(np.uint8)
    ax1.imshow(overlay)
    ax1.set_title(f"Wall region\n({result.n_wall_pixels:,} pixels sampled)")
    ax1.axis("off")

    # --- Panel 2: cluster swatches ---
    ax2 = fig.add_subplot(1, 3, 2)
    swatch_h = 60
    swatch_w = 200
    total_h  = k * swatch_h
    swatches = np.zeros((total_h, swatch_w, 3), dtype=np.uint8)

    for i, (color, frac) in enumerate(zip(result.clusters, result.fractions)):
        y0, y1 = i * swatch_h, (i + 1) * swatch_h
        swatches[y0:y1] = color
        r, g, b = color
        label = f"#{r:02X}{g:02X}{b:02X}  {frac*100:.1f}%"
        if i == 0:
            label += "  <- dominant"
        # Choose text color based on swatch brightness for readability
        brightness = 0.299 * r + 0.587 * g + 0.114 * b
        txt_color  = "black" if brightness > 128 else "white"
        ax2.text(
            10, y0 + swatch_h // 2, label,
            va="center", fontsize=9, color=txt_color,
            fontweight="bold" if i == 0 else "normal",
        )

    ax2.imshow(swatches)
    ax2.set_title(f"All {k} clusters\n(sorted by size, largest first)")
    ax2.axis("off")

    # --- Panel 3: dominant color ---
    ax3 = fig.add_subplot(1, 3, 3)
    r, g, b  = result.dominant_color
    big_swatch = np.full((200, 300, 3), [r, g, b], dtype=np.uint8)
    ax3.imshow(big_swatch)
    l, a_ch, b_ch = result.dominant_lab
    ax3.set_title(
        f"Dominant wall color\n"
        f"RGB({r}, {g}, {b})   #{r:02X}{g:02X}{b:02X}\n"
        f"LAB({l:.0f}, {a_ch:.0f}, {b_ch:.0f})  "
        f"{result.fractions[0]*100:.1f}% of wall"
    )
    ax3.axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"[color_detect] Visualisation saved: {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Quick test — python color_detect.py <image_path>
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from preprocess   import preprocess_image
    from segment      import load_model as load_deeplab, get_deeplab_mask
    from refine       import load_sam_model, refine_mask_with_sam
    from mask_process import process_mask
    from protect      import create_object_protection_mask, apply_protection

    if len(sys.argv) < 2:
        print("Usage:   python color_detect.py <image_path>")
        print("Example: python color_detect.py room.jpg")
        sys.exit(1)

    src  = sys.argv[1]
    stem = Path(src).stem

    print("[main] Stage 1: preprocessing ...")
    preprocessed = preprocess_image(src)

    print("[main] Stage 2: DeepLab segmentation ...")
    deeplab, dl_device = load_deeplab()
    coarse = get_deeplab_mask(preprocessed, model=deeplab, device=dl_device)

    print("[main] Stage 3: SAM refinement ...")
    predictor, _ = load_sam_model()
    sam_mask = refine_mask_with_sam(preprocessed, coarse, predictor=predictor)

    print("[main] Stage 4: mask processing ...")
    clean_mask = process_mask(sam_mask)

    print("[main] Stage 5: object protection ...")
    protection = create_object_protection_mask(clean_mask)
    final_mask = apply_protection(clean_mask, protection)

    print("[main] Stage 6: wall color detection ...")
    result = extract_wall_color(preprocessed, final_mask)

    print(f"\nWall color summary:")
    print(f"  {result}")
    print(f"  Sampled {result.n_wall_pixels:,} pixels")
    print(f"  All clusters:")
    for i, (c, cnt, frac) in enumerate(zip(result.clusters, result.counts, result.fractions)):
        r, g, b = c
        marker = " <- dominant" if i == result.dominant_index else ""
        print(f"    [{i}] RGB({r:3d},{g:3d},{b:3d})  #{r:02X}{g:02X}{b:02X}  "
              f"{cnt:,} px  {frac*100:.1f}%{marker}")

    visualize_clusters(
        preprocessed, final_mask, result,
        save_path=f"{stem}_wall_colors.png",
    )
