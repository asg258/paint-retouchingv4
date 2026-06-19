"""
mask_process.py — Stage 3b of the wall recoloring pipeline.

Sits between SAM refinement (Stage 3) and recoloring (Stage 4).
Takes the SAM-refined mask and makes it cleaner, softer, and safer to blend.

WHY IS THE SAM MASK STILL IMPERFECT?
-------------------------------------
SAM is a boundary detector — it finds edges extremely well, but the mask it
produces is essentially binary (0 or 1). Real-world problems remain:

  1. Hard edges — a hard 0/1 boundary between wall and sofa will produce a
     visible seam in the final recolored image, especially at compression
     artifacts or slightly misaligned boundaries.

  2. Boundary pixel contamination — at the very edge of the wall, single
     pixels from the sofa, curtain, or shelf can be included in the mask
     (SAM is not perfect, and sub-pixel aliasing in the image means "wall"
     and "not-wall" regions are not cleanly separated at the pixel level).

  3. Salt-and-pepper noise — isolated single pixels or tiny islands of high
     probability in the middle of a clearly-non-wall region. These come from
     DeepLab's coarse mask propagating small errors through to SAM.

  4. Jagged contours — SAM traces actual object boundaries, which for textured
     surfaces (curtains, woven upholstery) look jagged rather than smooth.

This stage fixes all four issues without modifying the segmentation models.
"""

from __future__ import annotations

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------

# Gaussian blur — controls how wide the soft edge transition zone is.
# Larger sigma = wider feather = softer boundary but more color bleed risk.
# Larger kernel = more thorough smoothing (must be odd number).
# Good starting values for a 1080p-ish image: sigma=3, kernel=21.
GAUSSIAN_SIGMA: float      = 3.0
GAUSSIAN_KERNEL_SIZE: int  = 21   # must be odd

# Erosion — shrinks the mask inward by this many pixels on every edge.
# This creates a safety margin so boundary pixels (shared between wall and
# adjacent objects) are zeroed out before blending. A little goes a long way:
# 3–8 pixels is usually enough without visibly shrinking the wall coverage.
EROSION_SIZE: int = 5

# Noise threshold — mask values below this are forced to 0 after smoothing.
# Removes faint probability halos and isolated low-confidence pixels that
# Gaussian blur spreads outward from the boundary.
NOISE_THRESHOLD: float = 0.05


# ---------------------------------------------------------------------------
# Core processing function
# ---------------------------------------------------------------------------

def process_mask(
    mask: np.ndarray,
    gaussian_sigma: float     = GAUSSIAN_SIGMA,
    kernel_size: int          = GAUSSIAN_KERNEL_SIZE,
    erosion_size: int         = EROSION_SIZE,
    noise_threshold: float    = NOISE_THRESHOLD,
) -> np.ndarray:
    """
    Clean and soften the SAM-refined wall mask before blending.

    Pipeline:
        1. Erode  — shrink mask inward to remove boundary-pixel contamination
        2. Gaussian blur — spread hard edges into a smooth probability gradient
        3. Threshold — zero out residual noise from the blur spread

    Why in this order?
    Eroding first removes the contaminated edge pixels BEFORE the blur can
    spread their color into the interior. If you blur first, the eroded
    edge would include already-blurred values and you'd lose the soft gradient.

    Args:
        mask:             (H, W) float32 array, values in [0, 1].
                          Typically binary or near-binary from SAM.
        gaussian_sigma:   Standard deviation of the Gaussian kernel.
                          Controls the width of the soft feather zone.
        kernel_size:      Gaussian kernel size (must be a positive odd integer).
                          Larger = smoother transition but slower.
        erosion_size:     Radius in pixels to shrink the mask.
                          Removes edge contamination from adjacent objects.
        noise_threshold:  Values below this are set to 0 after blurring.
                          Removes faint halos that blur spreads outward.

    Returns:
        Processed mask: (H, W) float32, values in [0, 1].
        Softer, cleaner, and safer for blending than the raw SAM output.
    """
    # Work in float32 throughout — cv2 morphology/blur functions accept it.
    m = mask.astype(np.float32)

    # --- Step 1: Erosion ---
    # WHY: SAM traces the visual boundary of the wall, but that boundary is
    # shared with the adjacent object (sofa, curtain, shelf). The pixels right
    # ON the boundary technically belong to both regions. Eroding by a few
    # pixels pulls the mask edge inward so those contested pixels are excluded
    # from recoloring entirely — preventing the color from "bleeding" onto the
    # edge of the neighboring object.
    #
    # We erode the float mask directly: treating the mask as a grayscale image,
    # erosion replaces each pixel with the minimum value in the kernel window.
    # On a binary mask that contracts the boundary; on a soft mask it darkens
    # the edges, which is exactly what we want.
    if erosion_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,              # round kernel = isotropic erosion
            (erosion_size * 2 + 1, erosion_size * 2 + 1),
        )
        m = cv2.erode(m, kernel, iterations=1)

    # --- Step 2: Gaussian smoothing (feathering) ---
    # WHY: after erosion the mask still has a hard binary edge. A hard edge
    # in the mask means a hard visible seam in the recolored image — the wall
    # color stops abruptly rather than fading naturally into the adjacent area.
    # Gaussian blur replaces each pixel's value with a weighted average of its
    # neighbours, turning the 0/1 step into a smooth 0→1 ramp over sigma pixels.
    #
    # Formula: M_soft[x,y] = Σ G(dx,dy,sigma) · M[x+dx, y+dy]
    # where G is the 2D Gaussian kernel.
    #
    # The result: recolored pixels near the wall boundary receive a
    # proportionally smaller colour shift, fading the painted area into the
    # original so no seam is visible.
    k = _validated_kernel_size(kernel_size)
    m = cv2.GaussianBlur(m, (k, k), gaussian_sigma)

    # --- Step 3: Noise threshold cleanup ---
    # WHY: Gaussian blur spreads the mask outward slightly — pixels just
    # outside the wall edge receive a small non-zero value from the blur.
    # Most of these are genuinely unwanted (they land on furniture or the floor).
    # Zeroing anything below noise_threshold removes this faint halo without
    # touching the meaningful soft gradient at the real wall edge.
    if noise_threshold > 0:
        m[m < noise_threshold] = 0.0

    # Clamp to [0,1] in case of any float rounding drift.
    m = np.clip(m, 0.0, 1.0)

    # WHY KEEP A SOFT MASK?
    # The recoloring stage blends the new colour into the image weighted by
    # this mask value. A binary mask (0 or 1) means full colour or no colour
    # with a hard edge between them. A soft mask (0.0 → 1.0 gradient) means
    # the colour fades in gradually at the boundary — exactly how real paint
    # looks at the edge of a masked-off area. Keeping the mask soft here
    # enables photorealistic transitions in Stage 4 at zero extra cost.

    return m


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validated_kernel_size(k: int) -> int:
    """Ensure k is a positive odd integer (cv2 requirement for GaussianBlur)."""
    k = max(1, int(k))
    if k % 2 == 0:
        k += 1   # make it odd
    return k


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_mask_processing(
    raw_mask: np.ndarray,
    processed_mask: np.ndarray,
    image_rgb: np.ndarray | None = None,
    save_path: str | Path | None = None,
) -> None:
    """
    Side-by-side comparison of the raw SAM mask and the processed mask.

    Panels:
        1. Raw SAM mask         — heatmap, shows noise + hard edges
        2. Processed mask       — heatmap, shows smoothed result
        3. Difference           — where the processing removed or softened
        4. Overlay on image     — processed mask blended over original (optional)

    Args:
        raw_mask:        (H, W) float32 raw mask from Stage 3.
        processed_mask:  (H, W) float32 mask from process_mask().
        image_rgb:       Optional (H, W, 3) uint8 image for the overlay panel.
        save_path:       If provided, save the figure here.
    """
    n_panels = 4 if image_rgb is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    fig.suptitle("Stage 3b — Mask Processing: SAM output vs. processed", fontsize=13)

    # Panel 1: raw mask
    axes[0].imshow(raw_mask, cmap="hot", vmin=0, vmax=1)
    axes[0].set_title(f"Raw SAM mask\n(range [{raw_mask.min():.2f}, {raw_mask.max():.2f}])")
    axes[0].axis("off")

    # Panel 2: processed mask
    axes[1].imshow(processed_mask, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title(
        f"Processed mask\n"
        f"erode={EROSION_SIZE}px  sigma={GAUSSIAN_SIGMA}  "
        f"thresh={NOISE_THRESHOLD}"
    )
    axes[1].axis("off")

    # Panel 3: difference (raw - processed) — what was removed / softened
    diff = raw_mask.astype(np.float32) - processed_mask.astype(np.float32)
    im3 = axes[2].imshow(diff, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    axes[2].set_title("Difference (raw - processed)\nred=removed, blue=softened")
    axes[2].axis("off")
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    # Panel 4 (optional): processed mask overlaid on the original image
    if image_rgb is not None:
        import cv2 as _cv2
        heatmap_gray = (processed_mask * 255).astype(np.uint8)
        heatmap_bgr  = _cv2.applyColorMap(heatmap_gray, _cv2.COLORMAP_HOT)
        heatmap_rgb  = _cv2.cvtColor(heatmap_bgr, _cv2.COLOR_BGR2RGB)
        overlay = _cv2.addWeighted(image_rgb, 0.6, heatmap_rgb, 0.4, 0)
        axes[3].imshow(overlay)
        axes[3].set_title("Processed mask on image")
        axes[3].axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"[mask_process] Visualisation saved: {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Quick test — python mask_process.py <image_path>
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from preprocess import preprocess_image
    from segment    import load_model as load_deeplab, get_deeplab_mask
    from refine     import load_sam_model, refine_mask_with_sam

    if len(sys.argv) < 2:
        print("Usage:   python mask_process.py <image_path>")
        print("Example: python mask_process.py room.jpg")
        sys.exit(1)

    src = sys.argv[1]

    print("[main] Stage 1: preprocessing ...")
    preprocessed = preprocess_image(src)

    print("[main] Stage 2: DeepLab segmentation ...")
    deeplab, dl_device = load_deeplab()
    coarse = get_deeplab_mask(preprocessed, model=deeplab, device=dl_device)

    print("[main] Stage 3: SAM refinement ...")
    predictor, _ = load_sam_model()
    sam_mask = refine_mask_with_sam(preprocessed, coarse, predictor=predictor)

    print("[main] Stage 3b: mask processing ...")
    clean_mask = process_mask(sam_mask)

    print(f"[main] Raw mask   range: [{sam_mask.min():.3f}, {sam_mask.max():.3f}]  "
          f"mean={sam_mask.mean():.3f}")
    print(f"[main] Clean mask range: [{clean_mask.min():.3f}, {clean_mask.max():.3f}]  "
          f"mean={clean_mask.mean():.3f}")

    stem = Path(src).stem
    visualize_mask_processing(
        sam_mask, clean_mask,
        image_rgb=preprocessed,
        save_path=f"{stem}_mask_processing.png",
    )
