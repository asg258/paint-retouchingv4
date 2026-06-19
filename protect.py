"""
protect.py — Stage 5 of the wall recoloring pipeline.

Builds an object protection mask that prevents the recoloring step from
touching furniture, appliances, and any other non-wall surface — even if
the wall mask from Stage 4 has leaked slightly past the true boundary.

WHY IS STAGE 4 (EROSION + GAUSSIAN) NOT ENOUGH?
-------------------------------------------------
Stage 4 attacks the problem from the WALL side:
  - Erosion shrinks the wall mask inward by a few pixels.
  - Gaussian blur tapers the edge to a soft gradient.

Both operations help, but they leave a gap: what if the wall mask is
accurate at the boundary but the *recoloring blend weight* at the edge
is still 0.05–0.20?  That small weight is enough to visibly tint a
sofa cushion or curtain hem in a vibrant colour like Amethyst Ice.

Stage 5 attacks the problem from the OBJECT side:
  - We build a protection zone around every non-wall region.
  - Any pixel inside that zone is forcibly zeroed in the final mask.
  - Even if a later pipeline step accidentally expands the wall mask,
    the protection zone absorbs the error.

Think of it as painter's tape: erosion tidies up the brush, but tape
guarantees zero bleed regardless of the brush quality.

HOW IT INTEGRATES WITH THE BLENDING FORMULA
--------------------------------------------
Stage 8 (blending) will compute:

    M_final(x,y) = M_wall(x,y) * (1 - M_protect(x,y))

    Output(x,y)  = M_final(x,y) * Recolored(x,y)
                 + (1 - M_final(x,y)) * Original(x,y)

Where:
  M_wall     = soft wall mask from Stage 4            (values 0–1)
  M_protect  = dilated object protection mask         (values 0–1)
  M_final    = wall mask with all protected zones zeroed out

Effect:
  - Pixels deep in the wall:       M_final ≈ 1 → fully recolored
  - Pixels near an object edge:    M_final ≈ 0 → protected, stays original
  - Pixels deep in furniture:      M_final  = 0 → untouched
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

# Erosion radius applied to the wall mask in apply_protection().
# 3px is enough to pull back contested boundary pixels without
# visibly shrinking large continuous wall planes.
EROSION_SIZE: int = 3


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def create_object_protection_mask(
    wall_mask: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """
    Stub — dilation-based protection removed.

    The previous multi-step dilation + alpha + luminance approach was
    compounding with erosion in Stage 3b and the post-SAM continuity dilation
    to fragment large wall planes into islands. All of that logic is gone.

    The protection function is now `apply_protection`, which applies a
    single lightweight erosion directly to the wall mask. This stub exists
    only for API compatibility — its return value (a zero array) is ignored
    by apply_protection.

    Args:
        wall_mask: (H, W) float32 — kept for signature compatibility.

    Returns:
        zeros: (H, W) float32 — ignored by apply_protection.
    """
    return np.zeros_like(wall_mask, dtype=np.float32)


def apply_protection(
    wall_mask:       np.ndarray,
    protection_mask: np.ndarray | None = None,
    erosion_size:    int = EROSION_SIZE,
) -> np.ndarray:
    """
    Produce M_final from M_wall using a single lightweight erosion.

    All previous complexity (dilation, alpha scaling, strong-wall floors,
    luminance boosts) is removed. The only operation is:

        M_final = erode(M_wall, ellipse kernel of radius erosion_size)

    A 3px erosion pulls boundary pixels inward just enough to absorb
    sub-pixel aliasing and SAM edge uncertainty without eating into the
    interior of large wall planes. The protection_mask argument is accepted
    for API compatibility but ignored.

    Args:
        wall_mask:       (H, W) float32 from Stage 3b.
        protection_mask: Ignored — kept for call-site compatibility.
        erosion_size:    Erosion kernel radius in pixels (default 3).

    Returns:
        M_final: (H, W) float32, values in [0, 1].
    """
    M = wall_mask.astype(np.float32)
    if erosion_size > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (erosion_size * 2 + 1, erosion_size * 2 + 1),
        )
        M = cv2.erode(M, k, iterations=1)
    return np.clip(M, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validated_kernel_size(k: int) -> int:
    """Return k as a positive odd integer (cv2 GaussianBlur requirement)."""
    k = max(1, int(k))
    return k if k % 2 == 1 else k + 1


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_protection(
    wall_mask: np.ndarray,
    protection_mask: np.ndarray,
    final_mask: np.ndarray,
    image_rgb: np.ndarray | None = None,
    save_path: str | Path | None = None,
) -> None:
    """
    Four-panel (or five-panel) figure showing the protection pipeline.

    Panels:
        1. Processed wall mask (Stage 4 output)
        2. Object mask  (inverted wall, before dilation)
        3. Dilated protection mask
        4. Final combined mask  (wall × (1 - protection))
        5. Final mask overlaid on original image  (if image_rgb provided)

    Args:
        wall_mask:        (H, W) float32 from Stage 4.
        protection_mask:  (H, W) float32 from create_object_protection_mask().
        final_mask:       (H, W) float32 from apply_protection().
        image_rgb:        Optional (H, W, 3) uint8 for overlay panel.
        save_path:        If set, figure saved here.
    """
    obj_mask_raw = 1.0 - wall_mask   # un-dilated object mask for display

    n = 5 if image_rgb is not None else 4
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    fig.suptitle("Stage 5 — Object Protection Mask", fontsize=13)

    _show(axes[0], wall_mask,      "Wall mask\n(Stage 4 output)",  "Greens")
    _show(axes[1], obj_mask_raw,   "Object mask\n(1 - wall)",      "Reds")
    _show(axes[2], protection_mask,"Protection mask\n(dilated + blurred)", "Oranges")
    _show(axes[3], final_mask,     "Final mask\nwall * (1 - protect)", "Blues")

    if image_rgb is not None:
        overlay = _make_overlay(image_rgb, final_mask)
        axes[4].imshow(overlay)
        axes[4].set_title("Final mask on image")
        axes[4].axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"[protect] Visualisation saved: {save_path}")

    plt.show()


def _show(ax, mask: np.ndarray, title: str, cmap: str) -> None:
    im = ax.imshow(mask, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(f"{title}\nrange [{mask.min():.2f}, {mask.max():.2f}]")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _make_overlay(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Tint wall region green and protected region red over the original image."""
    overlay = image_rgb.copy().astype(np.float32)
    # Green tint where wall will be recolored
    overlay[:, :, 1] = np.clip(overlay[:, :, 1] + mask * 60, 0, 255)
    # Red tint where protection zone blocks recoloring
    protected = np.clip(1.0 - mask - (1.0 - mask - (1.0 - (1.0 - mask))), 0, 1)
    protect_zone = np.clip((1.0 - mask) * 0.3, 0, 1)
    overlay[:, :, 0] = np.clip(overlay[:, :, 0] + protect_zone * 80, 0, 255)
    return overlay.astype(np.uint8)


# ---------------------------------------------------------------------------
# Quick test — python protect.py <image_path>
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from preprocess  import preprocess_image
    from segment     import load_model as load_deeplab, get_deeplab_mask
    from refine      import load_sam_model, refine_mask_with_sam
    from mask_process import process_mask

    if len(sys.argv) < 2:
        print("Usage:   python protect.py <image_path>")
        print("Example: python protect.py room.jpg")
        sys.exit(1)

    src = sys.argv[1]
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

    print("[main] Stage 5: object protection mask ...")
    protection = create_object_protection_mask(clean_mask)
    final      = apply_protection(clean_mask, protection)

    print(f"[main] Wall mask      — mean: {clean_mask.mean():.3f}")
    print(f"[main] Protection     — mean: {protection.mean():.3f}  "
          f"(covers {protection.mean()*100:.1f}% of image)")
    print(f"[main] Final mask     — mean: {final.mean():.3f}")

    visualize_protection(
        clean_mask, protection, final,
        image_rgb=preprocessed,
        save_path=f"{stem}_protection.png",
    )
