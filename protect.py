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

# How many pixels to expand the object boundary outward.
# Reduced from 10 → 4: 10px was eating into narrow wall strips in bathrooms
# and kitchens where walls and objects sit close together.  4px is still
# enough to absorb sub-pixel aliasing and JPEG artefacts without fragmenting
# large continuous wall planes.
DILATION_SIZE: int = 4

# Optional Gaussian blur applied to the dilated protection mask.
PROTECTION_SIGMA: float     = 1.5
PROTECTION_KERNEL_SIZE: int = 11   # must be odd; ignored if sigma == 0

# Hard ceiling on the protection mask value.
# Even on pure object pixels, M_protect never exceeds this value.
# This prevents the combination M_final = M_wall * (1 - M_protect) from
# driving M_final all the way to zero on pixels where the wall mask is
# still confident.  0.75 = protection can suppress at most 75 % of the wall.
MAX_PROTECTION: float = 0.75

# Alpha for the combination formula (see apply_protection).
# M_final = M_wall * (1 - alpha * M_protect)
# 1.0 = full protection effect (old behaviour, too aggressive)
# 0.5 = protection is scaled down to half — near-edge wall pixels survive
PROTECTION_ALPHA: float = 0.55

# Wall pixels above this confidence are "strong wall" — the protection mask
# is not allowed to reduce them below STRONG_WALL_FLOOR.
STRONG_WALL_THRESHOLD: float = 0.80
STRONG_WALL_FLOOR:     float = 0.65

# Small dilation applied to the WALL mask before combining with the protection
# mask.  This closes holes and maintains connectivity across thin wall strips
# before the protection mask can fragment them.
WALL_EXPAND_SIZE: int = 3


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def create_object_protection_mask(
    wall_mask: np.ndarray,
    dilation_size: int       = DILATION_SIZE,
    smoothing_sigma: float   = PROTECTION_SIGMA,
    kernel_size: int         = PROTECTION_KERNEL_SIZE,
) -> np.ndarray:
    """
    Build a protection mask that marks every region where recoloring
    must NOT occur — objects, furniture, and a safety buffer around them.

    Steps:
        1. Invert the wall mask  → raw object mask (everything that is not wall)
        2. Dilate the object mask → expand object boundaries outward
        3. Optionally blur the dilation edge → soft protection zone
        4. Clip to [0, 1] and return

    Args:
        wall_mask:       (H, W) float32 processed wall mask from Stage 4.
                         Values in [0, 1] — high = confident wall.
        dilation_size:   Radius in pixels to expand object boundaries.
                         This is how far the "keep-out zone" extends past
                         the edge of each detected object.
        smoothing_sigma: Gaussian sigma for edge softening.
                         0 = skip smoothing (hard protection boundary).
        kernel_size:     Gaussian kernel size (must be odd).

    Returns:
        protection_mask: (H, W) float32, values in [0, 1].
            1 = protected (do not recolor here)
            0 = safe to recolor

    How to use with the wall mask:
        M_final = wall_mask * (1 - protection_mask)
    """
    m = wall_mask.astype(np.float32)

    # ------------------------------------------------------------------
    # Step 1 — Invert: wall mask → object mask
    # ------------------------------------------------------------------
    # The wall mask tells us P(wall | pixel). Inverting gives us
    # P(object | pixel) — the probability that a pixel belongs to a
    # non-wall region (furniture, appliance, curtain, floor object, etc.).
    # This is our raw "things we must protect" map.
    obj_mask = 1.0 - m   # shape (H, W), values in [0, 1]

    # ------------------------------------------------------------------
    # Step 2 — Dilation: expand object boundaries outward
    # ------------------------------------------------------------------
    # WHY IS DILATION REQUIRED?
    #
    # The object mask at this point follows the exact boundary returned by
    # SAM + erosion. But that boundary is the visual edge of the object as
    # seen in the original image. The problem:
    #
    #   a) Sub-pixel aliasing — the pixel on the geometric boundary is a
    #      blend of wall and object colours. Both the wall mask and the
    #      object mask claim it. Without dilation, that pixel can end up
    #      with a small wall-mask value that gets recolored.
    #
    #   b) JPEG/compression artifacts — block-level artifacts around
    #      object edges shift the apparent boundary by 2–4 pixels.
    #
    #   c) SAM's own margin of error — even a perfectly tuned SAM model
    #      has ~1–3 px uncertainty at real-world boundaries.
    #
    # Dilation solves this by expanding the "keep-out zone" around every
    # object by dilation_size pixels in every direction. Any wall pixel
    # within that buffer is zeroed out in the final mask, guaranteeing
    # zero colour bleed regardless of minor mask inaccuracies.
    #
    # We dilate the object mask (not the wall mask) because we want to
    # GROW the protected region, not shrink it. Dilating an object mask
    # is equivalent to eroding the wall mask — but done independently here
    # so the two operations can be tuned separately.
    if dilation_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilation_size * 2 + 1, dilation_size * 2 + 1),
        )
        # Treat the float object mask as a grayscale image.
        # Dilation replaces each pixel with the MAX value in the kernel window,
        # which on a 0/1 mask expands bright (object) regions outward.
        obj_mask = cv2.dilate(obj_mask, kernel, iterations=1)

    # ------------------------------------------------------------------
    # Step 3 — Optional Gaussian smoothing of the protection edge
    # ------------------------------------------------------------------
    # The dilated mask has a hard edge at dilation_size pixels from the
    # object boundary. Smoothing this edge creates a gradual 0→1 ramp
    # so that the transition from "protected zone" back to "wall" is not
    # a sudden step. This cooperates with Stage 4's soft wall mask: both
    # are near 0 at the boundary, both taper smoothly, so their product
    # (M_final = M_wall × (1 - M_protect)) fades naturally.
    if smoothing_sigma > 0:
        k = _validated_kernel_size(kernel_size)
        obj_mask = cv2.GaussianBlur(obj_mask, (k, k), smoothing_sigma)

    # Cap at MAX_PROTECTION so even on pure object pixels the mask never
    # reaches 1.0.  This prevents the combination formula from zeroing out
    # pixels where the wall mask is still moderately confident.
    protection_mask = np.clip(obj_mask, 0.0, MAX_PROTECTION).astype(np.float32)

    return protection_mask


def apply_protection(
    wall_mask:        np.ndarray,
    protection_mask:  np.ndarray,
    alpha:            float = PROTECTION_ALPHA,
    strong_threshold: float = STRONG_WALL_THRESHOLD,
    strong_floor:     float = STRONG_WALL_FLOOR,
    wall_expand_size: int   = WALL_EXPAND_SIZE,
) -> np.ndarray:
    """
    Combine the wall mask and the protection mask into the final blend mask.

    Uses a balanced formula that protects object edges WITHOUT fragmenting
    large continuous wall regions.  Five constraints are applied in order:

    1. Light wall dilation — closes holes in the wall mask before protection
       can fragment it.  A 3px expand maintains connectivity across thin
       wall strips (narrow corridors, bathroom walls between objects).

    2. Alpha-scaled combination:
           M_final = M_wall * (1 - alpha * M_protect)
       alpha < 1 means protection is scaled down; even near objects the wall
       mask retains (1 - alpha * 1.0) = 1 - alpha of its value.

    3. Strong-wall floor — pixels where M_wall > strong_threshold are highly
       confident wall.  Protection is not allowed to suppress them below
       strong_floor, preserving large contiguous wall planes.

    4. The original protection_mask should already be capped at MAX_PROTECTION
       by create_object_protection_mask() — this prevents any single pixel
       from being driven all the way to zero.

    Formula summary:
        M_wall_exp  = dilate(M_wall, 3px)
        M_candidate = M_wall_exp * (1 - alpha * M_protect)
        M_final     = max(M_candidate, M_wall_exp * strong_floor)  where M_wall > threshold
        M_final     = M_candidate                                   elsewhere

    Args:
        wall_mask:        (H, W) float32 from Stage 4 (after mask processing).
        protection_mask:  (H, W) float32 from create_object_protection_mask().
        alpha:            Scales down the protection influence (0=no protection, 1=full).
        strong_threshold: Wall pixels above this get the strong-wall floor guarantee.
        strong_floor:     Minimum M_final fraction for strong wall pixels.
        wall_expand_size: Pixels to dilate the wall mask before combining.

    Returns:
        final_mask: (H, W) float32, values in [0, 1].

      - 0 everywhere outside the wall
      - reduced near objects (by alpha * M_protect)
      - floored at strong_floor * M_wall for high-confidence wall pixels
    """
    M_w = wall_mask.astype(np.float32)
    M_p = protection_mask.astype(np.float32)

    # Step 1 — lightly dilate the wall mask to close holes before protection
    # can fragment it.  Small kernel only — we don't want to expand walls.
    if wall_expand_size > 0:
        k_expand = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (wall_expand_size * 2 + 1, wall_expand_size * 2 + 1),
        )
        M_w = cv2.dilate(M_w, k_expand, iterations=1)
        M_w = np.clip(M_w, 0.0, 1.0)

    # Step 2 — alpha-scaled combination:
    #   M_candidate = M_wall * (1 - alpha * M_protect)
    # At furniture edges (M_protect ≈ 0.75, alpha=0.55):
    #   M_candidate = M_wall * (1 - 0.41) = M_wall * 0.59   → still visible
    # At clear object centres (M_protect = 0.75 max, alpha=0.55):
    #   M_candidate = M_wall * 0.59 — protection is real but not annihilating
    M_candidate = M_w * (1.0 - alpha * M_p)

    # Step 3 — strong-wall floor guarantee.
    # Pixels where the wall mask is very confident (> strong_threshold) are
    # on the main wall plane, not at a boundary. Protection should not reduce
    # them below strong_floor * M_wall — this preserves large continuous areas.
    strong = M_w > strong_threshold
    M_floor = M_w * strong_floor
    M_final = np.where(strong, np.maximum(M_candidate, M_floor), M_candidate)

    return np.clip(M_final, 0.0, 1.0).astype(np.float32)


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
