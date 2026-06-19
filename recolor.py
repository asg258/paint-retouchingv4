"""
recolor.py — Stage 4 of the wall recoloring pipeline.

Takes the refined wall mask from Stage 3 and repaints the wall region
to a target paint color, while keeping lighting, shadows, and sheen intact.

HOW IT WORKS (the short version):
  We work in LAB color space.  LAB separates:
    L = how light/dark the pixel is (lightness)
    A = how green or red it is
    B = how blue or yellow it is

  To repaint a wall:
    - Keep L exactly as-is → shadows, highlights, and texture are preserved
    - Replace A and B with the target color's A and B, blended by mask strength
    → the wall changes color but still looks like it's lit by the same light

  This is why professional color visualizer tools (including Sherwin-Williams'
  own ColorSnap) use LAB-based recoloring.
"""

from __future__ import annotations

import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")   # save figures without needing a display
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------------------------
# Target paint color
# ---------------------------------------------------------------------------

# SW 6244 "Knitting Needles" — warm gray/taupe.
# Source: Sherwin-Williams ColorSnap.
# NOTE: these values are approximate. To use the exact values, visit:
#   https://www.sherwin-williams.com and search "SW 6244", then read the hex.
# Update TARGET_RGB below with the correct values if needed.
TARGET_COLOR_NAME: str = "SW 6244 Knitting Needles"
TARGET_RGB: tuple[int, int, int] = (158, 148, 132)   # (R, G, B), approx

# How strongly to apply the new color.
# 1.0 = full replacement (the wall becomes exactly the target color)
# 0.8 = 80% new color, 20% original (leaves a little of the original through)
# Useful range: 0.7–1.0
COLOR_BLEND_STRENGTH: float = 0.90

# Pixels with mask value below this are not recolored at all.
# This protects furniture and objects that crept into the mask.
MASK_THRESHOLD: float = 0.25


# ---------------------------------------------------------------------------
# Core recoloring
# ---------------------------------------------------------------------------

def recolor_walls(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    target_rgb: tuple[int, int, int] = TARGET_RGB,
    blend_strength: float = COLOR_BLEND_STRENGTH,
    mask_threshold: float = MASK_THRESHOLD,
) -> np.ndarray:
    """
    Repaint wall pixels to the target color while preserving lighting.

    The approach:
      1. Convert image and target color to LAB.
      2. For every pixel, compute how much to shift A and B toward the target.
         The shift weight = mask_value × blend_strength.
         Pixels deep in the wall get the full shift; edge pixels get a
         proportionally smaller shift, creating a smooth feathered boundary.
      3. L channel is untouched — shadows and highlights are preserved.
      4. Convert back to RGB.

    Args:
        image_rgb:      (H, W, 3) uint8 RGB image (Stage 1 output).
        mask:           (H, W) float32 wall probability mask (Stage 3 output).
        target_rgb:     (R, G, B) target paint color as uint8 values.
        blend_strength: How fully to apply the new color (0–1).
        mask_threshold: Mask values below this are left unchanged.

    Returns:
        recolored: (H, W, 3) uint8 RGB image with walls repainted.
    """
    # --- Convert image to LAB ---
    # cv2 expects BGR input for COLOR_BGR2LAB, so convert accordingly.
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    lab_image  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # --- Convert target paint color to LAB ---
    # Build a 1×1 pixel in the target color, convert it to LAB,
    # and read off the A and B values we want to push the wall toward.
    target_pixel_bgr = np.array(
        [[[target_rgb[2], target_rgb[1], target_rgb[0]]]], dtype=np.uint8
    )
    target_lab = cv2.cvtColor(target_pixel_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_a = float(target_lab[0, 0, 1])
    target_b = float(target_lab[0, 0, 2])

    print(f"[recolor] Target: {TARGET_COLOR_NAME}  RGB{target_rgb}")
    print(f"[recolor] Target LAB — A: {target_a:.1f}  B: {target_b:.1f}")

    # --- Build a per-pixel blend weight ---
    # weight = 0 outside the wall (mask below threshold)
    # weight smoothly rises with mask confidence, capped at blend_strength
    weight = np.clip(mask, 0.0, 1.0)
    weight[weight < mask_threshold] = 0.0
    weight = weight * blend_strength    # (H, W), values in [0, blend_strength]

    # Expand to (H, W, 1) for broadcasting across A and B channels.
    w = weight[:, :, np.newaxis]

    # --- Shift A and B channels toward target, leave L alone ---
    lab_shifted = lab_image.copy()
    lab_shifted[:, :, 1] = lab_image[:, :, 1] * (1.0 - weight) + target_a * weight
    lab_shifted[:, :, 2] = lab_image[:, :, 2] * (1.0 - weight) + target_b * weight

    # Clip back to valid uint8 range before converting.
    lab_shifted = np.clip(lab_shifted, 0, 255).astype(np.uint8)

    # --- Convert back to RGB ---
    recolored_bgr = cv2.cvtColor(lab_shifted, cv2.COLOR_LAB2BGR)
    recolored_rgb = cv2.cvtColor(recolored_bgr, cv2.COLOR_BGR2RGB)

    return recolored_rgb


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def save_comparison(
    original_rgb: np.ndarray,
    recolored_rgb: np.ndarray,
    mask: np.ndarray,
    save_path: str | Path,
    color_name: str = TARGET_COLOR_NAME,
) -> None:
    """
    Save a 3-panel comparison: original | wall mask | recolored result.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"Wall Recoloring — {color_name}", fontsize=14)

    axes[0].imshow(original_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Wall mask (refined)")
    axes[1].axis("off")

    axes[2].imshow(recolored_rgb)
    axes[2].set_title(f"Recolored: {color_name}")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[recolor] Comparison saved: {save_path}")


# ---------------------------------------------------------------------------
# Full pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    image_path: str | Path,
    use_sam: bool = True,
) -> None:
    """
    Run all 4 stages end-to-end and save the recolored image.

    Stage 1 → preprocess
    Stage 2 → DeepLab mask
    Stage 3 → SAM refinement (optional, skip with use_sam=False)
    Stage 4 → LAB recoloring

    Args:
        image_path: Path to the source room photo.
        use_sam:    Whether to run SAM refinement.  Set False to skip Stage 3
                    (faster, slightly less precise edges).
    """
    from preprocess import preprocess_image
    from segment   import load_model as load_deeplab, get_deeplab_mask

    image_path = Path(image_path)
    stem       = image_path.stem

    # ── Stage 1: LAB preprocessing ─────────────────────────────────────────
    print("\n--- Stage 1: LAB preprocessing ---")
    preprocessed = preprocess_image(str(image_path))
    print(f"[stage1] Shape: {preprocessed.shape}")

    # Stage 2: DeepLab coarse mask
    print("\n--- Stage 2: DeepLab segmentation ---")
    deeplab, dl_device = load_deeplab()
    coarse_mask = get_deeplab_mask(preprocessed, model=deeplab, device=dl_device)
    print(f"[stage2] Mask range: [{coarse_mask.min():.3f}, {coarse_mask.max():.3f}]")

    # Stage 3: SAM boundary refinement (optional)
    if use_sam:
        print("\n--- Stage 3: SAM boundary refinement ---")
        try:
            from refine import load_sam_model, refine_mask_with_sam
            predictor, _ = load_sam_model()
            refined_mask = refine_mask_with_sam(
                preprocessed, coarse_mask, predictor=predictor
            )
            print(f"[stage3] Refined mask range: [{refined_mask.min():.3f}, {refined_mask.max():.3f}]")
        except FileNotFoundError as e:
            print(f"[stage3] WARNING: {e}")
            print("[stage3] Falling back to coarse DeepLab mask.")
            refined_mask = coarse_mask
    else:
        print("\n--- Stage 3: Skipped (use_sam=False) ---")
        refined_mask = coarse_mask

    # Stage 4: Recolor
    print("\n--- Stage 4: Recoloring ---")
    recolored = recolor_walls(preprocessed, refined_mask)

    # Save outputs
    out_recolored   = image_path.parent / f"{stem}_recolored.jpg"
    out_comparison  = image_path.parent / f"{stem}_comparison.jpg"

    # Save the recolored image on its own (full resolution, no borders).
    cv2.imwrite(
        str(out_recolored),
        cv2.cvtColor(recolored, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    print(f"[stage4] Recolored image saved: {out_recolored}")

    # Save the 3-panel comparison figure.
    save_comparison(preprocessed, recolored, refined_mask, out_comparison)

    print(f"\nDone. Output files:")
    print(f"   Recolored : {out_recolored}")
    print(f"   Comparison: {out_comparison}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:   python recolor.py <image_path> [--no-sam]")
        print("Example: python recolor.py room.jpg")
        print("         python recolor.py room.jpg --no-sam   # skip SAM, faster")
        sys.exit(1)

    src   = sys.argv[1]
    no_sam = "--no-sam" in sys.argv

    run_pipeline(src, use_sam=not no_sam)
