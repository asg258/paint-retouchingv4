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
from colors import get_color, search_colors, ColorEntry


# ---------------------------------------------------------------------------
# Default recoloring parameters
# ---------------------------------------------------------------------------

# Full A/B replacement — set to 1.0 so the wall actually becomes the target
# color rather than a faint tint. Lower only if you want a subtle wash effect.
COLOR_BLEND_STRENGTH: float = 1.0

# Pixels with mask value below this threshold are not recolored at all.
# This protects furniture and other objects that crept into the wall mask.
MASK_THRESHOLD: float = 0.25

# Minimum LAB A or B shift from neutral (128 in cv2 encoding).
# Light neutral walls often have A and B values very close to 128.
# If the target color is also near-neutral, the visual change is imperceptible.
# This parameter guarantees a minimum push away from gray so the color is always
# visible — without forcing a garish result. 8 LAB units is subtle but noticeable.
MIN_COLOR_SHIFT: float = 8.0

# After recoloring, scale A and B in the wall region slightly away from neutral.
# 1.0 = no boost.  1.15 = 15% more vivid.  Good range: 1.0–1.20.
# This helps test colours look "as painted" rather than washed-out, which
# matches how paint looks on a fresh dry coat vs. an image screenshot.
SATURATION_BOOST: float = 1.12


# ---------------------------------------------------------------------------
# Core recoloring
# ---------------------------------------------------------------------------

def recolor_walls(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    color: ColorEntry,
    blend_strength: float = COLOR_BLEND_STRENGTH,
    mask_threshold: float = MASK_THRESHOLD,
    min_color_shift: float = MIN_COLOR_SHIFT,
    saturation_boost: float = SATURATION_BOOST,
) -> np.ndarray:
    """
    Repaint wall pixels to the target paint color while preserving lighting.

    The approach:
      1. Convert image and target color to LAB.
      2. Enforce a minimum A/B shift from neutral so the color is always
         visible even on very light or near-neutral walls.
      3. For every pixel, blend A and B toward the target, weighted by the
         mask probability times blend_strength (smooth feathered edge).
      4. L channel is untouched — shadows and highlights are preserved.
      5. Apply a mild saturation boost inside the wall region so the result
         looks like a freshly-painted coat rather than a faint overlay.
      6. Convert back to RGB.

    Args:
        image_rgb:        (H, W, 3) uint8 RGB image (Stage 1 output).
        mask:             (H, W) float32 wall probability mask (Stage 3 output).
        color:            ColorEntry from the Valspar database (colors.py).
        blend_strength:   How fully to replace A/B channels (0-1). 1.0 = full.
        mask_threshold:   Mask values below this are not recolored.
        min_color_shift:  Minimum LAB A or B deviation from neutral (128).
                          Ensures the color change is always visible.
        saturation_boost: Scale factor applied to A/B in the wall region
                          after blending. > 1.0 makes the color more vivid.

    Returns:
        recolored: (H, W, 3) uint8 RGB image with walls repainted.
    """
    target_rgb = color.rgb

    # Convert image to float LAB for precise arithmetic.
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    lab        = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Convert the target paint color to LAB.
    target_bgr = np.array([[[target_rgb[2], target_rgb[1], target_rgb[0]]]], dtype=np.uint8)
    target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_a   = float(target_lab[0, 0, 1])
    target_b   = float(target_lab[0, 0, 2])

    # --- Enforce minimum color shift ---
    # In cv2 LAB encoding, neutral gray sits at A=128, B=128.
    # Very light or near-neutral walls can have A/B close to 128 already,
    # so a near-neutral target color would produce an invisible change.
    # We push the target A and B at least min_color_shift units away from 128
    # so there is always a perceptible difference on-screen.
    neutral = 128.0
    def _enforce_min_shift(val: float) -> float:
        offset = val - neutral
        if abs(offset) < min_color_shift:
            # Keep the direction (sign) but guarantee the minimum magnitude.
            offset = min_color_shift if offset >= 0 else -min_color_shift
        return neutral + offset

    target_a = _enforce_min_shift(target_a)
    target_b = _enforce_min_shift(target_b)

    print(f"[recolor] Color : {color.code}  '{color.name}'  RGB{target_rgb}  #{color.hex}")
    print(f"[recolor] Family: {color.family}  LRV: {color.lrv}")
    print(f"[recolor] Target LAB (after min-shift) - A: {target_a:.1f}  B: {target_b:.1f}")

    # --- Build per-pixel blend weight ---
    # weight = 0 outside the wall, rises smoothly with mask confidence.
    weight = np.clip(mask, 0.0, 1.0)
    weight[weight < mask_threshold] = 0.0
    weight = weight * blend_strength    # shape (H, W), values in [0, blend_strength]

    # --- Replace A and B, leave L untouched ---
    lab_out = lab.copy()
    lab_out[:, :, 1] = lab[:, :, 1] * (1.0 - weight) + target_a * weight
    lab_out[:, :, 2] = lab[:, :, 2] * (1.0 - weight) + target_b * weight

    # --- Saturation boost inside the wall region ---
    # Scale A and B away from neutral (128) by saturation_boost in the wall
    # region only. This makes the color read as a true painted coat rather
    # than a translucent wash — particularly important for light target colors.
    if saturation_boost != 1.0:
        wall_pixels = weight > 0   # boolean mask (H, W)
        for ch in [1, 2]:
            ch_f  = lab_out[:, :, ch]
            boosted = (ch_f - neutral) * saturation_boost + neutral
            lab_out[:, :, ch] = np.where(wall_pixels, boosted, ch_f)

    lab_out = np.clip(lab_out, 0, 255).astype(np.uint8)

    recolored_bgr = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
    recolored_rgb = cv2.cvtColor(recolored_bgr, cv2.COLOR_BGR2RGB)

    return recolored_rgb


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def save_comparison(
    original_rgb: np.ndarray,
    recolored_rgb: np.ndarray,
    mask: np.ndarray,
    color: ColorEntry,
    save_path: str | Path,
) -> None:
    """
    Save a 3-panel comparison: original | wall mask | recolored result.
    Includes a color swatch showing the exact target paint color.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    label = f"{color.code}  {color.name}  #{color.hex}  LRV {color.lrv}"
    fig.suptitle(f"Wall Recoloring — {label}", fontsize=12)

    axes[0].imshow(original_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Wall mask (refined)")
    axes[1].axis("off")

    # Draw a small color swatch in the corner of the recolored panel
    # so you can verify the target color at a glance.
    r, g, b = color.rgb
    swatch = np.full((40, 120, 3), [r, g, b], dtype=np.uint8)
    axes[2].imshow(recolored_rgb)
    axes[2].set_title(f"Recolored: {color.name}")
    # Embed swatch as inset axes (top-left corner)
    inset = axes[2].inset_axes([0.01, 0.88, 0.18, 0.10])
    inset.imshow(swatch)
    inset.axis("off")
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
    color_query: str = "4002-9A",
    use_sam: bool = True,
) -> None:
    """
    Run all 4 stages end-to-end and save the recolored image.

    Stage 1 - preprocess
    Stage 2 - DeepLab mask
    Stage 3 - SAM refinement (optional, skip with use_sam=False)
    Stage 4 - LAB recoloring with the chosen Valspar color

    Args:
        image_path:   Path to the source room photo.
        color_query:  Color name or code from the Valspar database.
                      e.g. "Lucy Blue", "5001-5C", "#81A9B2"
                      Fuzzy matching is used if no exact match is found.
        use_sam:      Whether to run SAM refinement. False = faster but
                      slightly less precise edges.
    """
    from preprocess import preprocess_image
    from segment   import load_model as load_deeplab, get_deeplab_mask

    # Resolve the paint color — exact first, fuzzy fallback.
    color = get_color(color_query)
    if color is None:
        suggestions = search_colors(color_query, n=3)
        if not suggestions:
            print(f"[recolor] ERROR: '{color_query}' not found and no close matches.")
            sys.exit(1)
        print(f"[recolor] '{color_query}' not found exactly. Closest match: {suggestions[0]}")
        print("[recolor] Other suggestions:")
        for s in suggestions[1:]:
            print(f"           {s}")
        color = suggestions[0]

    image_path = Path(image_path)
    stem       = image_path.stem

    # Stage 1: LAB preprocessing
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

    # Stage 3b: Mask processing — clean + soften before blending
    print("\n--- Stage 3b: Mask processing ---")
    from mask_process import process_mask
    clean_mask = process_mask(refined_mask)
    print(f"[stage3b] Clean mask range: [{clean_mask.min():.3f}, {clean_mask.max():.3f}]  "
          f"mean={clean_mask.mean():.3f}")

    # Stage 5: Lightweight erosion — pulls mask boundary inward by 3px to
    # absorb edge aliasing without fragmenting large wall planes.
    print("\n--- Stage 5: Object protection (lightweight erosion) ---")
    from protect import create_object_protection_mask, apply_protection
    final_mask = apply_protection(clean_mask)
    print(f"[stage5] Final mask range: [{final_mask.min():.3f}, {final_mask.max():.3f}]  "
          f"mean={final_mask.mean():.3f}")

    # Stage 4: Retinex-aware recoloring — preserves shading, handles dark LRV correctly.
    print("\n--- Stage 4: Recoloring (Retinex luminance transfer) ---")
    from wall_enhance import luminance_aware_recolor, compute_wall_color_stats
    wall_stats = compute_wall_color_stats(preprocessed, final_mask)
    recolored  = luminance_aware_recolor(
        preprocessed, final_mask, color.rgb,
        wall_stats=wall_stats, target_lrv=color.lrv,
    )

    # Save outputs — name files after the color code so you can compare easily.
    safe_code  = color.code.replace("/", "-")
    out_recolored  = image_path.parent / f"{stem}_{safe_code}_recolored.jpg"
    out_comparison = image_path.parent / f"{stem}_{safe_code}_comparison.jpg"

    cv2.imwrite(
        str(out_recolored),
        cv2.cvtColor(recolored, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    print(f"[stage4] Recolored image saved: {out_recolored}")

    save_comparison(preprocessed, recolored, final_mask, color, out_comparison)

    safe_name = color.name.encode("ascii", "replace").decode()
    print(f"\nDone.")
    print(f"   Color     : {color.code}  {safe_name}  #{color.hex}")
    print(f"   Recolored : {out_recolored}")
    print(f"   Comparison: {out_comparison}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:   python recolor.py <image_path> <color name or code> [--no-sam]")
        print()
        print("Examples:")
        print("   python recolor.py room.jpg \"Lucy Blue\"")
        print("   python recolor.py room.jpg 5001-5C")
        print("   python recolor.py room.jpg \"warm gray\" --no-sam")
        print()
        print("Run  python colors.py  to browse all available colors.")
        sys.exit(1)

    src        = sys.argv[1]
    no_sam     = "--no-sam" in sys.argv
    # Everything between the image path and any flags is the color query.
    flag_args  = {"--no-sam"}
    color_parts = [a for a in sys.argv[2:] if a not in flag_args]
    color_query = " ".join(color_parts) if color_parts else "4002-9A"

    run_pipeline(src, color_query=color_query, use_sam=not no_sam)
