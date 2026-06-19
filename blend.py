"""
blend.py — Stage 8 of the wall recoloring pipeline.

Final blending step: combines the original image I(x,y), the target
reflectance layer R(x,y) from Stage 7, and the final mask M_final(x,y)
from Stage 5 to produce the output image O(x,y).

This is the Magic-Wall / PRISM formulation:

    O(x,y) = M_final(x,y) * R(x,y) + (1 - M_final(x,y)) * I(x,y)

---
WHY A SOFT MASK M ∈ [0, 1] (NOT BINARY)?
-----------------------------------------
A binary mask (0 or 1) draws a hard line between wall and non-wall.
Where that line sits on a real boundary — e.g. the edge of a sofa cushion
— the recolored and original images meet with a sharp, visible seam.

A soft mask encodes confidence: pixels deep inside the wall get M ≈ 1
(fully recolored), pixels on the boundary get M ≈ 0.3–0.7 (partially
blended), and pixels outside the wall get M = 0 (untouched).

The blending formula then naturally fades the new color into the original
across those transitional pixels, making the seam invisible.

---
HOW BLENDING APPROXIMATES I = R * S (WITHOUT COMPUTING S EXPLICITLY)
---------------------------------------------------------------------
The ideal operation under the I = R * S image model would be:
    O = R_target * S = R_target * (I / R_true)

But computing I / R_true requires knowing R_true perfectly — which we don't.

Alpha blending skips that problem entirely. Where M = 1:
    O = R_target * 1 + I * 0 = R_target  (full new color)

Where M = 0:
    O = R_target * 0 + I * 1 = I  (full original, including all shading)

For intermediate M values the output is a weighted average. In practice,
the Stage 4 LAB-space preprocessing and the soft mask mean the blended
pixels already carry most of the shading information — the transition looks
natural because the boundary pixels retain a mix of the original shading.

The HSV variant (mode="hsv") goes further: it keeps the V (brightness)
channel of I unchanged across the ENTIRE wall region, replacing only hue
and saturation. This directly implements "preserve S, change R" — at the
cost of being slightly less accurate at very bright/dark extremes.

---
WHY STEP 5 (PROTECTION MASK) MATTERS SPECIFICALLY HERE
-------------------------------------------------------
M_final = M_wall * (1 - M_protect). The (1 - M_protect) factor zeros out
every pixel inside the safety buffer around objects. Those pixels get
M_final = 0 regardless of M_wall, so the blending formula gives:
    O = 0 * R + 1 * I = I   (original, untouched)

Without the protection mask, low-confidence wall pixels near a sofa edge
would receive a small M value (say 0.1), and blending would apply a faint
colour tint to the sofa. With protection, those pixels are guaranteed to be
zero — no tint, no matter how small.

---
WHY STEP 7 (COLOR LAYER) MUST BE SEPARATE
------------------------------------------
R(x,y) needs to exist as a proper array before the blend so that the
formula O = M*R + (1-M)*I can be evaluated as a single vectorized
broadcast with no conditional logic. If R were computed on-the-fly inside
the blend, you couldn't reuse it for multiple blend modes, multiple masks,
or A/B comparison renders without extra cost. Separation also means the
color layer can be brightness-normalized (Stage 7 option) independently of
how the blending is done.
"""

from __future__ import annotations

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Literal


# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------

# Default blend mode.
# "rgb"  — standard linear alpha blend on all three channels.
#           Simple and fast; works well when mask feathering handles edges.
# "hsv"  — preserve the original V (brightness) channel while blending
#           only H (hue) and S (saturation).  Better for rooms with strong
#           directional lighting or uneven brightness across the wall.
DEFAULT_BLEND_MODE: Literal["rgb", "hsv"] = "rgb"

# When mode="hsv", how much of the original saturation to preserve.
# 0.0 = use only the target color's saturation.
# 0.3 = 30% original saturation blended in (softens very vivid target colors).
HSV_SATURATION_PRESERVE: float = 0.0


# ---------------------------------------------------------------------------
# Core blend function
# ---------------------------------------------------------------------------

def blend_images(
    image:        np.ndarray,
    color_layer:  np.ndarray,
    mask:         np.ndarray,
    mode:         Literal["rgb", "hsv"] = DEFAULT_BLEND_MODE,
    hsv_sat_preserve: float = HSV_SATURATION_PRESERVE,
) -> np.ndarray:
    """
    Blend the target color layer into the original image using the soft mask.

    Magic-Wall / PRISM formula:
        O(x,y) = M_final(x,y) * R(x,y) + (1 - M_final(x,y)) * I(x,y)

    Two modes available:
        "rgb" — linear blend on all three channels (default).
        "hsv" — keep original brightness (V), blend only hue + saturation.
                Better preserves shading in strongly-lit rooms.

    Args:
        image:           (H, W, 3) uint8 RGB — original image from Stage 1.
        color_layer:     (H, W, 3) uint8 RGB — target reflectance from Stage 7.
        mask:            (H, W)    float32   — M_final from Stage 5, values [0,1].
        mode:            Blend mode: "rgb" or "hsv".
        hsv_sat_preserve: (HSV mode only) fraction of original saturation to keep.

    Returns:
        output: (H, W, 3) uint8 RGB — the recolored image.
    """
    if mode == "hsv":
        return _blend_hsv(image, color_layer, mask, hsv_sat_preserve)
    return _blend_rgb(image, color_layer, mask)


# ---------------------------------------------------------------------------
# RGB blending
# ---------------------------------------------------------------------------

def _blend_rgb(
    image:       np.ndarray,
    color_layer: np.ndarray,
    mask:        np.ndarray,
) -> np.ndarray:
    """
    Standard linear alpha blend.

        O = M * R + (1 - M) * I

    Everything stays in [0, 255] uint8 space but the arithmetic is done
    in float32 to avoid rounding errors from repeated integer truncation.

    Key implementation note:
    The mask is (H, W) but I and R are (H, W, 3). We add a trailing
    dimension with [:, :, np.newaxis] so NumPy broadcasts the scalar mask
    value across all three colour channels — no explicit loop needed.
    """
    # Work in float32 to preserve precision during the weighted sum.
    I = image.astype(np.float32)        # (H, W, 3)
    R = color_layer.astype(np.float32)  # (H, W, 3)

    # Expand mask: (H, W) → (H, W, 1) so it broadcasts over the 3 channels.
    M = np.clip(mask, 0.0, 1.0)[:, :, np.newaxis]  # (H, W, 1)

    # The Magic-Wall blend: weighted sum, fully vectorized.
    output = M * R + (1.0 - M) * I     # (H, W, 3) float32

    return np.clip(output, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# HSV blending — preserves original brightness (shading)
# ---------------------------------------------------------------------------

def _blend_hsv(
    image:           np.ndarray,
    color_layer:     np.ndarray,
    mask:            np.ndarray,
    sat_preserve:    float,
) -> np.ndarray:
    """
    HSV-space blend: replace hue and saturation, keep original brightness.

    Why this is closer to the true I = R * S model:
    In HSV, the V (value / brightness) channel directly encodes the shading
    component S(x,y) — how much light is hitting the surface at that point.
    By keeping V from the original image and replacing only H and S with the
    target color's hue and saturation, we effectively perform:

        O_hue = H_target,  O_sat = S_target,  O_val = V_original

    This is a discrete-color equivalent of R_target * S_original — the
    ideal I = R * S substitution — applied per-pixel with no S estimation.

    The soft mask still controls the TRANSITION at boundaries: near-edge
    pixels blend between (H_target, S_target, V_orig) and the fully
    original pixel (H_orig, S_orig, V_orig), so the hue change fades in
    smoothly rather than switching abruptly.

    Pipeline:
        I_rgb → I_hsv: (H_I, S_I, V_I)
        R_rgb → R_hsv: (H_R, S_R, V_R)

        H_out = M * H_R + (1-M) * H_I   — blend hue
        S_out = M * S_R + (1-M) * S_I   — blend saturation
                + sat_preserve * S_I     — optional original saturation mix-in
        V_out = V_I                      — brightness ALWAYS from original

        O_hsv → O_rgb
    """
    M = np.clip(mask, 0.0, 1.0)     # (H, W)

    # cv2 HSV scale: H ∈ [0,180], S ∈ [0,255], V ∈ [0,255]
    I_bgr = cv2.cvtColor(image,       cv2.COLOR_RGB2BGR)
    R_bgr = cv2.cvtColor(color_layer, cv2.COLOR_RGB2BGR)

    I_hsv = cv2.cvtColor(I_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    R_hsv = cv2.cvtColor(R_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    H_I, S_I, V_I = I_hsv[:,:,0], I_hsv[:,:,1], I_hsv[:,:,2]
    H_R, S_R       = R_hsv[:,:,0], R_hsv[:,:,1]
    # V_R is intentionally unused — we always keep V from the original.

    # --- Hue blending ---
    # Hue is circular (0–180 in cv2), so naive linear blending can
    # produce wrong results when the two hues straddle 0/180 (e.g.,
    # 5° and 175° should blend through 0°, not 90°). We handle this
    # with a circular mean approach.
    H_diff = H_R - H_I
    # Wrap difference into [-90, 90] so we always take the short arc.
    H_diff = (H_diff + 90) % 180 - 90
    H_out  = (H_I + M * H_diff) % 180

    # --- Saturation blending ---
    # sat_preserve lets you keep a fraction of the original saturation
    # so very vivid target colors don't look artificially punchy.
    S_target_blend = (1.0 - sat_preserve) * S_R + sat_preserve * S_I
    S_out = M * S_target_blend + (1.0 - M) * S_I

    # --- Brightness: always from the original ---
    V_out = V_I   # shading / lighting completely preserved

    out_hsv = np.stack([H_out, S_out, V_out], axis=2)
    out_hsv = np.clip(out_hsv, 0, 255).astype(np.uint8)

    out_bgr = cv2.cvtColor(out_hsv, cv2.COLOR_HSV2BGR)
    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

    # HSV conversion introduces ±1 rounding errors per channel on the roundtrip
    # (float32 HSV → uint8 → BGR → RGB). Where the mask is zero the output
    # MUST equal the original exactly — restore those pixels to guarantee it.
    out_rgb[M < 0.001] = image[M < 0.001]

    return out_rgb


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_blend(
    image:       np.ndarray,
    color_layer: np.ndarray,
    output:      np.ndarray,
    mask:        np.ndarray,
    mode:        str = DEFAULT_BLEND_MODE,
    save_path:   str | Path | None = None,
) -> None:
    """
    Four-panel comparison figure.

    Panels:
        1. Original image I(x,y)
        2. Target color layer R(x,y)
        3. Final mask M_final (heatmap)
        4. Blended output O(x,y)

    Args:
        image:       (H, W, 3) uint8 original.
        color_layer: (H, W, 3) uint8 color layer.
        output:      (H, W, 3) uint8 blended result.
        mask:        (H, W)    float32 final mask.
        mode:        Blend mode label for the title.
        save_path:   If set, saves figure here.
    """
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(
        f"Stage 8 — Magic-Wall Blending  (mode={mode})\n"
        "O(x,y) = M_final * R(x,y) + (1 - M_final) * I(x,y)",
        fontsize=12,
    )

    axes[0].imshow(image)
    axes[0].set_title("I(x,y)  Original")
    axes[0].axis("off")

    r, g, b = color_layer[0, 0]
    axes[1].imshow(color_layer)
    axes[1].set_title(f"R(x,y)  Color layer\nRGB({r},{g},{b})  #{r:02X}{g:02X}{b:02X}")
    axes[1].axis("off")

    im = axes[2].imshow(mask, cmap="hot", vmin=0, vmax=1)
    axes[2].set_title(
        f"M_final  Soft mask\n"
        f"mean={mask.mean():.3f}  "
        f"coverage={100*(mask>0.5).mean():.1f}%"
    )
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(output)
    axes[3].set_title("O(x,y)  Final output")
    axes[3].axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(str(save_path), dpi=160, bbox_inches="tight")
        print(f"[blend] Visualisation saved: {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Quick test — python blend.py <image_path> [color_code] [--hsv]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from colors       import get_color
    from preprocess   import preprocess_image
    from segment      import load_model as load_deeplab, get_deeplab_mask
    from refine       import load_sam_model, refine_mask_with_sam
    from mask_process import process_mask
    from protect      import create_object_protection_mask, apply_protection
    from color_layer  import generate_color_layer

    if len(sys.argv) < 2:
        print("Usage:   python blend.py <image_path> [color_code] [--hsv]")
        print("Example: python blend.py room.jpg 8001-1G --hsv")
        sys.exit(1)

    src         = sys.argv[1]
    stem        = Path(src).stem
    use_hsv     = "--hsv" in sys.argv
    mode        = "hsv" if use_hsv else "rgb"
    flag_args   = {"--hsv"}
    color_parts = [a for a in sys.argv[2:] if a not in flag_args]
    color_query = " ".join(color_parts) if color_parts else "8001-1G"

    color = get_color(color_query)
    if color is None:
        print(f"Color '{color_query}' not found."); sys.exit(1)
    print(f"[main] Target: {color}  mode={mode}")

    print("[main] Stage 1 ...")
    preprocessed = preprocess_image(src)

    print("[main] Stage 2 ...")
    deeplab, dl_dev = load_deeplab()
    coarse = get_deeplab_mask(preprocessed, model=deeplab, device=dl_dev)

    print("[main] Stage 3 ...")
    predictor, _ = load_sam_model()
    sam_mask = refine_mask_with_sam(preprocessed, coarse, predictor=predictor)

    print("[main] Stage 4 ...")
    clean_mask = process_mask(sam_mask)

    print("[main] Stage 5 ...")
    protection = create_object_protection_mask(clean_mask)
    final_mask = apply_protection(clean_mask, protection)

    print("[main] Stage 7 ...")
    layer = generate_color_layer(preprocessed, target_color=color.rgb)

    print(f"[main] Stage 8: blending (mode={mode}) ...")
    output = blend_images(preprocessed, layer, final_mask, mode=mode)

    print(f"[main] Output shape: {output.shape}  dtype: {output.dtype}")

    out_path  = f"{stem}_{color.code}_{mode}_blend.jpg"
    cv2.imwrite(
        out_path,
        cv2.cvtColor(output, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    print(f"[main] Saved: {out_path}")

    visualize_blend(
        preprocessed, layer, output, final_mask,
        mode=mode,
        save_path=f"{stem}_{color.code}_{mode}_blend_comparison.png",
    )
