"""
color_layer.py — Stage 7 of the wall recoloring pipeline.

Generates the target reflectance layer R(x,y) — a full-size synthetic image
filled with the desired paint color.  This layer is NOT the final output;
it feeds directly into the blending equation in Stage 8:

    O(x,y) = M_final(x,y) * R(x,y) + (1 - M_final(x,y)) * I(x,y)

---
THE IMAGE FORMATION MODEL:  I = R * S
---
Every pixel in a photograph is the product of two independent signals:

    I(x,y) = R_true(x,y) * S(x,y)

Where:
  I(x,y)      = the observed pixel value (what the camera captured)
  R_true(x,y) = reflectance — the intrinsic surface color of the material
                at that point, independent of lighting conditions
  S(x,y)      = shading — the local lighting intensity at that point
                (includes direct illumination, shadows, ambient occlusion)

This model explains why the same wall looks brighter near a window and
darker in a corner: R_true is the same paint color everywhere, but S(x,y)
varies across the surface.

What we want to do:
  Replace R_true(x,y) with R_target(x,y) = C_target (the new paint color)
  while KEEPING S(x,y) unchanged.

  If we could perfectly isolate S from I we could compute:
      O = R_target * S = R_target * (I / R_true)

  In practice we don't have a perfect separation, so Stage 8 uses the
  soft blending approach — but R(x,y) built here is the "what the wall
  should look like if it were painted C_target" layer.

---
WHY DO WE GENERATE A FULL-SIZE IMAGE, NOT JUST THE WALL PIXELS?
---
The blending formula O = M * R + (1 - M) * I is an element-wise matrix
operation. R must have the same spatial dimensions as I and M so that every
pixel can be computed in one vectorized step — no loops, no indexing tricks.
The mask M handles which pixels actually matter; R just needs to be there.

---
WHY IS MASKING NOT APPLIED HERE?
---
Separation of concerns — this stage ONLY answers "what color should the
wall be?"  The WHEN and WHERE of applying that color is entirely Stage 8's
responsibility.  Keeping the two operations separate means:
  - You can swap in a different target color without touching the mask logic.
  - You can swap in a different mask without regenerating the color layer.
  - You can inspect R(x,y) independently to verify the color looks right
    before committing to a full pipeline run.
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

# Brightness adjustment factor applied to the target color layer.
# 1.0 = no change.  0.9 = 10% darker (useful for dark-to-light transitions
# where the raw target color would look overexposed).  Range: 0.5–1.5.
BRIGHTNESS_FACTOR: float = 1.0

# Saturation scale applied AFTER generating the color layer (in LAB space).
# 1.0 = no change.  1.15 = 15% more vivid.  Range: 0.8–1.3.
# Useful when the target color looks washed out in the final blend because
# the original wall was very bright and the blending averages it down.
SATURATION_SCALE: float = 1.0


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def generate_color_layer(
    image:            np.ndarray,
    target_color:     tuple[int, int, int],
    brightness_factor: float = BRIGHTNESS_FACTOR,
    saturation_scale:  float = SATURATION_SCALE,
    dominant_color:   tuple[int, int, int] | None = None,
    normalize:        bool = False,
) -> np.ndarray:
    """
    Build a full-size target reflectance layer R(x,y) = C_target.

    The output is a (H, W, 3) image where every pixel is set to the
    target paint color (possibly brightness- and saturation-adjusted).
    It has exactly the same shape as the input image so it can be fed
    directly into the blending formula in Stage 8.

    Args:
        image:             (H, W, 3) uint8 RGB original image.
                           Only its spatial dimensions are used.
        target_color:      (R, G, B) desired paint color from the
                           Valspar database or any RGB triplet.
        brightness_factor: Multiply the L channel of the color layer
                           by this value.  1.0 = keep as-is.
        saturation_scale:  Scale A and B channels in LAB around
                           neutral (128) by this factor.  1.0 = no change.
        dominant_color:    Optional dominant wall color from Stage 6.
                           Required when normalize=True.
        normalize:         If True, scale the target color so its
                           brightness relative to the dominant wall
                           color is preserved.  See formula below.

    Returns:
        color_layer: (H, W, 3) uint8 RGB array.
                     Every pixel equals the (adjusted) target color.
                     Ready to be passed to Stage 8 blending.

    Normalization formula (when normalize=True):
        R_adjusted = C_target * (||I_wall_mean|| / ||C_wall||)

        Where:
          ||C_wall||   = Euclidean magnitude of the dominant wall color
          ||C_target|| = Euclidean magnitude of the target color
          The ratio scales C_target up or down to match the brightness
          the original wall had, so a dark-to-light swap doesn't suddenly
          look overexposed and a light-to-dark swap doesn't look crushed.
    """
    h, w = image.shape[:2]
    r_t, g_t, b_t = [int(c) for c in target_color]

    # ------------------------------------------------------------------
    # Step 1 — Build the flat color layer
    # ------------------------------------------------------------------
    # np.ones_like gives an array of the same shape and dtype as the image.
    # Multiplying by the target color fills every pixel with C_target.
    # This is a single vectorized broadcast — no Python loops.
    color_layer = np.ones((h, w, 3), dtype=np.float32)
    color_layer[:, :, 0] = r_t
    color_layer[:, :, 1] = g_t
    color_layer[:, :, 2] = b_t

    # ------------------------------------------------------------------
    # Step 2 — Optional normalization (brightness preservation)
    # ------------------------------------------------------------------
    # When do you want this?
    # The raw target color C_target comes from a paint swatch measured
    # under standard D65 illumination.  The original wall in the photo
    # may be significantly brighter or darker (e.g., a sun-lit wall vs.
    # a shade wall).  Without normalization, replacing R_true with C_target
    # changes the apparent brightness of the wall, which looks unrealistic.
    #
    # With normalization, we scale C_target so its brightness vector
    # magnitude matches the original wall's magnitude:
    #
    #   scale = ||C_wall|| / ||C_target||   →  then  C_adj = C_target * scale
    #
    # This is equivalent to assuming the shading S cancels out and only
    # the reflectance changes — which is the correct I = R * S assumption.
    if normalize and dominant_color is not None:
        c_wall   = np.array(dominant_color,  dtype=np.float32)
        c_target = np.array(target_color,    dtype=np.float32)
        mag_wall   = float(np.linalg.norm(c_wall))
        mag_target = float(np.linalg.norm(c_target))
        if mag_target > 1e-6 and mag_wall > 1e-6:
            scale = mag_wall / mag_target
            color_layer = color_layer * scale
            print(
                f"[color_layer] Normalization: "
                f"||C_wall||={mag_wall:.1f}  ||C_target||={mag_target:.1f}  "
                f"scale={scale:.3f}"
            )
        else:
            print("[color_layer] Normalization skipped (zero-magnitude color).")
    elif normalize and dominant_color is None:
        print("[color_layer] normalize=True but no dominant_color provided — skipping.")

    # ------------------------------------------------------------------
    # Step 3 — Brightness adjustment (in LAB lightness channel)
    # ------------------------------------------------------------------
    # Working in LAB means adjusting brightness (L) independently of
    # hue (A, B), so the color doesn't shift toward white or black as
    # it brightens/darkens — it stays the same hue, just lighter/darker.
    if brightness_factor != 1.0:
        layer_uint8 = np.clip(color_layer, 0, 255).astype(np.uint8)
        layer_bgr   = cv2.cvtColor(layer_uint8, cv2.COLOR_RGB2BGR)
        layer_lab   = cv2.cvtColor(layer_bgr,   cv2.COLOR_BGR2LAB).astype(np.float32)
        layer_lab[:, :, 0] = np.clip(layer_lab[:, :, 0] * brightness_factor, 0, 255)
        layer_lab_u8 = np.clip(layer_lab, 0, 255).astype(np.uint8)
        layer_bgr_out = cv2.cvtColor(layer_lab_u8, cv2.COLOR_LAB2BGR)
        color_layer   = cv2.cvtColor(layer_bgr_out, cv2.COLOR_BGR2RGB).astype(np.float32)

    # ------------------------------------------------------------------
    # Step 4 — Saturation scaling (in LAB A/B channels)
    # ------------------------------------------------------------------
    # Scaling A and B away from neutral (128) makes the color more vivid
    # without touching lightness — the same technique used in mask_process.py.
    # This is useful when the blending in Stage 8 will partially average the
    # color layer with the (possibly dull) original image.
    if saturation_scale != 1.0:
        neutral   = 128.0
        layer_u8  = np.clip(color_layer, 0, 255).astype(np.uint8)
        layer_bgr = cv2.cvtColor(layer_u8, cv2.COLOR_RGB2BGR)
        layer_lab = cv2.cvtColor(layer_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        for ch in [1, 2]:   # A and B channels
            layer_lab[:, :, ch] = np.clip(
                (layer_lab[:, :, ch] - neutral) * saturation_scale + neutral,
                0, 255,
            )
        layer_u8_out  = np.clip(layer_lab, 0, 255).astype(np.uint8)
        layer_bgr_out = cv2.cvtColor(layer_u8_out, cv2.COLOR_LAB2BGR)
        color_layer   = cv2.cvtColor(layer_bgr_out, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Final clip and cast to uint8 — Stage 8 expects uint8 input.
    color_layer = np.clip(color_layer, 0, 255).astype(np.uint8)

    r_out, g_out, b_out = color_layer[0, 0]
    print(
        f"[color_layer] Generated layer: "
        f"target=RGB({r_t},{g_t},{b_t})  "
        f"adjusted=RGB({r_out},{g_out},{b_out})  "
        f"shape={color_layer.shape}"
    )
    return color_layer


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_color_layer(
    image_rgb:    np.ndarray,
    color_layer:  np.ndarray,
    mask:         np.ndarray | None = None,
    target_color: tuple[int, int, int] | None = None,
    save_path:    str | Path | None = None,
) -> None:
    """
    Three-panel figure:
        1. Original image
        2. The color layer R(x,y)  — the flat target color
        3. Preview blend: what Stage 8 will produce
           (uses mask if provided, otherwise shows the flat layer)

    Args:
        image_rgb:    (H, W, 3) uint8 original image.
        color_layer:  (H, W, 3) uint8 color layer from generate_color_layer().
        mask:         Optional (H, W) float32 for the preview blend.
        target_color: Optional (R, G, B) triplet for the title label.
        save_path:    If set, saves figure here.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    label = f"RGB{target_color}" if target_color else "target color"
    fig.suptitle(f"Stage 7 — Color Layer Generation  ({label})", fontsize=13)

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original image")
    axes[0].axis("off")

    axes[1].imshow(color_layer)
    r, g, b = color_layer[0, 0]
    axes[1].set_title(
        f"Color layer  R(x,y)\n"
        f"RGB({r},{g},{b})  #{r:02X}{g:02X}{b:02X}\n"
        f"shape {color_layer.shape}"
    )
    axes[1].axis("off")

    # Preview blend: O = M * R + (1 - M) * I
    # This shows what Stage 8 will produce with the current mask.
    if mask is not None:
        m = np.clip(mask, 0, 1)[:, :, np.newaxis].astype(np.float32)
        preview = (
            m * color_layer.astype(np.float32) +
            (1.0 - m) * image_rgb.astype(np.float32)
        )
        preview = np.clip(preview, 0, 255).astype(np.uint8)
        axes[2].imshow(preview)
        axes[2].set_title(
            "Preview blend\n"
            "O = M_final * R(x,y) + (1 - M_final) * I(x,y)\n"
            "(Stage 8 preview)"
        )
    else:
        # No mask — just show the layer again with a note.
        axes[2].imshow(color_layer)
        axes[2].set_title("Color layer\n(pass mask= for blend preview)")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"[color_layer] Visualisation saved: {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Quick test — python color_layer.py <image_path> <color_code>
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from colors       import get_color
    from preprocess   import preprocess_image
    from segment      import load_model as load_deeplab, get_deeplab_mask
    from refine       import load_sam_model, refine_mask_with_sam
    from mask_process import process_mask
    from protect      import create_object_protection_mask, apply_protection
    from color_detect import extract_wall_color

    if len(sys.argv) < 2:
        print("Usage:   python color_layer.py <image_path> [color_code]")
        print("Example: python color_layer.py room.jpg 8001-1G")
        sys.exit(1)

    src          = sys.argv[1]
    color_query  = sys.argv[2] if len(sys.argv) > 2 else "8001-1G"
    stem         = Path(src).stem

    color = get_color(color_query)
    if color is None:
        print(f"Color '{color_query}' not found."); sys.exit(1)
    print(f"[main] Target color: {color}")

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
    wall_color_result = extract_wall_color(preprocessed, final_mask)

    print("[main] Stage 7: generate color layer ...")
    layer = generate_color_layer(
        preprocessed,
        target_color=color.rgb,
        dominant_color=tuple(wall_color_result.dominant_color),
        normalize=False,   # set True to try brightness normalization
    )

    visualize_color_layer(
        preprocessed, layer,
        mask=final_mask,
        target_color=color.rgb,
        save_path=f"{stem}_{color.code}_color_layer.png",
    )
