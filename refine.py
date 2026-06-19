"""
refine.py — Stage 3 of the wall recoloring pipeline.

Takes the coarse probability mask from DeepLabV3 (Stage 2) and sharpens it
using Meta's Segment Anything Model (SAM).

WHY DO WE NEED THIS?
DeepLab is a CNN trained to classify every pixel, but CNNs are fundamentally
limited by their training resolution and the size of their receptive fields.
They tend to:
  - Blur boundaries — the "wall vs. sofa" edge becomes a gradient of uncertain
    pixels several pixels wide instead of a clean line.
  - Miss thin structures — window frames, door edges, light switches.
  - Bleed across sharp depth edges — the model "fills in" corners it can't
    cleanly classify.

SAM was designed from the ground up for precise, class-agnostic segmentation.
Its Vision Transformer (ViT) backbone encodes the full image at once (no
sliding window), so it can see global context while still resolving fine
boundary detail. Given a prompt (a handful of point coordinates), SAM
produces a crisp binary mask that accurately follows object edges.

The strategy here:
  1. Threshold the coarse DeepLab mask to extract high-confidence "wall" pixels
     and high-confidence "not wall" pixels.
  2. Sample a small set of coordinate prompts from each group.
  3. Feed those prompts into SAM — it figures out exactly where the wall
     boundary is.
  4. Multiply the resulting SAM binary mask by the original probability mask
     so the output is still a soft float array (not a hard 0/1 mask).

--------------------------------------------------------------------------------
SETUP — REQUIRED BEFORE USE
--------------------------------------------------------------------------------
Install the segment-anything package:
    pip install git+https://github.com/facebookresearch/segment-anything.git

Download a pretrained checkpoint (choose one):
    ViT-B  (~375 MB, faster,  good for most room photos):
        wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

    ViT-H  (~2.4 GB, slower, best boundary precision):
        wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

Set SAM_CHECKPOINT and SAM_MODEL_TYPE below to match your downloaded file.
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Try to import SAM — give a helpful error if it is not installed.
try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError as exc:
    raise ImportError(
        "segment-anything is not installed.\n"
        "Run: pip install git+https://github.com/facebookresearch/segment-anything.git\n"
        "Then download a checkpoint — see the docstring at the top of refine.py."
    ) from exc


# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------

# Path to the downloaded SAM checkpoint file.
SAM_CHECKPOINT: str = "sam_vit_b_01ec64.pth"

# Model type must match the checkpoint file:
#   "vit_b" → sam_vit_b_01ec64.pth  (default, balanced)
#   "vit_l" → sam_vit_l_0b3195.pth
#   "vit_h" → sam_vit_h_4b8939.pth  (most accurate, most VRAM)
SAM_MODEL_TYPE: str = "vit_b"

# Pixels with P(wall) above this are treated as confident foreground.
# Raised to 0.80 so only truly high-confidence wall pixels become prompts.
# This prevents ambiguous surfaces (wood panels, trim) from polluting the
# foreground signal and reduces over-segmentation.
FG_THRESHOLD: float = 0.80

# Pixels with P(wall) below this are treated as confident background.
# Lowered to 0.20 — tighter band means only clearly-non-wall pixels are used.
# The wider gap between FG and BG thresholds (0.80 vs 0.20) leaves the
# uncertain middle zone unprompted, letting SAM decide those boundaries itself.
BG_THRESHOLD: float = 0.20

# Total prompt budget per category.
# FG budget kept high for full wall coverage across all zones.
# BG budget reduced to 4 — enough to hint at object locations without
# aggressively suppressing wall regions near ambiguous boundaries.
MAX_FG_POINTS: int = 30
MAX_BG_POINTS: int = 4

# Divide the image into an N_ZONES x N_ZONES grid before sampling.
# Each zone contributes its fair share of FG and BG points, guaranteeing
# spatial spread even if the wall only covers one corner of the image.
N_ZONES: int = 4


# ---------------------------------------------------------------------------
# Step 1 — Load the SAM model
# ---------------------------------------------------------------------------

def load_sam_model(
    checkpoint: str | Path = SAM_CHECKPOINT,
    model_type: str = SAM_MODEL_TYPE,
    device: torch.device | None = None,
) -> tuple[SamPredictor, torch.device]:
    """
    Load a pretrained SAM model and wrap it in a SamPredictor.

    SamPredictor is SAM's high-level inference interface. You give it an image
    once (set_image), then call predict() as many times as you like with
    different prompts — the expensive image encoding only happens once.

    How SAM works under the hood:
      1. A Vision Transformer (ViT) encodes the full image into a rich feature
         map (the "image embedding"). This is the slow part (~0.1–1 s).
      2. A lightweight prompt encoder converts your point/box/mask prompts
         into vectors.
      3. A mask decoder combines the image embedding + prompt vectors to
         produce up to 3 candidate binary masks in milliseconds.

    Args:
        checkpoint:  Path to the .pth checkpoint file.
        model_type:  One of "vit_b", "vit_l", "vit_h".
        device:      Torch device; auto-detected if None.

    Returns:
        (predictor, device)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found: {checkpoint}\n"
            "Download it with:\n"
            "  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        )

    print(f"[refine] Loading SAM {model_type} from {checkpoint} on {device} ...")
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device)
    sam.eval()

    predictor = SamPredictor(sam)
    print("[refine] SAM loaded and ready.")
    return predictor, device


# ---------------------------------------------------------------------------
# Step 2 — Convert the coarse mask into SAM prompt points
# ---------------------------------------------------------------------------

def generate_sam_prompts(
    coarse_mask:   np.ndarray,
    fg_threshold:  float = FG_THRESHOLD,
    bg_threshold:  float = BG_THRESHOLD,
    max_fg_points: int   = MAX_FG_POINTS,
    max_bg_points: int   = MAX_BG_POINTS,
    n_zones:       int   = N_ZONES,
    image:         np.ndarray | None = None,
    wall_stats:    dict  | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample spatially-spread foreground and background prompts from the mask.

    What are prompts in SAM?
    SAM is a promptable model — it doesn't classify every pixel like DeepLab.
    Instead, you give it coordinate hints and labels:
        label = 1  → "this point is INSIDE the wall"
        label = 0  → "this point is OUTSIDE the wall"
    SAM draws the sharpest boundary it can find that separates the two groups.

    Why higher thresholds (0.80 / 0.20)?
    Wood panels and decorative surfaces score in the 0.4–0.7 range — they are
    ambiguous to DeepLab. By only sampling points where DeepLab is truly
    confident (> 0.80 for wall, < 0.20 for non-wall), we avoid giving SAM
    misleading hints about ambiguous regions and let it resolve those areas
    from visual evidence instead of noisy prompts.

    Why zone-based spatial sampling?
    The old grid approach sampled from wherever the mask was densest, which
    in practice means the horizontal center of the room where the wall is most
    clearly visible. That left corners, edges, and upper portions under-prompted.
    Zone-based sampling divides the image into an n_zones x n_zones grid and
    allocates a prompt budget to each cell. Every part of the image gets
    representation, so SAM understands the wall boundary all the way around —
    not just in the middle.

    Args:
        coarse_mask:   (H, W) float32 probability mask from DeepLab.
        fg_threshold:  Min confidence to treat a pixel as a foreground prompt.
        bg_threshold:  Max confidence to treat a pixel as a background prompt.
        max_fg_points: Total foreground prompt budget (spread across zones).
        max_bg_points: Total background prompt budget (spread across zones).
        n_zones:       Number of grid divisions per axis (n_zones x n_zones).

    Returns:
        point_coords: (N, 2) float array of (x, y) prompt coordinates.
        point_labels: (N,)   int array of 1 (foreground) or 0 (background).

    Raises:
        ValueError: if no foreground prompts can be found.
    """
    h, w = coarse_mask.shape
    rng = np.random.default_rng(seed=42)   # fixed seed for reproducibility

    # How many points each zone contributes from its local budget.
    # Integer division — any remainder is silently dropped (acceptable rounding).
    n_cells = n_zones * n_zones
    fg_per_zone = max(1, max_fg_points // n_cells)
    bg_per_zone = max(1, max_bg_points // n_cells)

    zone_h = max(1, h // n_zones)
    zone_w = max(1, w // n_zones)

    fg_list: list[tuple[int, int]] = []   # (x, y) pairs
    bg_list: list[tuple[int, int]] = []

    for zi in range(n_zones):
        for zj in range(n_zones):
            # Pixel bounds for this zone — clamp to image edges.
            y0, y1 = zi * zone_h, min(h, (zi + 1) * zone_h)
            x0, x1 = zj * zone_w, min(w, (zj + 1) * zone_w)

            zone = coarse_mask[y0:y1, x0:x1]

            # --- Foreground pixels in this zone (high-confidence wall) ---
            fy, fx = np.where(zone >= fg_threshold)
            if len(fy) > 0:
                k = min(fg_per_zone, len(fy))
                chosen = rng.choice(len(fy), k, replace=False)
                for idx in chosen:
                    fg_list.append((int(x0 + fx[idx]), int(y0 + fy[idx])))

            # --- Background pixels in this zone (high-confidence non-wall) ---
            by, bx = np.where(zone <= bg_threshold)
            if len(by) > 0:
                k = min(bg_per_zone, len(by))
                chosen = rng.choice(len(by), k, replace=False)
                for idx in chosen:
                    bg_list.append((int(x0 + bx[idx]), int(y0 + by[idx])))

    if not fg_list:
        raise ValueError(
            f"No foreground prompts found (no pixels with mask > {fg_threshold} "
            f"in any of the {n_zones}x{n_zones} zones). "
            "Try lowering FG_THRESHOLD."
        )

    fg_arr    = np.array(fg_list, dtype=np.float32)      # (n_fg, 2)
    fg_labels = np.ones(len(fg_arr), dtype=int)

    if not bg_list and image is not None:
        # Mask-based approach found no background points.
        # This happens when DeepLab assigns high background probability to
        # everything (bathrooms, kitchens) — nothing scores below bg_threshold.
        # Fall back to image color analysis: dark pixels, desaturated pixels,
        # and hue-deviant pixels are reliable non-wall indicators.
        from wall_enhance import color_based_background_points
        cb_coords, cb_labels = color_based_background_points(
            image, coarse_mask, wall_stats=wall_stats, n_points=max_bg_points,
        )
        if cb_coords is not None:
            bg_list = [tuple(int(v) for v in p) for p in cb_coords]
            print("[refine] Zero mask-based BG — using color-analysis fallback "
                  f"({len(bg_list)} points from dark/desat/hue-deviant pixels).")

    if bg_list:
        bg_arr    = np.array(bg_list, dtype=np.float32)
        bg_labels = np.zeros(len(bg_arr), dtype=int)
        point_coords = np.concatenate([fg_arr, bg_arr], axis=0)
        point_labels = np.concatenate([fg_labels, bg_labels], axis=0)
    else:
        point_coords = fg_arr
        point_labels = fg_labels

    n_fg = int(fg_labels.sum())
    n_bg = int((point_labels == 0).sum())
    print(
        f"[refine] Prompts: {n_fg} foreground + {n_bg} background "
        f"across {n_zones}x{n_zones} zones "
        f"(thresholds: fg>{fg_threshold}, bg<{bg_threshold})"
    )
    return point_coords, point_labels


# ---------------------------------------------------------------------------
# Step 3 — Run SAM and select the best candidate mask
# ---------------------------------------------------------------------------

def _select_best_mask(
    sam_masks: np.ndarray,
    sam_scores: np.ndarray,
    coarse_mask: np.ndarray,
    fg_threshold: float = FG_THRESHOLD,
) -> np.ndarray:
    """
    Choose which of SAM's 3 candidate masks best matches the coarse mask.

    SAM always returns exactly 3 candidates (ordered rough→fine by default).
    We rank them by IoU against the thresholded coarse mask, then break ties
    with SAM's own confidence score. This gives us the mask that:
      (a) covers roughly the same region DeepLab found, AND
      (b) SAM itself considers high-quality.

    Args:
        sam_masks:    (3, H, W) boolean array — SAM's candidate masks.
        sam_scores:   (3,) float array — SAM's own quality scores.
        coarse_mask:  (H, W) float32 probability mask from DeepLab.

    Returns:
        best_mask: (H, W) boolean array.
    """
    binary_coarse = coarse_mask >= fg_threshold
    best_idx = 0
    best_score = -1.0

    for i, (mask, score) in enumerate(zip(sam_masks, sam_scores)):
        # IoU between this SAM candidate and the thresholded DeepLab mask.
        intersection = np.logical_and(mask, binary_coarse).sum()
        union = np.logical_or(mask, binary_coarse).sum()
        iou = intersection / (union + 1e-6)

        # Combine IoU with SAM's own quality score (equal weight).
        combined = iou * 0.5 + score * 0.5
        if combined > best_score:
            best_score = combined
            best_idx = i

    print(
        f"[refine] Selected SAM mask #{best_idx} "
        f"(score={sam_scores[best_idx]:.3f})"
    )
    return sam_masks[best_idx]


# ---------------------------------------------------------------------------
# Step 4 — Combine SAM binary mask with the coarse probability mask
# ---------------------------------------------------------------------------

def _combine_masks(
    sam_binary: np.ndarray,
    coarse_mask: np.ndarray,
) -> np.ndarray:
    """
    Multiply SAM's binary mask by the original probability mask.

    Why not just use the SAM binary mask directly?
    SAM is excellent at edges but it outputs 0 or 1 — there is no uncertainty.
    The coarse probability mask still carries useful confidence information in
    the interior (e.g. 0.95 in the middle of the wall, 0.72 near a shadow).
    By multiplying:
        M_refined = SAM_binary × M_coarse
    we keep:
      - SAM's precise boundary (wherever SAM says 0, the output is 0 regardless
        of what DeepLab thought)
      - DeepLab's soft interior confidence (pixels deep inside the wall stay
        near their original probability value)

    This is the best of both worlds: crisp edges from SAM, calibrated
    confidence from DeepLab.

    Args:
        sam_binary:   (H, W) boolean array from SAM.
        coarse_mask:  (H, W) float32 probability mask from DeepLab.

    Returns:
        refined_mask: (H, W) float32, values in [0, 1].
    """
    return (sam_binary.astype(np.float32)) * coarse_mask


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def refine_mask_with_sam(
    image: np.ndarray,
    coarse_mask: np.ndarray,
    predictor: SamPredictor | None = None,
    device: torch.device | None = None,
) -> np.ndarray:
    """
    Full Stage 3 pipeline: coarse probability mask → refined probability mask.

    Encodes the image once in SAM, generates prompts from the coarse mask,
    runs SAM prediction, selects the best candidate, and blends the result
    back into a soft probability mask.

    Pass a pre-loaded predictor to avoid reloading SAM on every call.

    Args:
        image:        (H, W, 3) uint8 RGB array — Stage 1 output.
        coarse_mask:  (H, W) float32 array — Stage 2 output (P(wall)).
        predictor:    Optional pre-loaded SamPredictor.
        device:       Torch device override.

    Returns:
        refined_mask: (H, W) float32 array, values in [0, 1].
                      Same spatial size as the input image.
                      Edges are sharper than coarse_mask.
    """
    if predictor is None:
        predictor, device = load_sam_model(device=device)

    # SAM encodes the full image into a feature embedding.
    # This is the expensive operation (~0.1–1 s depending on model size).
    # If you call refine_mask_with_sam multiple times on the SAME image with
    # different masks, call predictor.set_image() once outside and pass the
    # predictor in — it caches the embedding automatically.
    predictor.set_image(image)   # expects uint8 RGB

    # Stage 2.5 (color-based mask refinement) REMOVED.
    # It was compounding suppression with erosion and protection, causing
    # severe mask fragmentation. SAM is fed the raw DeepLab mask directly.

    # Generate prompts from the unmodified coarse mask.
    # image is still passed for the color-based BG fallback when
    # the mask yields zero background points (e.g. bathrooms).
    point_coords, point_labels = generate_sam_prompts(
        coarse_mask, image=image
    )

    # Run SAM — returns 3 candidate masks, their quality scores, and logits.
    sam_masks, sam_scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )

    # Pick the candidate that best overlaps with the DeepLab mask.
    best_mask = _select_best_mask(sam_masks, sam_scores, coarse_mask)

    # Blend SAM's crisp binary boundary with DeepLab's soft probabilities.
    refined_mask = _combine_masks(best_mask, coarse_mask)

    # Small dilation to close holes and restore continuity in large wall planes
    # that SAM may have fragmented into islands.  2px is subtle — just enough
    # to reconnect regions separated by thin gaps, not enough to leak onto objects.
    continuity_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    refined_mask = cv2.dilate(refined_mask, continuity_kernel, iterations=1)
    refined_mask = np.clip(refined_mask, 0.0, 1.0).astype(np.float32)

    return refined_mask   # float32, (H, W), values in [0, 1]


# ---------------------------------------------------------------------------
# Debug visualisation
# ---------------------------------------------------------------------------

def visualize_refinement(
    image_rgb: np.ndarray,
    coarse_mask: np.ndarray,
    refined_mask: np.ndarray,
    point_coords: np.ndarray | None = None,
    point_labels: np.ndarray | None = None,
    save_path: str | Path | None = None,
) -> None:
    """
    Side-by-side figure comparing the coarse and refined masks.
    Optionally overlays SAM prompt points for debugging.

    Panels:
        1. Original image
        2. Coarse DeepLab mask  (heatmap)
        3. Refined SAM mask     (heatmap)
        4. Difference           (refined - coarse, shows where SAM changed things)

    Args:
        image_rgb:     (H, W, 3) uint8 RGB image.
        coarse_mask:   (H, W) float32 DeepLab probability mask.
        refined_mask:  (H, W) float32 refined probability mask.
        point_coords:  (N, 2) optional prompt coordinates to plot (x, y).
        point_labels:  (N,)   optional prompt labels (1=fg, 0=bg).
        save_path:     If set, figure is saved here.
    """
    diff = refined_mask.astype(np.float32) - coarse_mask.astype(np.float32)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle("Stage 3 — SAM Boundary Refinement", fontsize=13)

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original image")
    if point_coords is not None and point_labels is not None:
        for (x, y), lbl in zip(point_coords, point_labels):
            color = "lime" if lbl == 1 else "red"
            marker = "+" if lbl == 1 else "x"
            axes[0].plot(x, y, marker=marker, color=color, markersize=10, markeredgewidth=2)
        axes[0].set_title("Original + SAM prompts\n(green=fg, red=bg)")
    axes[0].axis("off")

    im1 = axes[1].imshow(coarse_mask, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Coarse mask (DeepLab)")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(refined_mask, cmap="jet", vmin=0, vmax=1)
    axes[2].set_title("Refined mask (SAM)")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Difference map: positive = SAM added, negative = SAM removed.
    im3 = axes[3].imshow(diff, cmap="RdBu", vmin=-0.5, vmax=0.5)
    axes[3].set_title("Difference (refined − coarse)\nblue=removed, red=added")
    axes[3].axis("off")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"[refine] Visualisation saved: {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Quick test — python refine.py <image_path>
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from preprocess import preprocess_image
    from segment import load_model as load_deeplab, get_deeplab_mask

    if len(sys.argv) < 2:
        print("Usage:   python refine.py <image_path>")
        print("Example: python refine.py room.jpg")
        sys.exit(1)

    src_path = sys.argv[1]

    # Stage 1 — LAB preprocessing
    print("[main] Stage 1: preprocessing ...")
    preprocessed = preprocess_image(src_path)

    # Stage 2 — DeepLab coarse mask
    print("[main] Stage 2: DeepLab segmentation ...")
    deeplab, dl_device = load_deeplab()
    coarse = get_deeplab_mask(preprocessed, model=deeplab, device=dl_device)

    # Stage 3 — SAM refinement
    print("[main] Stage 3: SAM boundary refinement ...")
    predictor, sam_device = load_sam_model()
    refined = refine_mask_with_sam(preprocessed, coarse, predictor=predictor)

    print(f"[main] Refined mask — shape: {refined.shape}, range: [{refined.min():.3f}, {refined.max():.3f}]")

    # Re-generate prompts for debug visualisation (cheap, no SAM call needed).
    coords, labels = generate_sam_prompts(coarse)

    stem = Path(src_path).stem
    visualize_refinement(
        preprocessed, coarse, refined,
        point_coords=coords,
        point_labels=labels,
        save_path=f"{stem}_refined_mask.png",
    )
