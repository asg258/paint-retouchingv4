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
# Lower this if SAM generates too few foreground prompts.
FG_THRESHOLD: float = 0.70

# Pixels with P(wall) below this are treated as confident background.
BG_THRESHOLD: float = 0.30

# How many prompt points to sample from each region.
# More points = more context for SAM, but diminishing returns past ~10.
MAX_FG_POINTS: int = 8
MAX_BG_POINTS: int = 4

# Spacing (in pixels) for the grid used to sample prompt points.
# Smaller = denser sampling (more representative), larger = faster.
POINT_GRID_SPACING: int = 30


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
    coarse_mask: np.ndarray,
    fg_threshold: float = FG_THRESHOLD,
    bg_threshold: float = BG_THRESHOLD,
    max_fg_points: int = MAX_FG_POINTS,
    max_bg_points: int = MAX_BG_POINTS,
    grid_spacing: int = POINT_GRID_SPACING,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample foreground and background coordinate prompts from the coarse mask.

    What are prompts in SAM?
    SAM is a promptable model — instead of classifying every pixel like DeepLab,
    it responds to hints about WHERE the object of interest is. A prompt is
    simply a 2D coordinate plus a label:
        label = 1  → "this point is INSIDE the wall"
        label = 0  → "this point is OUTSIDE the wall"

    SAM uses these sparse hints together with the image embedding to decide
    exactly where the boundary should be drawn.

    Why derive prompts from the DeepLab mask?
    We already know roughly where the wall is — DeepLab told us. We just don't
    trust its edges. So we trust its high-confidence interior pixels (P > 0.7)
    as reliable "this is wall" evidence, and its high-confidence exterior pixels
    (P < 0.3) as reliable "this is not wall" evidence. We sample a handful of
    each to give SAM enough context.

    Sampling strategy — sparse grid:
    Rather than picking random pixels (which might cluster), we evaluate every
    point on a regular grid and then keep only the ones that fall in confident
    regions. This gives spatially spread-out prompts that cover the object area.

    Args:
        coarse_mask:   (H, W) float32 probability mask from DeepLab.
        fg_threshold:  Confidence above which a pixel is a foreground prompt.
        bg_threshold:  Confidence below which a pixel is a background prompt.
        max_fg_points: Maximum number of foreground prompts to return.
        max_bg_points: Maximum number of background prompts to return.
        grid_spacing:  Distance between grid sample points in pixels.

    Returns:
        point_coords: (N, 2) array of (x, y) prompt coordinates.
        point_labels: (N,)   array of 1 (foreground) or 0 (background).

    Raises:
        ValueError: if no foreground prompts can be found (mask too dim).
    """
    h, w = coarse_mask.shape

    # Build a sparse grid of candidate pixel locations.
    ys = np.arange(0, h, grid_spacing)
    xs = np.arange(0, w, grid_spacing)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_x = grid_x.ravel()
    grid_y = grid_y.ravel()

    # Read the mask value at each grid point.
    grid_vals = coarse_mask[grid_y, grid_x]

    # Foreground candidates: high-confidence wall pixels.
    fg_mask = grid_vals >= fg_threshold
    fg_xs, fg_ys = grid_x[fg_mask], grid_y[fg_mask]

    # Background candidates: high-confidence non-wall pixels.
    bg_mask = grid_vals <= bg_threshold
    bg_xs, bg_ys = grid_x[bg_mask], grid_y[bg_mask]

    if fg_xs.size == 0:
        raise ValueError(
            f"No foreground prompt points found (no pixels with mask > {fg_threshold}). "
            "Try lowering FG_THRESHOLD or check that the coarse mask is not empty."
        )

    # Subsample to the requested maximum counts.
    fg_idx = _subsample_indices(fg_xs.size, max_fg_points)
    bg_idx = _subsample_indices(bg_xs.size, max_bg_points)

    fg_coords = np.stack([fg_xs[fg_idx], fg_ys[fg_idx]], axis=1)   # (n_fg, 2)
    fg_labels = np.ones(len(fg_idx), dtype=int)

    if bg_xs.size > 0:
        bg_coords = np.stack([bg_xs[bg_idx], bg_ys[bg_idx]], axis=1)
        bg_labels = np.zeros(len(bg_idx), dtype=int)
        point_coords = np.concatenate([fg_coords, bg_coords], axis=0)
        point_labels = np.concatenate([fg_labels, bg_labels], axis=0)
    else:
        # No confident background found — proceed with foreground only.
        point_coords = fg_coords
        point_labels = fg_labels

    print(
        f"[refine] Prompts — {fg_labels.sum()} foreground, "
        f"{(point_labels == 0).sum()} background"
    )
    return point_coords, point_labels


def _subsample_indices(total: int, max_count: int) -> np.ndarray:
    """Return up to max_count evenly-spaced indices from [0, total)."""
    if total <= max_count:
        return np.arange(total)
    return np.linspace(0, total - 1, max_count, dtype=int)


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

    # Derive prompt points from the coarse mask.
    point_coords, point_labels = generate_sam_prompts(coarse_mask)

    # Run SAM — returns 3 candidate masks, their quality scores, and the
    # logits (we ignore logits here; they're useful for chained predictions).
    sam_masks, sam_scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,   # produce 3 candidates, pick the best one
    )
    # sam_masks:  (3, H, W) bool
    # sam_scores: (3,) float

    # Pick the candidate that best overlaps with the DeepLab mask.
    best_mask = _select_best_mask(sam_masks, sam_scores, coarse_mask)

    # Blend SAM's crisp binary boundary with DeepLab's soft probabilities.
    refined_mask = _combine_masks(best_mask, coarse_mask)

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
