"""
segment_mit.py — Stage 2: ADE20K wall segmentation using MIT CSAIL model.

Uses the MIT CSAIL semantic-segmentation-pytorch ResNet18-Dilated model
trained on ADE20K (150 classes). No HuggingFace, no mmcv, no Google Drive.
Weights downloaded from: http://sceneparsing.csail.mit.edu

ADE20K Class Index (0-based in model output):
    Index 0 = "wall" (dataset Idx 1, ratio 0.1576 — most common class)

WHY ADE20K OVER COCO/DeepLab?
------------------------------
DeepLab COCO class 0 = "background" includes EVERYTHING that isn't a
named COCO object: walls, tiles, floors, ceilings, mirrors, countertops.
In a bathroom, every pixel scores > 0.6 as "background".

ADE20K was designed specifically for indoor scene understanding.
It has 150 fine-grained classes including:
    0: wall (painted drywall)    3: floor
    8: windowpane                9: door
    10: ceiling                  17: cabinet;locker;wardrobe

The MIT CSAIL model trained on ADE20K outputs P(wall | pixel) that is
genuinely near 0 on tile, wood cabinets, and floor — the exact
discrimination DeepLab cannot make.

ARCHITECTURE
------------
Encoder: ResNet-18 Dilated (dilated conv rates [1,2,4] in last 2 stages)
Decoder: Pyramid Pooling Module (PPM) with deep supervision
Input:   any resolution, padded to multiple of 32
Output:  (1, 150, H/8, W/8) logits → upsample → softmax → extract class 0
"""

from __future__ import annotations

import sys
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Add the MIT CSAIL repo to sys.path so we can import their models.
_REPO_PATH = Path(__file__).parent / "mit_semseg_repo"
if str(_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(_REPO_PATH))

try:
    from mit_semseg.models import ModelBuilder, SegmentationModule
    from mit_semseg.utils import colorEncode
except ImportError:
    raise ImportError(
        "MIT CSAIL semantic-segmentation-pytorch not found.\n"
        f"Expected at: {_REPO_PATH}\n"
        "Run: git clone --depth 1 "
        "https://github.com/CSAILVision/semantic-segmentation-pytorch.git "
        "mit_semseg_repo"
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WEIGHTS_DIR:    Path = Path(__file__).parent / "mit_semseg_weights"
ENCODER_CKPT:   Path = WEIGHTS_DIR / "encoder_epoch_20.pth"
DECODER_CKPT:   Path = WEIGHTS_DIR / "decoder_epoch_20.pth"

ENCODER_ARCH:   str  = "resnet18dilated"
DECODER_ARCH:   str  = "ppm_deepsup"

# ADE20K class 0 (0-indexed) = "wall" (dataset Idx 1)
WALL_CLASS_IDX: int  = 0

# ImageNet normalisation — same as used during MIT CSAIL training.
IMAGENET_MEAN   = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD    = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Coverage threshold for diagnostic reporting.
COVERAGE_THRESHOLD: float = 0.4


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_mit_model(
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, torch.device]:
    """
    Load the MIT CSAIL ResNet18-Dilated ADE20K segmentation model.

    The model has two parts loaded separately:
        Encoder: ResNet-18 with dilated convolutions — extracts feature maps
        Decoder: Pyramid Pooling Module — aggregates multi-scale context
                 and produces per-pixel class logits

    Args:
        device: Torch device. Auto-detects GPU if available.

    Returns:
        (seg_module, device)  — SegmentationModule in eval mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not ENCODER_CKPT.exists() or not DECODER_CKPT.exists():
        raise FileNotFoundError(
            f"Model weights not found in {WEIGHTS_DIR}\n"
            "Run: python -c \""
            "import urllib.request, os; os.makedirs('mit_semseg_weights', exist_ok=True); "
            "[urllib.request.urlretrieve(f'http://sceneparsing.csail.mit.edu/model/pytorch/"
            "ade20k-resnet18dilated-ppm_deepsup/{f}', f'mit_semseg_weights/{f}') "
            "for f in ['encoder_epoch_20.pth', 'decoder_epoch_20.pth']]\""
        )

    print(f"[mit_seg] Loading ResNet18-Dilated ADE20K on {device} ...")

    net_encoder = ModelBuilder.build_encoder(
        arch=ENCODER_ARCH,
        fc_dim=512,
        weights=str(ENCODER_CKPT),
    )
    net_decoder = ModelBuilder.build_decoder(
        arch=DECODER_ARCH,
        fc_dim=512,
        num_class=150,
        weights=str(DECODER_CKPT),
        use_softmax=True,   # outputs probabilities directly
    )

    crit = torch.nn.NLLLoss(ignore_index=-1)
    seg_module = SegmentationModule(net_encoder, net_decoder, crit)
    seg_module.to(device)
    seg_module.eval()

    print(f"[mit_seg] Model loaded. ADE20K class {WALL_CLASS_IDX} = 'wall'.")
    return seg_module, device


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _preprocess(image_rgb: np.ndarray) -> torch.Tensor:
    """
    Normalise a uint8 RGB image to the distribution used during MIT training.

    Steps:
        1. Convert to float32, scale to [0, 1]
        2. Subtract ImageNet channel means
        3. Divide by ImageNet channel stds
        4. Convert HWC → CHW, add batch dim → (1, 3, H, W)
    """
    img = image_rgb.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def get_wall_mask(
    image:      np.ndarray,
    model:      torch.nn.Module | None = None,
    device:     torch.device | None = None,
) -> np.ndarray:
    """
    Run ADE20K segmentation and return the wall probability mask.

    Pipeline:
        1. Preprocess: normalise to ImageNet stats → (1, 3, H, W) tensor
        2. Forward pass: model outputs (1, 150, H/8, W/8) probabilities
        3. Upsample: bilinear → original (H, W) resolution
        4. Extract wall channel (index 0): P(wall | pixel)
        5. Return as float32 NumPy array

    Args:
        image:  (H, W, 3) uint8 RGB image from Stage 1.
        model:  Pre-loaded SegmentationModule. Loaded if None.
        device: Torch device.

    Returns:
        wall_mask: (H, W) float32, values in [0, 1].
    """
    if model is None:
        model, device = load_mit_model(device)
    if device is None:
        device = next(model.parameters()).device

    h, w   = image.shape[:2]
    tensor = _preprocess(image).to(device)

    with torch.no_grad():
        output = model({"img_data": tensor}, segSize=(h, w))
        # output shape: (1, 150, H, W) — probabilities (use_softmax=True)

    probs_np = output[0].cpu().numpy()           # (150, H, W)
    wall_prob = probs_np[WALL_CLASS_IDX].copy()  # P(wall) raw

    # ------------------------------------------------------------------
    # Fix 1: Multi-class exclusion using ADE20K's non-wall predictions.
    #
    # Diagnostic showed ADE20K correctly labels:
    #   mirror=27 (5% of pixels), cabinet=10 (20.5%), floor=3 (10.8%)
    # but incorrectly labels shower tile as wall=0 with P≈0.999.
    #
    # For all correctly-labelled non-wall surfaces, use ADE20K's own
    # confidence as a suppression signal:
    #   P_corrected = P(wall) × Π (1 − weight × P(non_wall_class))
    #
    # Each factor (1 − w × P_class) ≈ 1 when that class is not predicted,
    # and suppresses wall probability proportionally when it IS predicted.
    # Mirror (weight=1.2) and countertop (0.9) get the hardest suppression;
    # door/ceiling get softer suppression since they adjoin walls.
    # ------------------------------------------------------------------
    NON_WALL_EXCLUSIONS = {
        27: 1.2,   # mirror       — directly contradicts wall
        3:  0.9,   # floor        — clearly not a wall surface
        28: 0.9,   # rug          — never wall
        70: 0.9,   # countertop   — never wall
        10: 0.7,   # cabinet      — moderate (cabinet backs can look like wall)
        58: 0.8,   # screen/door  — mostly not wall
        5:  0.4,   # ceiling      — gentle (ceiling-wall junction exists)
        14: 0.4,   # door         — gentle (door frame adjoins wall)
    }

    exclusion = np.ones((h, w), dtype=np.float32)
    for cls_idx, weight in NON_WALL_EXCLUSIONS.items():
        # (1 - w × P_class) clipped to [0,1] so it cannot amplify
        exclusion *= np.clip(1.0 - weight * probs_np[cls_idx], 0.0, 1.0)

    P_wall_excl = (wall_prob * exclusion).clip(0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Fix 2: Texture-based pre-suppression for shower tile.
    #
    # ADE20K outputs P(wall)≈0.999 on bathroom tile — a complete model
    # failure. Since ADE20K cannot distinguish beige tile from the
    # original salmon wall, we must use visual analysis.
    #
    # compute_periodic_texture_penalty() detects structured, repeating
    # Laplacian patterns (grout lines) and returns low values for tile.
    # We apply it as a HARD gate here (before SAM prompts are generated)
    # so tile pixels NEVER become foreground prompts that tell SAM
    # "this tile is a wall". Hard threshold at 0.40: regions with
    # W_tile < 0.40 are zeroed, eliminating tile from the wall prior.
    # ------------------------------------------------------------------
    # --- Texture penalty (catches grout-heavy tile patterns) ---
    from mask_pipeline import compute_periodic_texture_penalty
    W_tile = compute_periodic_texture_penalty(image)
    W_tile_gate = np.where(W_tile < 0.40, 0.0, W_tile).astype(np.float32)

    # NOTE: Saturation-based suppression moved to mask_pipeline.py (Stage 3b)
    # as W_sat signal. Applying it here reduced M0 below the SAM FG threshold,
    # causing SAM to lose wall anchors and under-color the actual walls.
    # The texture gate alone acts as a pre-SAM filter; saturation discrimination
    # happens post-SAM where the mask values are independent of prompt selection.

    wall_prob_final = (P_wall_excl * W_tile_gate).clip(0.0, 1.0).astype(np.float32)

    coverage_raw   = float((wall_prob       > COVERAGE_THRESHOLD).mean() * 100)
    coverage_final = float((wall_prob_final > COVERAGE_THRESHOLD).mean() * 100)
    print(f"[mit_seg] Raw P(wall): {wall_prob.mean():.3f}  coverage={coverage_raw:.1f}%")
    print(f"[mit_seg] Corrected:   {wall_prob_final.mean():.3f}  coverage={coverage_final:.1f}%"
          f"  (excl+tile suppression applied)")

    return wall_prob_final


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_wall_mask(
    image_rgb:  np.ndarray,
    wall_mask:  np.ndarray,
    save_path:  str | Path | None = None,
) -> None:
    """Three-panel: original | wall probability heatmap | overlay."""
    heatmap_gray = (wall_mask * 255).astype(np.uint8)
    heatmap_bgr  = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
    heatmap_rgb  = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay      = cv2.addWeighted(image_rgb, 0.55, heatmap_rgb, 0.45, 0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"MIT CSAIL ADE20K — Wall Probability  "
        f"(class 0 = wall, ADE20K Idx 1)  "
        f"coverage={(wall_mask > 0.4).mean()*100:.1f}%",
        fontsize=12,
    )
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original"); axes[0].axis("off")
    im = axes[1].imshow(wall_mask, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("P(wall) — ADE20K ResNet18"); axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay"); axes[2].axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"[mit_seg] Saved: {save_path}")
    plt.show()
