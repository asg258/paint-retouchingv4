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

    # Extract ADE20K class 0 = wall.
    wall_prob = output[0, WALL_CLASS_IDX].cpu().numpy().astype(np.float32)

    coverage = float((wall_prob > COVERAGE_THRESHOLD).mean() * 100)
    print(f"[mit_seg] Wall mask: range=[{wall_prob.min():.3f}, {wall_prob.max():.3f}]  "
          f"mean={wall_prob.mean():.3f}  "
          f"coverage={coverage:.1f}% (ADE20K class 0 = wall)")

    return wall_prob


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
