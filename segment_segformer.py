"""
segment_segformer.py — Stage 2 (replacement for segment.py / DeepLab).

Uses SegFormer fine-tuned on ADE20K to produce a semantically correct
wall probability mask.

WHY ADE20K INSTEAD OF COCO?
----------------------------
DeepLabV3 (COCO VOC-21) has no "wall" class. It only has 20 named object
classes + class 0 (background). In indoor scenes the background class
captures walls, floors, ceilings, tiles, and anything else the model cannot
assign to a COCO foreground object. This means DeepLab's output for a
bathroom gives M ∈ [0.60, 1.0] for literally every pixel — there is no
discriminative signal between painted drywall and beige tile.

ADE20K is a 150-class indoor/outdoor segmentation dataset designed
specifically for scene understanding. It includes:
    Class 0 → "wall"         ← painted drywall
    Class 3 → "floor"        ← excluded
    Class 9 → "windowpane"   ← excluded
    Class 10 → "door"        ← excluded
    etc.

Crucially, ADE20K distinguishes "wall" (painted drywall) from
"wall — other materials" (tile, brick). Using class 0 directly gives us a
clean, semantically accurate wall prior that DeepLab simply cannot provide.

MODEL
-----
    nvidia/segformer-b2-finetuned-ade-512-512
    ~100 MB, downloaded automatically on first use via HuggingFace Hub.

    Architecture: SegFormer-B2
      - Mix-Transformer (MiT-B2) backbone: hierarchical ViT-style encoder
        that outputs multi-scale feature maps without positional embedding
        size constraints — handles any input resolution.
      - All-MLP decoder: lightweight decoder that fuses multi-scale features
        and outputs per-pixel class logits.
      - Trained on: ADE20K (150 classes, 20k training images)

    Why SegFormer over other ADE20K models?
      - Smaller than SegFormer-B4/B5 but still high accuracy
      - No fixed-resolution input constraint (unlike some CNN models)
      - Official HuggingFace integration → simple, stable API

ADE20K WALL CLASS INDEX
-----------------------
    Index 0 = "wall" in the model's id2label mapping.

    Verification:
        from transformers import SegformerForSemanticSegmentation
        m = SegformerForSemanticSegmentation.from_pretrained(...)
        print(m.config.id2label[0])   # → "wall"

    The model outputs logits of shape (1, 150, H/4, W/4).
    After softmax and extraction of channel 0, we get P(wall | pixel).
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

import os, ssl, warnings

# Enterprise SSL bypass — HuggingFace Hub ≥ 0.20 uses httpx internally,
# which ignores the standard REQUESTS_CA_BUNDLE env var.
# We patch httpx directly so both the processor and model downloads succeed
# on networks with SSL inspection proxies.
os.environ["CURL_CA_BUNDLE"]                  = ""
os.environ["REQUESTS_CA_BUNDLE"]              = ""
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
warnings.filterwarnings("ignore", message=".*SSL.*")
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

try:
    import httpx
    _orig_client_init = httpx.Client.__init__
    def _ssl_off_client(self, *a, **kw):
        kw["verify"] = False
        _orig_client_init(self, *a, **kw)
    httpx.Client.__init__ = _ssl_off_client

    _orig_async_init = httpx.AsyncClient.__init__
    def _ssl_off_async(self, *a, **kw):
        kw["verify"] = False
        _orig_async_init(self, *a, **kw)
    httpx.AsyncClient.__init__ = _ssl_off_async
except ImportError:
    pass   # httpx not installed — will use requests fallback

try:
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
except ImportError:
    raise ImportError(
        "transformers is required for SegFormer.\n"
        "Install with: pip install transformers"
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# HuggingFace model identifier. SegFormer-B2 on ADE20K.
# Larger variants (B4, B5) are more accurate but require more VRAM.
SEGFORMER_MODEL: str = "nvidia/segformer-b2-finetuned-ade-512-512"

# ADE20K class index 0 = "wall" (painted drywall).
# This is the ONLY class we extract from the 150-class output.
WALL_CLASS_INDEX: int = 0

# Confidence threshold for reporting: pixels above this count as "wall".
WALL_CONFIDENCE_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_segformer(
    model_id: str = SEGFORMER_MODEL,
    device:   torch.device | None = None,
) -> tuple[SegformerForSemanticSegmentation, object, torch.device]:
    """
    Load SegFormer-B2 fine-tuned on ADE20K.

    How SegFormer works:
        1. The Mix-Transformer (MiT) backbone processes the image through 4
           hierarchical stages, producing feature maps at 1/4, 1/8, 1/16,
           and 1/32 of the input resolution with increasing channel depth.
           Unlike standard ViTs, MiT uses overlapping patch embeddings and
           efficient self-attention — no fixed-resolution positional encoding.

        2. The All-MLP decoder takes these 4 feature maps, projects each to
           a unified embedding dimension, concatenates them, and applies a
           final linear layer to produce per-pixel logits.

        3. Output: (1, 150, H/4, W/4) — one logit per class per pixel.
           We upsample to the original image resolution and apply softmax.

    Args:
        model_id: HuggingFace model identifier.
        device:   Torch device; auto-detects GPU if available.

    Returns:
        (model, processor, device)
        model:     SegformerForSemanticSegmentation in eval mode.
        processor: AutoImageProcessor for image preprocessing.
        device:    The device the model lives on.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check for a local copy first (useful on networks that block HuggingFace).
    local_dir = Path(__file__).parent / "segformer_model"
    source    = str(local_dir) if local_dir.exists() else model_id

    if local_dir.exists():
        print(f"[segformer] Loading from local directory: {local_dir}")
    else:
        print(f"[segformer] Loading {model_id} on {device} ...")
        print("[segformer] (First run downloads ~100 MB of weights from HuggingFace.)")
        print("[segformer] If download fails (corporate network), manually download files from:")
        print("[segformer]   https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512/tree/main")
        print(f"[segformer] and save them to: {local_dir}")

    processor = AutoImageProcessor.from_pretrained(source)
    model     = SegformerForSemanticSegmentation.from_pretrained(source)
    model.to(device)
    model.eval()

    # Confirm the wall class index from the model's own label map.
    wall_label = model.config.id2label.get(WALL_CLASS_INDEX, "?")
    print(f"[segformer] Ready. Class {WALL_CLASS_INDEX} = '{wall_label}' (expect 'wall')")

    return model, processor, device


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def get_segformer_mask(
    image:     np.ndarray,
    model:     SegformerForSemanticSegmentation | None = None,
    processor: object | None = None,
    device:    torch.device | None = None,
) -> np.ndarray:
    """
    Run SegFormer on an RGB image and return the wall probability mask.

    Mathematical pipeline:
        1. Preprocess: the HuggingFace processor normalises pixel values
           to the distribution SegFormer was trained on (ImageNet-style
           mean/std applied channel-wise), then tensors are batched.

        2. Forward pass: model(inputs) → logits of shape (1, 150, H', W')
           where H' = H/4 and W' = W/4 (SegFormer downsamples 4×).

        3. Upsample: bilinear interpolation restores to original (H, W).

        4. Softmax across the 150-class dimension:
               P(class=k | pixel x) = exp(z_k) / Σ_{j=0}^{149} exp(z_j)
           Every pixel now has a valid probability distribution over 150 classes.

        5. Extract wall channel:
               M₀(x,y) = P(class=0 | x,y)
           This is the semantically correct wall probability — zero on tile,
           wood, floor, and fixtures because SegFormer was explicitly trained
           to distinguish them.

    Args:
        image:     (H, W, 3) uint8 RGB array from Stage 1.
        model:     Pre-loaded SegFormer model (load once, reuse).
        processor: Pre-loaded AutoImageProcessor.
        device:    Torch device.

    Returns:
        wall_mask: (H, W) float32 array, values in [0, 1].
                   Each value is P(wall | pixel) under the ADE20K taxonomy.
    """
    if model is None or processor is None:
        model, processor, device = load_segformer(device=device)
    if device is None:
        device = next(model.parameters()).device

    h_orig, w_orig = image.shape[:2]

    # Preprocess: normalise and tensorise for the SegFormer processor.
    pil_image = Image.fromarray(image)
    inputs    = processor(images=pil_image, return_tensors="pt")
    inputs    = {k: v.to(device) for k, v in inputs.items()}

    # Forward pass — no gradients needed at inference.
    with torch.no_grad():
        outputs = model(**inputs)

    # logits: (1, 150, H/4, W/4)
    logits = outputs.logits

    # Upsample back to original image resolution using bilinear interpolation.
    logits_up = F.interpolate(
        logits,
        size=(h_orig, w_orig),
        mode="bilinear",
        align_corners=False,
    )   # (1, 150, H, W)

    # Softmax: convert logits to class probability distribution per pixel.
    probs = F.softmax(logits_up, dim=1)   # (1, 150, H, W)

    # Extract wall class (index 0 = "wall" in ADE20K).
    wall_prob = probs[0, WALL_CLASS_INDEX]  # (H, W)

    wall_mask = wall_prob.cpu().numpy().astype(np.float32)

    # Diagnostic printout.
    wall_pct = float((wall_mask > WALL_CONFIDENCE_THRESHOLD).mean() * 100)
    print(f"[segformer] Wall mask: range=[{wall_mask.min():.3f}, {wall_mask.max():.3f}]  "
          f"mean={wall_mask.mean():.3f}  "
          f"coverage={wall_pct:.1f}% (pixels > {WALL_CONFIDENCE_THRESHOLD})")

    return wall_mask


# ---------------------------------------------------------------------------
# Debug visualisation
# ---------------------------------------------------------------------------

def visualize_segformer_mask(
    image_rgb: np.ndarray,
    wall_mask: np.ndarray,
    save_path: str | Path | None = None,
) -> None:
    """
    Three-panel: original | wall probability heatmap | overlay.
    """
    heatmap_gray = (wall_mask * 255).astype(np.uint8)
    heatmap_bgr  = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
    heatmap_rgb  = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay      = cv2.addWeighted(image_rgb, 0.55, heatmap_rgb, 0.45, 0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"SegFormer ADE20K — Wall Probability  "
        f"(class 0 = wall)  "
        f"coverage={(wall_mask > 0.5).mean()*100:.1f}%",
        fontsize=12,
    )
    axes[0].imshow(image_rgb);       axes[0].set_title("Original");            axes[0].axis("off")
    im = axes[1].imshow(wall_mask, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("P(wall) — ADE20K class 0"); axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    axes[2].imshow(overlay);         axes[2].set_title("Overlay");             axes[2].axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"[segformer] Visualisation saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from preprocess import preprocess_image

    if len(sys.argv) < 2:
        print("Usage: python segment_segformer.py <image_path>")
        sys.exit(1)

    src  = sys.argv[1]
    pre  = preprocess_image(src)
    sfm, proc, dev = load_segformer()
    mask = get_segformer_mask(pre, model=sfm, processor=proc, device=dev)

    visualize_segformer_mask(pre, mask,
                             save_path=Path(src).stem + "_segformer_wall.png")
