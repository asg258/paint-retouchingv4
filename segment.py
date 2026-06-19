"""
segment.py — Stage 2 of the wall recoloring pipeline.

Runs DeepLabV3+ (ResNet101 backbone) to produce a dense probability mask
representing how likely each pixel is to belong to a "wall" region.

This file is intentionally self-contained: it only depends on Stage 1's
output (a preprocessed RGB NumPy array) and returns a float32 NumPy mask.
Nothing here knows about wall colors, blending, or downstream stages.

--------------------------------------------------------------------------------
IMPORTANT NOTE ON CLASS LABELS
--------------------------------------------------------------------------------
The pretrained torchvision DeepLabV3 model was trained on COCO with the
Pascal VOC 21-class vocabulary:

    0: background      6: bus           12: dog         18: sofa
    1: aeroplane       7: car           13: horse        19: train
    2: bicycle         8: cat           14: motorbike    20: tv/monitor
    3: bird            9: chair         15: person
    4: boat           10: cow           16: potted plant
    5: bottle         11: dining table  17: sheep

There is NO explicit "wall" class. We use CLASS 0 (background) as the best
available proxy — it captures all image regions the model cannot assign to a
named foreground object, which in interior photos is primarily walls, ceilings,
and floors.

For a production system, swap in a model trained on ADE20K (150 classes,
class index 1 = "wall") or COCO-Stuff (182 classes, explicit wall class).
The rest of this code stays identical — only WALL_CLASS_INDEX changes.
--------------------------------------------------------------------------------
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
from pathlib import Path


# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------

# ImageNet normalisation stats — required because DeepLabV3's ResNet101
# backbone was initialised from ImageNet-pretrained weights. Every input image
# must be normalised to the same distribution the backbone originally saw.
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD:  list[float] = [0.229, 0.224, 0.225]

# Which class index to treat as "wall" after softmax (see module docstring).
WALL_CLASS_INDEX: int = 0

# Shorter-edge target before inference.
# Smaller  → faster, less VRAM, mask slightly blurrier after upsampling.
# Larger   → slower, more VRAM, mask preserves finer boundary detail.
# None     → use original image resolution (safe if VRAM is not a concern).
INFERENCE_SIZE: int | None = 512


# ---------------------------------------------------------------------------
# Step 1 — Model initialisation
# ---------------------------------------------------------------------------

def load_model(
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, torch.device]:
    """
    Load the pretrained DeepLabV3+ ResNet101 model and put it in eval mode.

    What is the CNN doing?
    ResNet101 is the backbone — a deep convolutional neural network that
    processes the image through hundreds of learned filters arranged in 101
    layers.  Each filter detects a specific pattern (edges, textures, object
    parts).  As the image passes through deeper layers the receptive field
    grows, so later layers "see" larger context (whole walls, not just edges).
    The DeepLab head on top aggregates multi-scale features using dilated
    (atrous) convolutions and, for every pixel, outputs one score per class.

    Args:
        device: Explicit device override. Auto-detects GPU if available.

    Returns:
        (model, device) — model in eval mode on the selected device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[segment] Loading DeepLabV3+ ResNet101 → {device} ...")
    print("[segment] (First run downloads ~250 MB of pretrained weights.)")

    # weights=DEFAULT fetches the best available COCO-pretrained checkpoint.
    model = models.segmentation.deeplabv3_resnet101(
        weights=DeepLabV3_ResNet101_Weights.DEFAULT
    )
    model.to(device)

    # eval() switches off Dropout and sets BatchNorm to use running statistics
    # instead of batch statistics — critical for deterministic inference.
    model.eval()

    print("[segment] Model loaded and ready.")
    return model, device


# ---------------------------------------------------------------------------
# Step 2 — Image preprocessing
# ---------------------------------------------------------------------------

def preprocess_for_model(
    image_rgb: np.ndarray,
    inference_size: int | None = INFERENCE_SIZE,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    Convert a uint8 RGB NumPy array into a normalised, batched float tensor.

    Why normalise?
    The model's weights encode expectations about the input distribution.
    If we feed raw [0, 255] pixels, the first layer activations are ~100×
    too large, completely breaking the pretrained feature detectors.
    Normalising to the same mean/std the model was trained with keeps the
    internal activations in the range the model expects.

    Why resize?
    DeepLabV3 is fully convolutional (no fixed-size FC layers), so it accepts
    any spatial size. However, the receptive field and anchor sizes were tuned
    around ~512 px inputs. Very small images under-use the multi-scale context;
    very large images are slow with diminishing accuracy returns.

    Args:
        image_rgb:      (H, W, 3) uint8 RGB array — Stage 1 output.
        inference_size: Shorter edge is scaled to this many pixels.
                        Pass None to keep original resolution.

    Returns:
        tensor:         (1, 3, H', W') float32 tensor, ready for the model.
        original_size:  (H, W) — saved so we can resize the mask back later.
    """
    original_size: tuple[int, int] = image_rgb.shape[:2]   # (H, W)

    pil_img = Image.fromarray(image_rgb)   # NumPy → PIL for torchvision

    transform_steps = []

    if inference_size is not None:
        h, w = original_size
        scale = inference_size / min(h, w)
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        transform_steps.append(transforms.Resize((new_h, new_w)))

    transform_steps += [
        # ToTensor: HWC uint8 [0,255] → CHW float32 [0.0, 1.0]
        transforms.ToTensor(),
        # Normalise: (pixel - mean) / std per channel
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    preprocess = transforms.Compose(transform_steps)
    tensor = preprocess(pil_img)         # (3, H', W')
    tensor = tensor.unsqueeze(0)         # (1, 3, H', W') — add batch dimension

    return tensor, original_size


# ---------------------------------------------------------------------------
# Step 3 — Inference
# ---------------------------------------------------------------------------

def run_inference(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Run a forward pass through DeepLabV3 and return raw output logits.

    What are logits?
    For every pixel in the (possibly resized) image, the network outputs one
    real-valued score per class — these are the logits (z_k for class k).
    They are raw, unbounded numbers:
      - z_k >> 0  → the model is strongly predicting class k
      - z_k ≈ 0   → the model is neutral
      - z_k << 0  → the model is strongly predicting NOT class k
    They are NOT yet probabilities because they don't sum to 1.

    Args:
        model:   Loaded DeepLabV3 in eval mode.
        tensor:  (1, 3, H', W') preprocessed tensor.
        device:  Device the model lives on.

    Returns:
        logits: (1, num_classes, H', W') float32 tensor.
    """
    tensor = tensor.to(device)

    # torch.no_grad() tells PyTorch not to build the computation graph.
    # At inference time we never call .backward(), so this saves memory
    # and speeds up the forward pass by ~15–30 %.
    with torch.no_grad():
        output = model(tensor)

    # torchvision segmentation models return an OrderedDict.
    # "out"  → main decoder output  (1, num_classes, H', W')  ← we want this
    # "aux"  → auxiliary loss head  (1, num_classes, H', W')  ← training only
    return output["out"]


# ---------------------------------------------------------------------------
# Step 4 — Logits → probability mask
# ---------------------------------------------------------------------------

def logits_to_wall_probability(
    logits: torch.Tensor,
    original_size: tuple[int, int],
    wall_class_index: int = WALL_CLASS_INDEX,
) -> np.ndarray:
    """
    Apply softmax, extract the wall-class channel, and upsample to the
    original image resolution.

    Why softmax?
    Softmax converts the raw logit vector at each pixel into a probability
    distribution over all classes:

        P(class = k | pixel x) = exp(z_k) / Σ_j exp(z_j)

    This guarantees:
      1. Every value is in (0, 1)      — valid probability range
      2. All class probabilities sum to 1 per pixel

    Why NOT argmax (a hard label)?
    argmax would give a single winning class per pixel — a binary 0/1 mask.
    That discards the model's confidence. A pixel at the sofa/wall boundary
    might score P(wall)=0.55, P(sofa)=0.45. Forcing it to 1 ignores how
    uncertain the model was. Downstream blending stages can use the soft
    probability to create smooth, natural transitions — especially important
    at depth edges where wall colour should fade into object colour.

    What is a dense probability mask?
    "Dense" = every pixel has a value, covering the full image grid (H × W).
    Contrast with sparse outputs like bounding boxes (just corners) or
    keypoints (just N points). Our output is a 2D map where each cell stores
    P(wall | that pixel) ∈ [0, 1].

    Args:
        logits:            (1, num_classes, H', W') raw model output.
        original_size:     (H, W) target size for the final mask.
        wall_class_index:  Class channel to extract after softmax.

    Returns:
        prob_mask: (H, W) float32 NumPy array with values in [0, 1].
    """
    # Apply softmax across the class dimension (dim=1).
    # Result: (1, num_classes, H', W') — each pixel sums to 1 across classes.
    probabilities = F.softmax(logits, dim=1)

    # Extract the channel for our wall proxy class → (H', W').
    wall_prob = probabilities[0, wall_class_index]   # squeeze batch dim

    # Upsample from inference resolution back to original image size.
    # bilinear interpolation gives smooth gradients at boundaries;
    # align_corners=False is the standard choice for feature maps.
    h_orig, w_orig = original_size
    wall_prob_upsampled = F.interpolate(
        wall_prob.unsqueeze(0).unsqueeze(0),   # (1, 1, H', W') for interpolate
        size=(h_orig, w_orig),
        mode="bilinear",
        align_corners=False,
    ).squeeze()    # back to (H, W)

    # Move to CPU and convert to NumPy. Values are guaranteed in [0, 1]
    # because softmax output is bounded and interpolation preserves range.
    return wall_prob_upsampled.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Public API — the single function the next pipeline stage calls
# ---------------------------------------------------------------------------

def get_deeplab_mask(
    image: np.ndarray,
    model: torch.nn.Module | None = None,
    device: torch.device | None = None,
) -> np.ndarray:
    """
    End-to-end Stage 2: preprocessed RGB image → wall probability mask.

    Accepts the uint8 RGB array produced by Stage 1 (preprocess.py) and
    returns a (H, W) float32 mask where each value is P(wall | pixel).

    Pass a pre-loaded model to avoid reloading weights on every call —
    this matters a lot in batch pipelines (loading takes ~2–5 s).

    Args:
        image:  (H, W, 3) uint8 RGB array.
        model:  Optional pre-loaded model in eval mode.
        device: Optional torch.device override.

    Returns:
        prob_mask: (H, W) float32 array, values ∈ [0, 1].
    """
    if model is None or device is None:
        model, device = load_model(device)

    tensor, original_size = preprocess_for_model(image)
    logits               = run_inference(model, tensor, device)
    prob_mask            = logits_to_wall_probability(logits, original_size)

    return prob_mask


# ---------------------------------------------------------------------------
# Debug visualisation
# ---------------------------------------------------------------------------

def visualize_mask(
    image_rgb: np.ndarray,
    prob_mask: np.ndarray,
    alpha: float = 0.55,
    save_path: str | Path | None = None,
) -> None:
    """
    Show (and optionally save) a side-by-side figure:
      Left  — original image
      Centre — raw probability heatmap (jet colormap, 0=blue, 1=red)
      Right  — heatmap blended over the original image

    Args:
        image_rgb:  (H, W, 3) uint8 RGB original image.
        prob_mask:  (H, W) float32 probability mask from get_deeplab_mask().
        alpha:      Heatmap opacity for the blended overlay (0–1).
        save_path:  If provided, saves the figure to this path.
    """
    # Build a coloured heatmap from the grayscale probability mask.
    heatmap_gray = (prob_mask * 255).astype(np.uint8)
    heatmap_bgr  = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
    heatmap_rgb  = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Blend heatmap over the original image for the overlay panel.
    overlay = cv2.addWeighted(image_rgb, 1 - alpha, heatmap_rgb, alpha, 0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Stage 2 — DeepLabV3 Wall Probability Mask\n"
        f"(class index {WALL_CLASS_INDEX} = background / wall proxy)",
        fontsize=13,
    )

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original (Stage 1 output)")
    axes[0].axis("off")

    im = axes[1].imshow(prob_mask, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("P(wall) heatmap")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (heatmap + original)")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"[segment] Visualisation saved → {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Quick test — python segment.py <image_path>
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from preprocess import preprocess_image

    if len(sys.argv) < 2:
        print("Usage:   python segment.py <image_path>")
        print("Example: python segment.py room.jpg")
        sys.exit(1)

    src_path = sys.argv[1]

    # Stage 1 — preprocess
    print("[main] Running Stage 1 preprocessing ...")
    preprocessed_rgb = preprocess_image(src_path)

    # Stage 2 — segment
    print("[main] Running Stage 2 segmentation ...")
    deeplab_model, torch_device = load_model()
    mask = get_deeplab_mask(preprocessed_rgb, model=deeplab_model, device=torch_device)

    print(f"[main] Mask shape : {mask.shape}")
    print(f"[main] Mask range : [{mask.min():.4f}, {mask.max():.4f}]")
    print(f"[main] Mean P(wall): {mask.mean():.4f}")

    # Visualise
    visualize_mask(
        preprocessed_rgb,
        mask,
        save_path=Path(src_path).stem + "_wall_mask.png",
    )
