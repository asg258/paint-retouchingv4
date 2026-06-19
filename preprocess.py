"""
preprocess.py — Stage 1 of the wall recoloring pipeline.

Loads an image and prepares it for downstream segmentation and recoloring
by enhancing contrast and optionally boosting color saturation. All of this
happens in LAB color space, which separates luminance (L) from color (A, B),
making it much easier to touch one without accidentally breaking the other.
"""

import cv2
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Tunable parameters — change these without touching the logic below.
# ---------------------------------------------------------------------------

# CLAHE: limits how aggressively contrast is boosted in any local tile.
# Higher = more dramatic; 2.0–4.0 is usually the sweet spot for interiors.
CLAHE_CLIP_LIMIT: float = 2.0

# CLAHE: how many tiles the image is divided into for local equalization.
# (8, 8) works well for typical room photos (1–4 MP).
CLAHE_TILE_GRID_SIZE: tuple[int, int] = (8, 8)

# Saturation: images whose average saturation falls below this value are
# considered "washed out" and receive the boost below.  Range: 0–1.
SATURATION_THRESHOLD: float = 0.15

# Saturation: how much to scale the A and B channels when boosting.
# 1.2 = 20% boost — subtle enough to look natural on wall photos.
SATURATION_SCALE: float = 1.2


# ---------------------------------------------------------------------------
# Core preprocessing function
# ---------------------------------------------------------------------------

def preprocess_image(image_path: str | Path) -> np.ndarray:
    """
    Load an image from disk, enhance its contrast, and conditionally boost
    saturation.  Returns a processed uint8 RGB image ready for the next
    pipeline stage.

    Args:
        image_path: Path to the source image (JPEG, PNG, etc.).

    Returns:
        Processed image as a NumPy array in RGB format, shape (H, W, 3),
        dtype uint8.

    Raises:
        FileNotFoundError: if the path does not exist.
        ValueError: if OpenCV cannot decode the file.
    """
    image_path = Path(image_path)

    # ------------------------------------------------------------------
    # Step 1 — Load the image from disk.
    # cv2.imread returns BGR by default, so we immediately flip to RGB so
    # the rest of the pipeline works in the expected channel order.
    # ------------------------------------------------------------------
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError(f"OpenCV could not decode the image at: {image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------------
    # Step 2 — Convert RGB → LAB.
    # LAB encodes:  L = lightness,  A = green↔red,  B = blue↔yellow.
    # cv2 scales the channels to [0, 255] uint8 internally.
    # ------------------------------------------------------------------
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

    # Split into individual channels so we can operate on them separately.
    l_channel, a_channel, b_channel = cv2.split(lab)

    # ------------------------------------------------------------------
    # Step 3 — Apply CLAHE to the L (lightness) channel.
    # CLAHE is like histogram equalization but local: it boosts contrast
    # within small tiles rather than globally, so it doesn't blow out
    # bright areas or crush dark ones.
    # ------------------------------------------------------------------
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE,
    )
    l_channel = clahe.apply(l_channel)

    # ------------------------------------------------------------------
    # Step 4 — Compute average saturation from the A and B channels.
    # In LAB space, saturation is the distance from the neutral gray axis.
    # We normalize to [0, 1] by dividing by 127 (half the uint8 range).
    # ------------------------------------------------------------------
    # Shift A/B from [0, 255] to [-127, 128] so neutral gray is at 0.
    a_centered = a_channel.astype(np.float32) - 128.0
    b_centered = b_channel.astype(np.float32) - 128.0

    # Per-pixel saturation = Euclidean distance in the A–B plane.
    saturation_map = np.sqrt(a_centered ** 2 + b_centered ** 2)

    # Normalize so 1.0 represents the maximum possible saturation (≈127√2).
    max_possible_saturation = 127.0 * np.sqrt(2)
    avg_saturation = float(saturation_map.mean() / max_possible_saturation)

    # ------------------------------------------------------------------
    # Step 5 — Conditionally boost A and B channels.
    # Only runs when the image looks washed out (avg saturation below the
    # threshold).  We scale around the neutral midpoint (128) so gray
    # pixels stay gray.
    # ------------------------------------------------------------------
    if avg_saturation < SATURATION_THRESHOLD:
        a_boosted = _scale_channel_around_midpoint(a_channel, SATURATION_SCALE)
        b_boosted = _scale_channel_around_midpoint(b_channel, SATURATION_SCALE)
        print(
            f"[preprocess] Low saturation detected ({avg_saturation:.3f} < "
            f"{SATURATION_THRESHOLD}).  Boosting A/B channels by "
            f"{SATURATION_SCALE}×."
        )
    else:
        a_boosted = a_channel
        b_boosted = b_channel
        print(
            f"[preprocess] Saturation OK ({avg_saturation:.3f}).  "
            "Skipping A/B boost."
        )

    # ------------------------------------------------------------------
    # Step 6 — Merge channels and convert LAB → RGB.
    # ------------------------------------------------------------------
    lab_processed = cv2.merge([l_channel, a_boosted, b_boosted])
    rgb_out = cv2.cvtColor(lab_processed, cv2.COLOR_LAB2RGB)

    return rgb_out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _scale_channel_around_midpoint(
    channel: np.ndarray,
    scale: float,
    midpoint: int = 128,
) -> np.ndarray:
    """
    Scale a uint8 channel around a midpoint and clip back to [0, 255].

    Scaling around 128 (the neutral gray value in cv2's LAB encoding)
    means that gray pixels are unchanged while colored pixels move further
    from gray — i.e., saturation increases without shifting the hue.
    """
    shifted = channel.astype(np.float32) - midpoint
    scaled = shifted * scale
    return np.clip(scaled + midpoint, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Quick visual test — run this file directly to see before/after
# ---------------------------------------------------------------------------

def _show_before_after(original: np.ndarray, processed: np.ndarray) -> None:
    """Display a side-by-side before/after comparison using OpenCV."""
    # Add simple text labels so we know which side is which.
    def _label(img: np.ndarray, text: str) -> np.ndarray:
        out = img.copy()
        cv2.putText(
            out, text,
            org=(10, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color=(255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        # Dark drop-shadow so the label is readable on bright images too.
        cv2.putText(
            out, text,
            org=(12, 32),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color=(0, 0, 0),
            thickness=4,
            lineType=cv2.LINE_AA,
        )
        return out

    before_bgr = cv2.cvtColor(_label(original, "Before"), cv2.COLOR_RGB2BGR)
    after_bgr = cv2.cvtColor(_label(processed, "After"), cv2.COLOR_RGB2BGR)

    # Resize to the same height in case the images differ in size.
    h = min(before_bgr.shape[0], after_bgr.shape[0])
    w = min(before_bgr.shape[1], after_bgr.shape[1])
    before_bgr = cv2.resize(before_bgr, (w, h))
    after_bgr = cv2.resize(after_bgr, (w, h))

    comparison = np.hstack([before_bgr, after_bgr])
    cv2.imshow("Preprocessing — Before / After (press any key to close)", comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <image_path>")
        print("Example: python preprocess.py room.jpg")
        sys.exit(1)

    src_path = sys.argv[1]
    print(f"[preprocess] Loading: {src_path}")

    original_bgr = cv2.imread(src_path)
    if original_bgr is None:
        print(f"Error: could not load '{src_path}'")
        sys.exit(1)
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

    result_rgb = preprocess_image(src_path)

    print(f"[preprocess] Done. Output shape: {result_rgb.shape}, dtype: {result_rgb.dtype}")

    _show_before_after(original_rgb, result_rgb)
