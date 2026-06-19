"""
evaluate.py — Stage 9 of the wall recoloring pipeline.

Quantitative and visual evaluation of the recolored output O(x,y).

Given:
    I       = original image
    O       = recolored output from Stage 8
    M_final = soft wall mask from Stage 5

Each metric answers a different question about whether the blending worked:

    Metric              Question it answers
    ──────────────────────────────────────────────────────────────────
    edge_error          Did recoloring break or smear visual edges?
    color_variance      Is the wall one uniform colour, or blotchy?
    leakage             Did colour spill onto furniture / outside the wall?
    brightness_error    Were shadows and highlights preserved?
    mean_wall_change    How strongly did the wall actually change colour?
    mean_outside_change How much did the non-wall region accidentally change?
    change_ratio        What fraction of all change happened INSIDE the wall?
    score               Weighted aggregate of all the above
    ──────────────────────────────────────────────────────────────────

All metrics are computed in float32.  No pipeline outputs are modified.
"""

from __future__ import annotations

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Score weights — must sum to 1.0
# ---------------------------------------------------------------------------

# How much each metric contributes to the final composite score.
# Tune these if you run an optimisation loop over pipeline parameters.
WEIGHT_EDGE:       float = 0.25   # edge integrity matters a lot visually
WEIGHT_VARIANCE:   float = 0.20   # uniform wall = realistic result
WEIGHT_LEAKAGE:    float = 0.25   # bleeding onto furniture is very noticeable
WEIGHT_BRIGHTNESS: float = 0.30   # lighting preservation is the hardest to fake

# Normalisation references — the value of each raw metric that is
# considered "completely bad" (= normalised score 0).  Values in [0,1]
# after normalisation: 0 = perfect, 1 = worst possible.
NORM_EDGE:       float = 15.0    # mean gradient diff (0-255 scale)
NORM_VARIANCE:   float = 800.0   # per-channel variance in the wall region
NORM_LEAKAGE:    float = 5.0     # mean pixel diff outside wall
NORM_BRIGHTNESS: float = 15.0    # mean V-channel diff (0-255 scale)

# Only pixels above this threshold count as "inside the wall".
MASK_THRESHOLD: float = 0.5

# Pixels below this threshold count as "outside the wall / protected".
LEAKAGE_THRESHOLD: float = 0.05


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """
    All evaluation metrics for one I → O transition.

    Every raw metric is also available in its normalised [0,1] form
    (attribute name ends in _norm) and as a contribution to the score.
    """
    # --- raw metrics ---
    edge_error:          float   # mean |∇I - ∇O|  lower = better
    color_variance:      float   # Var(O | wall)   lower = more uniform
    leakage:             float   # mean |O-I| outside wall  lower = better
    brightness_error:    float   # mean |V_I - V_O|  lower = better
    mean_wall_change:    float   # mean |O-I| inside wall  (informational)
    mean_outside_change: float   # mean |O-I| outside wall (same as leakage)
    change_ratio:        float   # sum_wall / (sum_total + ε)  closer to 1 = better

    # --- normalised [0,1] versions (0=perfect, 1=worst) ---
    edge_error_norm:       float = 0.0
    color_variance_norm:   float = 0.0
    leakage_norm:          float = 0.0
    brightness_error_norm: float = 0.0

    # --- composite score [0,1] (higher = better) ---
    score: float = 0.0

    def as_dict(self) -> dict:
        return {k: round(float(v), 5) for k, v in self.__dict__.items()}

    def __str__(self) -> str:
        return (
            f"Score:             {self.score:.4f}  (higher = better)\n"
            f"  Edge error:      {self.edge_error:.3f}  (norm {self.edge_error_norm:.3f})\n"
            f"  Color variance:  {self.color_variance:.2f}  (norm {self.color_variance_norm:.3f})\n"
            f"  Leakage:         {self.leakage:.4f}  (norm {self.leakage_norm:.3f})\n"
            f"  Brightness err:  {self.brightness_error:.3f}  (norm {self.brightness_error_norm:.3f})\n"
            f"  Wall change:     {self.mean_wall_change:.3f}\n"
            f"  Outside change:  {self.mean_outside_change:.4f}\n"
            f"  Change ratio:    {self.change_ratio:.4f}  (1.0 = all change inside wall)"
        )


# ---------------------------------------------------------------------------
# Main evaluation class
# ---------------------------------------------------------------------------

class EvaluationMetrics:
    """
    Evaluates the quality of a single I → O recoloring.

    Usage:
        evaluator = EvaluationMetrics()
        result    = evaluator.compute_all(original, output, mask)
        print(result)

    Parameters (all exposed at init time for optimisation loops):
        mask_threshold    — pixels above this count as "inside wall"
        leakage_threshold — pixels below this count as "outside wall"
        norm_*            — reference values for normalisation
        weight_*          — contribution of each metric to the score
    """

    def __init__(
        self,
        mask_threshold:    float = MASK_THRESHOLD,
        leakage_threshold: float = LEAKAGE_THRESHOLD,
        norm_edge:         float = NORM_EDGE,
        norm_variance:     float = NORM_VARIANCE,
        norm_leakage:      float = NORM_LEAKAGE,
        norm_brightness:   float = NORM_BRIGHTNESS,
        weight_edge:       float = WEIGHT_EDGE,
        weight_variance:   float = WEIGHT_VARIANCE,
        weight_leakage:    float = WEIGHT_LEAKAGE,
        weight_brightness: float = WEIGHT_BRIGHTNESS,
    ) -> None:
        self.mask_threshold    = mask_threshold
        self.leakage_threshold = leakage_threshold
        self.norm_edge         = norm_edge
        self.norm_variance     = norm_variance
        self.norm_leakage      = norm_leakage
        self.norm_brightness   = norm_brightness
        self.weight_edge       = weight_edge
        self.weight_variance   = weight_variance
        self.weight_leakage    = weight_leakage
        self.weight_brightness = weight_brightness

    # ------------------------------------------------------------------ #
    # Public entry point                                                   #
    # ------------------------------------------------------------------ #

    def compute_all(
        self,
        original: np.ndarray,
        output:   np.ndarray,
        mask:     np.ndarray,
    ) -> EvalResult:
        """
        Compute every metric and return an EvalResult.

        Args:
            original: (H, W, 3) uint8 RGB — the image before recoloring.
            output:   (H, W, 3) uint8 RGB — the image after recoloring.
            mask:     (H, W)    float32   — M_final from Stage 5.

        Returns:
            EvalResult with all raw + normalised metrics and the composite score.
        """
        I = original.astype(np.float32)
        O = output.astype(np.float32)
        M = np.clip(mask, 0.0, 1.0)

        wall_px    = M > self.mask_threshold
        outside_px = M < self.leakage_threshold

        # ── Metric 1: Edge artifact detection ─────────────────────────
        # WHY: The blending formula O = M*R + (1-M)*I should leave non-wall
        # edges unchanged and only add a soft colour gradient where M≈0.5.
        # If edges in O differ significantly from I it means the mask was
        # too aggressive (erased genuine object edges) or the colour layer
        # R introduced false edges (e.g. because M wasn't smooth enough).
        #
        # Formula:  E = mean( |∇I - ∇O| )
        # ∇ = Sobel gradient magnitude.  We compare them rather than just
        # computing ∇O because some increase in edge strength inside the
        # wall is normal (the new colour has its own contrast against the
        # floor). What we penalise is UNEXPECTED changes outside the wall.
        grad_I = self._gradient(original)
        grad_O = self._gradient(output)
        edge_error = float(np.mean(np.abs(grad_I - grad_O)))

        # ── Metric 2: Colour consistency (wall variance) ───────────────
        # WHY: A uniformly painted wall should look uniform in O. High
        # variance inside the wall means the blending was uneven — common
        # causes are a poorly-calibrated mask (so some wall pixels got
        # less recoloring than others) or the original wall having complex
        # texture that the colour layer didn't fully suppress.
        #
        # Formula:  V_wall = mean variance across RGB channels of O inside wall.
        if wall_px.sum() > 0:
            wall_vals = O[wall_px]   # (N, 3) float32
            color_variance = float(np.mean(np.var(wall_vals, axis=0)))
        else:
            color_variance = 0.0

        # ── Metric 3: Leakage detection ────────────────────────────────
        # WHY: The protection mask from Stage 5 should guarantee zero
        # change outside the wall.  Any non-zero difference in the clearly-
        # non-wall region (M < leakage_threshold) is a bleed artifact — the
        # colour from the wall has crept onto furniture or floor.
        #
        # Formula:  L = mean( |O(x,y) - I(x,y)| ) where M_final ≈ 0
        # We use the mean (not sum) so the metric is size-independent.
        diff = np.abs(O - I)   # (H, W, 3)
        if outside_px.sum() > 0:
            leakage = float(np.mean(diff[outside_px]))
        else:
            leakage = 0.0

        # ── Metric 4: Brightness preservation ─────────────────────────
        # WHY: Lighting (the S in I = R*S) must remain unchanged after
        # recoloring — only the reflectance R should change.  HSV Value
        # channel encodes perceived brightness, so comparing V_I and V_O
        # measures how faithfully Stage 8 preserved the shading.
        #
        # Formula:  B = mean( |V_I - V_O| )
        brightness_error = self._brightness_error(original, output)

        # ── Metric 5+6: Overlay difference analysis ────────────────────
        # WHY: A well-executed recoloring should cause LARGE changes inside
        # the wall (that's the point) and NEAR-ZERO changes outside.
        # These two numbers together tell you HOW WELL the mask confined
        # the recoloring to where it should be.
        diff_mean = np.mean(diff, axis=2)   # (H, W) — average across channels
        mean_wall_change    = float(np.mean(diff_mean[wall_px]))    if wall_px.sum()    > 0 else 0.0
        mean_outside_change = float(np.mean(diff_mean[outside_px])) if outside_px.sum() > 0 else 0.0

        # ── Metric 7: Mask-aligned change ratio ───────────────────────
        # WHY: ratio = sum_wall / (sum_total + ε) ranges from 0 to 1.
        # Ratio ≈ 1 means virtually all pixel change happened inside the
        # wall — the mask was well-aligned with the actual changes.
        # Ratio << 1 means a lot of change leaked outside the wall.
        sum_wall  = float(diff_mean[wall_px].sum())    if wall_px.sum()  > 0 else 0.0
        sum_total = float(diff_mean.sum())
        change_ratio = sum_wall / (sum_total + 1e-6)

        # ── Normalise and compute composite score ─────────────────────
        # Each raw metric is mapped to [0,1] by dividing by its reference
        # "bad" value and clipping. Score = 1 - weighted average of norms.
        e_n  = float(np.clip(edge_error       / self.norm_edge,      0.0, 1.0))
        v_n  = float(np.clip(color_variance   / self.norm_variance,  0.0, 1.0))
        l_n  = float(np.clip(leakage          / self.norm_leakage,   0.0, 1.0))
        b_n  = float(np.clip(brightness_error / self.norm_brightness, 0.0, 1.0))

        # Composite score — higher is better.
        # S = w1*(1-E_n) + w2*(1-V_n) + w3*(1-L_n) + w4*(1-B_n)
        score = (
            self.weight_edge       * (1.0 - e_n) +
            self.weight_variance   * (1.0 - v_n) +
            self.weight_leakage    * (1.0 - l_n) +
            self.weight_brightness * (1.0 - b_n)
        )

        return EvalResult(
            edge_error=edge_error,
            color_variance=color_variance,
            leakage=leakage,
            brightness_error=brightness_error,
            mean_wall_change=mean_wall_change,
            mean_outside_change=mean_outside_change,
            change_ratio=change_ratio,
            edge_error_norm=e_n,
            color_variance_norm=v_n,
            leakage_norm=l_n,
            brightness_error_norm=b_n,
            score=score,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _gradient(image: np.ndarray) -> np.ndarray:
        """
        Compute per-pixel gradient magnitude using Sobel operators.

        Sobel is preferred over Canny here because it returns a continuous
        float magnitude rather than a binary edge map — this lets us compute
        meaningful mean differences between ∇I and ∇O.

        The gradient is computed on the luminance channel (greyscale) so it
        captures structural edges without being confused by the colour change.
        """
        grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gx   = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
        gy   = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx**2 + gy**2)   # (H, W) float32

    @staticmethod
    def _brightness_error(original: np.ndarray, output: np.ndarray) -> float:
        """
        Mean absolute difference in the HSV V (brightness) channel.

        HSV V directly encodes how much light is hitting the surface — it is
        the shading component S in the I=R*S model.  A low brightness error
        means the recoloring preserved the original lighting faithfully.
        """
        def to_v(img: np.ndarray) -> np.ndarray:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
            return hsv[:, :, 2]   # (H, W) V channel

        return float(np.mean(np.abs(to_v(original) - to_v(output))))


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_evaluation(
    original: np.ndarray,
    output:   np.ndarray,
    mask:     np.ndarray,
    result:   EvalResult,
    save_path: str | Path | None = None,
) -> None:
    """
    Six-panel diagnostic figure.

    Row layout:
        [0] Original       [1] Recolored output    [2] Difference heatmap
        [3] ∇I edges       [4] ∇O edges             [5] |∇I - ∇O| artifact map

    The difference heatmap uses a hot colormap so faint leakage is visible.
    The edge artifact map shows exactly where edges changed after recoloring.

    Args:
        original:  (H, W, 3) uint8 RGB — pre-recoloring.
        output:    (H, W, 3) uint8 RGB — post-recoloring.
        mask:      (H, W)    float32   — M_final.
        result:    EvalResult from compute_all().
        save_path: If set, saves figure here.
    """
    ev = EvaluationMetrics()
    diff        = np.mean(np.abs(output.astype(float) - original.astype(float)), axis=2)
    grad_I      = ev._gradient(original)
    grad_O      = ev._gradient(output)
    grad_diff   = np.abs(grad_I - grad_O)

    # Wall overlay: green tint shows where M_final is active.
    overlay = original.copy().astype(float)
    overlay[:,:,1] = np.clip(overlay[:,:,1] + mask * 60, 0, 255)
    overlay = overlay.astype(np.uint8)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        f"Stage 9 — Evaluation   Score: {result.score:.4f}  |  "
        f"Leakage: {result.leakage:.4f}  |  "
        f"Brightness err: {result.brightness_error:.3f}  |  "
        f"Change ratio: {result.change_ratio:.3f}",
        fontsize=12,
    )

    axes[0,0].imshow(original)
    axes[0,0].set_title("I(x,y)  Original")
    axes[0,0].axis("off")

    axes[0,1].imshow(output)
    axes[0,1].set_title("O(x,y)  Recolored output")
    axes[0,1].axis("off")

    im2 = axes[0,2].imshow(diff, cmap="hot", vmin=0, vmax=60)
    axes[0,2].set_title(
        f"Difference |O - I|\n"
        f"wall change={result.mean_wall_change:.2f}  "
        f"outside={result.mean_outside_change:.4f}"
    )
    axes[0,2].axis("off")
    plt.colorbar(im2, ax=axes[0,2], fraction=0.046, pad=0.04)

    axes[1,0].imshow(overlay)
    axes[1,0].set_title(
        f"Wall mask overlay\n"
        f"coverage={(mask > 0.5).mean()*100:.1f}%  "
        f"variance={result.color_variance:.1f}"
    )
    axes[1,0].axis("off")

    axes[1,1].imshow(grad_I, cmap="gray")
    axes[1,1].set_title("∇I  Original edges (Sobel)")
    axes[1,1].axis("off")

    axes[1,2].imshow(grad_diff, cmap="hot")
    axes[1,2].set_title(
        f"|∇I - ∇O|  Edge artifact map\n"
        f"edge error={result.edge_error:.3f}"
    )
    axes[1,2].axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"[evaluate] Visualisation saved: {save_path}")

    plt.show()


def print_report(result: EvalResult, color_name: str = "") -> None:
    """Print a formatted evaluation report to stdout."""
    sep = "-" * 52
    header = f"Evaluation Report{': ' + color_name if color_name else ''}"
    print(f"\n{sep}")
    print(f"  {header}")
    print(sep)
    print(result)
    print(sep)
    # Qualitative interpretation
    if result.score >= 0.85:
        print("  Quality: EXCELLENT")
    elif result.score >= 0.70:
        print("  Quality: GOOD")
    elif result.score >= 0.55:
        print("  Quality: FAIR — check leakage and brightness")
    else:
        print("  Quality: POOR — review mask thresholds and blend mode")
    if result.leakage > 2.0:
        print("  WARNING: High leakage — colour may have bled outside wall.")
    if result.change_ratio < 0.80:
        print("  WARNING: Less than 80% of change is inside the wall.")
    if result.brightness_error > 10.0:
        print("  WARNING: Significant brightness shift — consider HSV blend mode.")
    print(sep)


# ---------------------------------------------------------------------------
# Quick test — python evaluate.py <image_path> [color_code] [--hsv]
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
    from blend        import blend_images

    if len(sys.argv) < 2:
        print("Usage:   python evaluate.py <image_path> [color_code] [--hsv]")
        print("Example: python evaluate.py room.jpg 8001-1G --hsv")
        sys.exit(1)

    src         = sys.argv[1]
    stem        = Path(src).stem
    use_hsv     = "--hsv" in sys.argv
    blend_mode  = "hsv" if use_hsv else "rgb"
    flags       = {"--hsv"}
    color_parts = [a for a in sys.argv[2:] if a not in flags]
    color_query = " ".join(color_parts) if color_parts else "8001-1G"

    color = get_color(color_query)
    if color is None:
        print(f"Color '{color_query}' not found."); sys.exit(1)

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
    print(f"[main] Stage 8: blending (mode={blend_mode}) ...")
    output = blend_images(preprocessed, layer, final_mask, mode=blend_mode)
    print("[main] Stage 9: evaluation ...")
    evaluator = EvaluationMetrics()
    result    = evaluator.compute_all(preprocessed, output, final_mask)

    print_report(result, color_name=f"{color.code} {color.name}")

    visualize_evaluation(
        preprocessed, output, final_mask, result,
        save_path=f"{stem}_{color.code}_evaluation.png",
    )
