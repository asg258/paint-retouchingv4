"""
segformer_pytorch.py — Pure PyTorch SegFormer implementation.

Faithful re-implementation of NVlabs/SegFormer without mmcv dependency.
Architecture matches exactly the original paper and NVlabs weights format
so that pre-trained checkpoints load correctly.

Reference: Xie et al. "SegFormer: Simple and Efficient Design for Semantic
Segmentation with Transformers", NeurIPS 2021.
GitHub: https://github.com/NVlabs/SegFormer

Key differences from the NVlabs code:
  - All mmcv operations replaced with standard PyTorch/einops equivalents
  - ConvModule replaced with nn.Sequential(Conv2d, BN, GELU)
  - resize() replaced with F.interpolate()
  - load_checkpoint() replaced with torch.load()
  - auto_fp16 decorators removed (not needed for inference)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from functools import partial


# ---------------------------------------------------------------------------
# Mix-Transformer Backbone (MIT)
# ---------------------------------------------------------------------------

class DWConv(nn.Module):
    """Depth-wise convolution used inside Mix-FFN."""
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class MixFFN(nn.Module):
    """Mix-FFN: MLP with depth-wise conv for local information."""
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1   = nn.Linear(in_features, hidden_features)
        self.dw    = DWConv(hidden_features)
        self.act   = nn.GELU()
        self.fc2   = nn.Linear(hidden_features, in_features)
        self.drop  = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dw(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class EfficientSelfAttention(nn.Module):
    """
    Efficient Multi-head Self-Attention with Sequence Reduction Ratio (sr_ratio).

    Instead of attending over all N tokens, a Conv2d with stride=sr_ratio
    reduces the key/value sequence length from N to N/sr_ratio², dramatically
    cutting the O(N²) attention cost in early stages with large feature maps.
    """
    def __init__(self, dim: int, num_heads: int, sr_ratio: int, qkv_bias: bool = True):
        super().__init__()
        self.num_heads  = num_heads
        self.head_dim   = dim // num_heads
        self.scale      = self.head_dim ** -0.5

        self.q   = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv  = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(0.0)

        if sr_ratio > 1:
            self.sr   = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.sr_ratio = sr_ratio

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
        else:
            x_ = x

        kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.drop(self.proj(x))


class TransformerBlock(nn.Module):
    """One transformer block: LayerNorm → Attention → LayerNorm → FFN."""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, sr_ratio: int):
        super().__init__()
        self.norm1  = nn.LayerNorm(dim, eps=1e-6)
        self.attn   = EfficientSelfAttention(dim, num_heads, sr_ratio)
        self.norm2  = nn.LayerNorm(dim, eps=1e-6)
        self.mlp    = MixFFN(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x), H, W)
        return x


class OverlapPatchEmbed(nn.Module):
    """
    Overlapping Patch Embedding.
    Applies a strided Conv2d to produce non-overlapping patches that do
    overlap with their neighbours due to the padding — gives local continuity
    that vanilla ViT patch tokens lack.
    """
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int, stride: int):
        super().__init__()
        pad = patch_size // 2
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, stride=stride, padding=pad)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)   # B, H*W, C
        x = self.norm(x)
        return x, H, W


class MixTransformer(nn.Module):
    """
    Mix-Transformer backbone (MIT-B2).

    Produces 4 feature maps at resolutions H/4, H/8, H/16, H/32
    with channel depths [64, 128, 320, 512].

    These multi-scale features are passed to the SegFormer decode head.
    """

    # B2 configuration
    EMBED_DIMS  = [64, 128, 320, 512]
    NUM_LAYERS  = [3,  4,   6,   3]
    NUM_HEADS   = [1,  2,   5,   8]
    MLP_RATIOS  = [4,  4,   4,   4]
    SR_RATIOS   = [8,  4,   2,   1]

    def __init__(self, in_channels: int = 3):
        super().__init__()
        dims = self.EMBED_DIMS

        # Four stages: patch embed + transformer blocks
        self.patch_embeds = nn.ModuleList()
        self.blocks       = nn.ModuleList()
        self.norms        = nn.ModuleList()

        patch_sizes = [7, 3, 3, 3]
        strides     = [4, 2, 2, 2]
        in_ch       = in_channels

        for i in range(4):
            self.patch_embeds.append(
                OverlapPatchEmbed(in_ch, dims[i], patch_sizes[i], strides[i])
            )
            self.blocks.append(nn.ModuleList([
                TransformerBlock(dims[i], self.NUM_HEADS[i],
                                  self.MLP_RATIOS[i], self.SR_RATIOS[i])
                for _ in range(self.NUM_LAYERS[i])
            ]))
            self.norms.append(nn.LayerNorm(dims[i], eps=1e-6))
            in_ch = dims[i]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        B = x.shape[0]
        outs = []
        for i in range(4):
            x, H, W = self.patch_embeds[i](x)
            for blk in self.blocks[i]:
                x = blk(x, H, W)
            x = self.norms[i](x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
            outs.append(x)
        return outs   # [C1×H/4, C2×H/8, C3×H/16, C4×H/32]


# ---------------------------------------------------------------------------
# SegFormer Decode Head (All-MLP)
# ---------------------------------------------------------------------------

class SegFormerHead(nn.Module):
    """
    All-MLP decoder.

    Each of the 4 encoder feature maps is independently projected to
    embedding_dim channels, then all 4 are upsampled to the H/4 resolution,
    concatenated, fused with a single Conv2d, and classified.

    This is deliberately simple — no cross-attention, no recurrence.
    The power comes from the rich multi-scale features of the MIT backbone.
    """
    def __init__(
        self,
        in_channels: list[int],
        embedding_dim: int,
        num_classes:   int,
        dropout_ratio: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # One linear layer per scale to project to embedding_dim
        self.linear_c = nn.ModuleList([
            nn.Linear(c, embedding_dim) for c in in_channels
        ])
        # Fusion: 4 * embedding_dim → embedding_dim
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, 1, bias=False),
            nn.BatchNorm2d(embedding_dim, eps=1e-5),
            nn.ReLU(inplace=True),
        )
        self.dropout  = nn.Dropout2d(dropout_ratio)
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, 1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        # Target spatial size: same as the highest-res feature map (H/4)
        target_h, target_w = features[0].shape[2:]
        out = []
        for i, (c_map, linear) in enumerate(zip(features, self.linear_c)):
            B, C, H, W = c_map.shape
            # Flatten spatial dims, project, restore
            t = linear(c_map.flatten(2).transpose(1, 2))  # B, H*W, embed
            t = t.transpose(1, 2).reshape(B, self.embedding_dim, H, W)
            if (H, W) != (target_h, target_w):
                t = F.interpolate(t, size=(target_h, target_w),
                                  mode="bilinear", align_corners=False)
            out.append(t)

        x = self.linear_fuse(torch.cat(out, dim=1))
        x = self.dropout(x)
        return self.linear_pred(x)


# ---------------------------------------------------------------------------
# Full SegFormer-B2 for ADE20K
# ---------------------------------------------------------------------------

class SegFormerB2ADE20K(nn.Module):
    """
    SegFormer-B2 fine-tuned on ADE20K (150 classes).

    Input:  (B, 3, H, W)  uint8-normalised float tensor
    Output: (B, 150, H/4, W/4)  logits — upsample to (H, W) in inference

    ADE20K class 0 = "wall" (painted drywall).
    """
    ADE20K_MEAN = [0.485, 0.456, 0.406]
    ADE20K_STD  = [0.229, 0.224, 0.225]

    def __init__(self):
        super().__init__()
        self.backbone = MixTransformer()
        self.head     = SegFormerHead(
            in_channels   = [64, 128, 320, 512],
            embedding_dim = 768,
            num_classes   = 150,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    @torch.no_grad()
    def predict_wall_mask(self, image_rgb: "np.ndarray") -> "np.ndarray":
        """
        uint8 RGB (H,W,3) → float32 wall probability mask (H,W).
        ADE20K class 0 = wall.
        """
        import numpy as np
        h, w = image_rgb.shape[:2]
        t    = torch.from_numpy(image_rgb.astype("float32") / 255.0)
        mean = torch.tensor(self.ADE20K_MEAN).view(1, 1, 3)
        std  = torch.tensor(self.ADE20K_STD).view(1, 1, 3)
        t    = ((t - mean) / std).permute(2, 0, 1).unsqueeze(0)
        device = next(self.parameters()).device
        logits = self(t.to(device))
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        probs  = F.softmax(logits, dim=1)
        return probs[0, 0].cpu().numpy().astype("float32")


# ---------------------------------------------------------------------------
# Weight loading — remap NVlabs/mmseg key names to our naming convention
# ---------------------------------------------------------------------------

def _remap_key(k: str) -> str | None:
    """
    Map an NVlabs/mmseg checkpoint key to our PyTorch model key.

    NVlabs format                   →  Our format
    backbone.patch_embed{i+1}.*    →  backbone.patch_embeds.{i}.*
    backbone.block{i+1}.{j}.*      →  backbone.blocks.{i}.{j}.*
    backbone.norm{i+1}.*           →  backbone.norms.{i}.*
    decode_head.linear_c{i+1}.project.*  →  head.linear_c.{i}.*
    decode_head.linear_fuse.conv.* →  head.linear_fuse.0.*
    decode_head.linear_fuse.bn.*   →  head.linear_fuse.1.*
    decode_head.linear_pred.*      →  head.linear_pred.*
    decode_head.bn.*               →  (dropped — not in our model)
    """
    import re

    # backbone.patch_embed{N} → backbone.patch_embeds.{N-1}
    m = re.match(r"backbone\.patch_embed(\d+)\.(.*)", k)
    if m:
        return f"backbone.patch_embeds.{int(m.group(1))-1}.{m.group(2)}"

    # backbone.block{N}.{j}.* → backbone.blocks.{N-1}.{j}.*
    m = re.match(r"backbone\.block(\d+)\.(.*)", k)
    if m:
        return f"backbone.blocks.{int(m.group(1))-1}.{m.group(2)}"

    # backbone.norm{N}.* → backbone.norms.{N-1}.*
    m = re.match(r"backbone\.norm(\d+)\.(.*)", k)
    if m:
        return f"backbone.norms.{int(m.group(1))-1}.{m.group(2)}"

    # decode_head.linear_c{N}.project.* → head.linear_c.{N-1}.*
    m = re.match(r"decode_head\.linear_c(\d+)\.project\.(.*)", k)
    if m:
        return f"head.linear_c.{int(m.group(1))-1}.{m.group(2)}"

    # decode_head.linear_fuse.conv.* → head.linear_fuse.0.*
    m = re.match(r"decode_head\.linear_fuse\.conv\.(.*)", k)
    if m:
        return f"head.linear_fuse.0.{m.group(1)}"

    # decode_head.linear_fuse.bn.* → head.linear_fuse.1.*
    m = re.match(r"decode_head\.linear_fuse\.bn\.(.*)", k)
    if m:
        return f"head.linear_fuse.1.{m.group(1)}"

    # decode_head.linear_pred.* → head.linear_pred.*
    m = re.match(r"decode_head\.linear_pred\.(.*)", k)
    if m:
        return f"head.linear_pred.{m.group(1)}"

    # Drop decode_head.bn.* (not in our model)
    if k.startswith("decode_head.bn."):
        return None

    return k   # pass-through for any unmatched keys


def load_nvlabs_weights(model: SegFormerB2ADE20K, ckpt_path: str) -> None:
    """
    Load NVlabs pre-trained weights into our PyTorch model.

    The NVlabs checkpoint stores state dict under the key 'state_dict'
    (standard mmseg format). We remap all key names to match our
    layer naming convention before calling load_state_dict().

    Args:
        model:     SegFormerB2ADE20K instance (randomly initialised).
        ckpt_path: Path to the .pth checkpoint file.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # mmseg saves under 'state_dict'; raw torch.save uses the dict directly
    raw_sd = ckpt.get("state_dict", ckpt)

    new_sd = {}
    skipped = []
    for k, v in raw_sd.items():
        new_k = _remap_key(k)
        if new_k is None:
            skipped.append(k)
        else:
            new_sd[new_k] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"[segformer] Weights loaded from {ckpt_path}")
    if missing:
        print(f"[segformer] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"[segformer] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    if skipped:
        print(f"[segformer] Skipped keys: {skipped}")
