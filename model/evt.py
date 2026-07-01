"""EVT Spatial Prior Attention Module.

Based on: "Advancing Vision Transformer with Enhanced Spatial Priors"
(Qihang Fan, Huaibo Huang, et al., IEEE TPAMI 2026)

Core idea: inject explicit Euclidean distance-based spatial priors into
self-attention. Nearby spatial positions attend more to each other,
guiding the model to learn geometrically meaningful features.

Attention = Softmax(QK^T / sqrt(d) + log(S)) * V
where S_ij = exp(-gamma * D_ij), D_ij = Euclidean distance between pos i and j
"""

import math
import torch
import torch.nn as nn


class EVTSpatialAttention(nn.Module):
  """Euclidean-distance enhanced Vision Transformer Spatial Attention.

  Replaces the channel-wise transformer with spatially-aware attention.
  The spatial decay matrix S guides attention based on geometric proximity:
  closer positions in the CSI feature map get higher attention weights.

  Args:
      channels: number of input feature channels
      num_heads: multi-head attention heads (default 8)
      gamma_init: initial value for learnable spatial decay rate
  """

  def __init__(self, channels: int, num_heads: int = 8, gamma_init: float = 0.5):
    super().__init__()
    assert channels % num_heads == 0, f"channels {channels} must be divisible by num_heads {num_heads}"

    self.channels = channels
    self.num_heads = num_heads
    self.head_dim = channels // num_heads
    self.scale = math.sqrt(self.head_dim)

    # Learnable spatial decay rate
    self.gamma = nn.Parameter(torch.tensor(gamma_init))
    self._decay_cache = {}

    # Multi-head QKV projection
    self.qkv = nn.Linear(channels, channels * 3, bias=False)

    # Output projection
    self.proj = nn.Linear(channels, channels)

    # Layer norms (pre-norm style)
    self.norm1 = nn.LayerNorm(channels)
    self.norm2 = nn.LayerNorm(channels)

    # Feed-forward (MLP with GELU, expansion ratio 4)
    self.mlp = nn.Sequential(
      nn.Linear(channels, channels * 4),
      nn.GELU(),
      nn.Dropout(0.1),
      nn.Linear(channels * 4, channels),
      nn.Dropout(0.1),
    )

  def _build_spatial_decay(self, H: int, W: int, device, dtype) -> torch.Tensor:
    """Build Euclidean distance-based spatial decay matrix.

    For each pair of spatial positions (i, j), compute:
        D_ij = ||pos_i - pos_j||_2

    Returns:
        D: [N, N] where N = H * W
    """
    cache_key = (H, W, device, dtype)
    if cache_key in self._decay_cache:
      return self._decay_cache[cache_key]

    # Build coordinate grid
    y = torch.arange(H, device=device, dtype=dtype)
    x = torch.arange(W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    coords = torch.stack([yy.flatten(), xx.flatten()], dim=-1)  # [N, 2]

    # Pairwise Euclidean distances
    # ||pos_i - pos_j||_2 = sqrt((y_i - y_j)^2 + (x_i - x_j)^2)
    coords_i = coords.unsqueeze(1)  # [N, 1, 2]
    coords_j = coords.unsqueeze(0)  # [1, N, 2]
    D = torch.sqrt(((coords_i - coords_j) ** 2).sum(dim=-1) + 1e-8)  # [N, N]

    self._decay_cache[cache_key] = D
    return D

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Apply EVT spatial prior attention.

    Args:
        x: [B, C, H, W] feature map

    Returns:
        out: [B, C, H, W] refined feature map
    """
    B, C, H, W = x.shape
    N = H * W  # spatial positions

    # Flatten spatial dims: [B, C, H, W] → [B, N, C]
    x_flat = x.flatten(2).transpose(1, 2)  # [B, N, C]

    # ---- Pre-norm + Multi-head Self-Attention with spatial prior ----
    shortcut = x_flat
    x_norm = self.norm1(x_flat)

    # QKV projection: [B, N, C] → [B, N, 3*C] → [3, B, num_heads, N, head_dim]
    qkv = (
      self.qkv(x_norm)
      .reshape(B, N, 3, self.num_heads, self.head_dim)
      .permute(2, 0, 3, 1, 4)
    )
    q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, num_heads, N, head_dim]

    # Scaled dot-product attention
    attn = (q @ k.transpose(-2, -1)) / self.scale  # [B, num_heads, N, N]

    # ---- Inject Euclidean spatial prior ----
    D = self._build_spatial_decay(H, W, x.device, x.dtype)  # [N, N]
    spatial_bias = -torch.abs(self.gamma) * D
    attn = attn + spatial_bias.unsqueeze(0).unsqueeze(0)  # [B, num_heads, N, N]

    attn = attn.softmax(dim=-1)

    # Apply attention to values
    out = attn @ v  # [B, num_heads, N, head_dim]

    # Merge heads: [B, num_heads, N, head_dim] → [B, N, C]
    out = out.transpose(1, 2).reshape(B, N, C)
    out = self.proj(out)

    # ---- Residual connection ----
    out = shortcut + out

    # ---- MLP with residual ----
    out = out + self.mlp(self.norm2(out))

    # Reshape back: [B, N, C] → [B, C, H, W]
    out = out.transpose(1, 2).reshape(B, C, H, W)

    return out


class EVTSpatiallyGroupedAttention(nn.Module):
  """Spatially-independent grouped attention variant.

  Instead of global attention, divides the feature map into spatial groups
  and applies attention within each group independently. More flexible than
  decomposed attention, and more computationally efficient for large feature maps.
  """

  def __init__(
    self,
    channels: int,
    num_heads: int = 8,
    group_size: tuple = (4, 4),
    gamma_init: float = 0.5,
  ):
    super().__init__()
    self.channels = channels
    self.num_heads = num_heads
    self.group_size = group_size

    # Shared EVT attention for all groups
    self.evt_attn = EVTSpatialAttention(channels, num_heads, gamma_init)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Apply EVT attention within spatial groups.

    Args:
        x: [B, C, H, W]

    Returns:
        out: [B, C, H, W]
    """
    B, C, H, W = x.shape
    gH, gW = self.group_size

    # Pad if needed
    padH = (gH - H % gH) % gH
    padW = (gW - W % gW) % gW
    if padH > 0 or padW > 0:
      x = nn.functional.pad(x, (0, padW, 0, padH))

    B, C, Hp, Wp = x.shape

    # Reshape into groups: [B, C, H/gH, gH, W/gW, gW]
    x = x.reshape(B, C, Hp // gH, gH, Wp // gW, gW)

    # Merge group dims: [B * (H/gH) * (W/gW), C, gH, gW]
    x = x.permute(0, 2, 4, 1, 3, 5).reshape(-1, C, gH, gW)

    # Apply EVT attention to each group
    out = self.evt_attn(x)  # [B * n_groups, C, gH, gW]

    # Restore original shape
    out = out.reshape(B, Hp // gH, Wp // gW, C, gH, gW)
    out = out.permute(0, 3, 1, 4, 2, 5).reshape(B, C, Hp, Wp)

    # Remove padding
    if padH > 0 or padW > 0:
      out = out[:, :, :H, :W]

    return out
