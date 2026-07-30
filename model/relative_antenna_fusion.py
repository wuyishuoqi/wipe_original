"""Adaptive relative-attention fusion for multiple CSI antennas."""

import math

import torch
import torch.nn as nn


class RelativeAntennaFusion(nn.Module):
  """Fuse antenna features with reliability weights and pairwise attention.

  The module starts as an equal-weight antenna mean. The confidence head and
  relative-attention projection are zero-initialized so that both adaptive
  paths are learned gradually instead of perturbing the backbone immediately.
  """

  def __init__(
    self,
    channels: int,
    n_antennas: int = 3,
    attention_dim: int = 128,
    num_heads: int = 4,
    relative_gate_init: float = -3.0,
    relative_gate_max: float = 0.25,
  ):
    super().__init__()
    if attention_dim % num_heads != 0:
      raise ValueError("attention_dim must be divisible by num_heads")

    self.channels = channels
    self.n_antennas = n_antennas
    self.attention_dim = attention_dim
    self.num_heads = num_heads
    self.head_dim = attention_dim // num_heads
    self.relative_gate_max = relative_gate_max

    hidden_channels = max(channels // 4, 32)
    self.confidence_head = nn.Sequential(
      nn.LayerNorm(channels),
      nn.Linear(channels, hidden_channels),
      nn.GELU(),
      nn.Linear(hidden_channels, 1),
    )

    self.token_norm = nn.LayerNorm(channels)
    self.qkv = nn.Linear(channels, attention_dim * 3, bias=False)
    self.relative_projection = nn.Linear(attention_dim, channels, bias=False)
    self.relative_gate_logit = nn.Parameter(torch.tensor(relative_gate_init))

    self._reset_parameters()

  def _reset_parameters(self):
    nn.init.xavier_uniform_(self.confidence_head[1].weight)
    nn.init.zeros_(self.confidence_head[1].bias)
    nn.init.zeros_(self.confidence_head[3].weight)
    nn.init.zeros_(self.confidence_head[3].bias)
    nn.init.xavier_uniform_(self.qkv.weight)
    nn.init.zeros_(self.relative_projection.weight)

  def _relative_attention(self, features: torch.Tensor) -> torch.Tensor:
    batch, antennas, channels, height, width = features.shape
    tokens = features.permute(0, 3, 4, 1, 2)
    normalized = self.token_norm(tokens)

    q, k, v = self.qkv(normalized).chunk(3, dim=-1)

    def split_heads(x: torch.Tensor) -> torch.Tensor:
      x = x.reshape(
        batch, height, width, antennas, self.num_heads, self.head_dim
      )
      return x.permute(0, 1, 2, 4, 3, 5)

    q = split_heads(q)
    k = split_heads(k)
    v = split_heads(v)

    attention = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
    attention = attention.softmax(dim=-1)
    relative = attention @ v
    relative = relative.permute(0, 1, 2, 4, 3, 5).reshape(
      batch, height, width, antennas, self.attention_dim
    )
    return self.relative_projection(relative)

  def antenna_weights(self, features: torch.Tensor) -> torch.Tensor:
    """Return normalized sample-level antenna reliability weights."""
    descriptors = features.mean(dim=(-1, -2))
    logits = self.confidence_head(descriptors).squeeze(-1)
    return logits.softmax(dim=1)

  def forward(self, features: torch.Tensor) -> torch.Tensor:
    """Fuse features shaped ``[B, A, C, H, W]`` into ``[B, C, H, W]``."""
    if features.ndim != 5:
      raise ValueError(
        f"expected antenna features [B, A, C, H, W], got {features.shape}"
      )
    if features.shape[1] != self.n_antennas:
      raise ValueError(
        f"expected {self.n_antennas} antennas, got {features.shape[1]}"
      )
    if features.shape[2] != self.channels:
      raise ValueError(
        f"expected {self.channels} channels, got {features.shape[2]}"
      )

    relative = self._relative_attention(features)
    tokens = features.permute(0, 3, 4, 1, 2)
    gate = self.relative_gate_max * torch.sigmoid(self.relative_gate_logit)
    refined = tokens + gate * relative

    weights = self.antenna_weights(features)
    weights = weights[:, None, None, :, None]
    fused = (refined * weights).sum(dim=3)
    return fused.permute(0, 3, 1, 2).contiguous()
