"""Time-Frequency Dual-Dimensional Tokenization for CSI.

Based on MultiFormer (Qu et al., IEEE IoT-J 2026).
Processes CSI as two parallel token streams to preserve both
temporal continuity and spectral (subcarrier) structure:

  - Time tokens: each of the 9 timesteps as a token → captures temporal dynamics
  - Frequency tokens: each of the 114 subcarriers as a token → captures spectral correlations

Self-attention within each domain + cross-attention between domains
produces enhanced CSI features that retain both temporal and spectral structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TransformerBlock(nn.Module):
  """Standard Pre-Norm Transformer block with self-attention + MLP."""

  def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
    super().__init__()
    self.norm1 = nn.LayerNorm(d_model)
    self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
    self.norm2 = nn.LayerNorm(d_model)
    self.mlp = nn.Sequential(
      nn.Linear(d_model, d_model * 4),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(d_model * 4, d_model),
      nn.Dropout(dropout),
    )

  def forward(self, x: torch.Tensor):
    # Self-attention with residual
    y = self.norm1(x)
    y, _ = self.attn(y, y, y)
    x = x + y
    # MLP with residual
    x = x + self.mlp(self.norm2(x))
    return x


class CrossAttentionBlock(nn.Module):
  """Cross-attention: one token stream attends to another."""

  def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
    super().__init__()
    self.norm_q = nn.LayerNorm(d_model)
    self.norm_kv = nn.LayerNorm(d_model)
    self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
    self.norm_out = nn.LayerNorm(d_model)
    self.mlp = nn.Sequential(
      nn.Linear(d_model, d_model * 4),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(d_model * 4, d_model),
      nn.Dropout(dropout),
    )

  def forward(self, query: torch.Tensor, key_value: torch.Tensor):
    y = self.norm_q(query)
    kv = self.norm_kv(key_value)
    y, _ = self.attn(y, kv, kv)
    y = query + y
    y = y + self.mlp(self.norm_out(y))
    return y


class TimeFreqDualToken(nn.Module):
  """Time-Frequency Dual-Dimensional Tokenization for WiFi CSI.

  Args:
      n_timesteps: number of CSI timesteps (9)
      n_subcarriers: number of WiFi subcarriers (114)
      n_antennas: number of MIMO antennas (3)
      d_model: internal token dimension
      n_heads: multi-head attention heads
      n_layers: Transformer layers per domain
  """

  def __init__(
    self,
    n_timesteps: int = 9,
    n_subcarriers: int = 114,
    n_antennas: int = 3,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
  ):
    super().__init__()
    self.n_timesteps = n_timesteps
    self.n_subcarriers = n_subcarriers
    self.n_antennas = n_antennas
    self.d_model = d_model

    time_in_dim = n_subcarriers * n_antennas      # 114 * 3 = 342
    freq_in_dim = n_timesteps * n_antennas         # 9 * 3 = 27

    # Input projections
    self.time_proj = nn.Linear(time_in_dim, d_model)
    self.freq_proj = nn.Linear(freq_in_dim, d_model)

    # Learnable position embeddings
    self.time_pos = nn.Parameter(torch.randn(1, n_timesteps, d_model) * 0.02)
    self.freq_pos = nn.Parameter(torch.randn(1, n_subcarriers, d_model) * 0.02)

    # Domain-specific Transformer encoders
    self.time_blocks = nn.ModuleList([
      TransformerBlock(d_model, n_heads) for _ in range(n_layers)
    ])
    self.freq_blocks = nn.ModuleList([
      TransformerBlock(d_model, n_heads) for _ in range(n_layers)
    ])

    # Cross-domain attention
    self.time_cross = CrossAttentionBlock(d_model, n_heads)
    self.freq_cross = CrossAttentionBlock(d_model, n_heads)

    # Output projections
    self.time_out = nn.Linear(d_model, time_in_dim)
    self.freq_out = nn.Linear(d_model, freq_in_dim)

  def forward(self, x: torch.Tensor):
    """Enhance CSI with dual-token time-frequency attention.

    Args:
        x: [B, 9, 114, 3] raw CSI (after squeeze of fusion dim)

    Returns:
        out: [B, 9, 114, 3] enhanced CSI
    """
    B = x.shape[0]

    # ---- Build dual token streams ----
    # Time tokens: [B, 9, 114, 3] → [B, 9, 342]
    time_tokens = x.reshape(B, self.n_timesteps, -1)
    time_tokens = self.time_proj(time_tokens) + self.time_pos  # [B, 9, d]

    # Frequency tokens: [B, 9, 114, 3] → permute → [B, 114, 9, 3] → [B, 114, 27]
    freq_tokens = x.permute(0, 2, 1, 3).reshape(B, self.n_subcarriers, -1)
    freq_tokens = self.freq_proj(freq_tokens) + self.freq_pos  # [B, 114, d]

    # ---- Domain-specific self-attention ----
    for blk in self.time_blocks:
      time_tokens = blk(time_tokens)
    for blk in self.freq_blocks:
      freq_tokens = blk(freq_tokens)

    # ---- Cross-domain attention ----
    time_tokens = self.time_cross(time_tokens, freq_tokens)
    freq_tokens = self.freq_cross(freq_tokens, time_tokens)

    # ---- Project back to original dimensions ----
    time_out = self.time_out(time_tokens)   # [B, 9, 342]
    freq_out = self.freq_out(freq_tokens)   # [B, 114, 27]

    # ---- Fuse and reshape back to [B, 9, 114, 3] ----
    time_out = time_out.reshape(B, self.n_timesteps, self.n_subcarriers, self.n_antennas)
    freq_out = freq_out.reshape(B, self.n_subcarriers, self.n_timesteps, self.n_antennas)
    freq_out = freq_out.permute(0, 2, 1, 3)

    out = (time_out + freq_out) / 2.0  # simple average fusion
    return out
