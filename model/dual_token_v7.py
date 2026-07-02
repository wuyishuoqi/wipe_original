"""Time-Frequency Dual-Dimensional Tokenization for CSI (v7 version).

Based on MultiFormer (Qu et al., IEEE IoT-J 2026).
Single-stage cross-attention variant — used by EvtformerV7.

Differences from v8:
  - Single cross-attention (not multi-stage per-layer)
  - Simple average fusion (not gate fusion)
  - No residual scaling
"""

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
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
    y = self.norm1(x)
    y, _ = self.attn(y, y, y)
    x = x + y
    x = x + self.mlp(self.norm2(x))
    return x


class CrossAttentionBlock(nn.Module):
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


class TimeFreqDualTokenV7(nn.Module):

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

    time_in_dim = n_subcarriers * n_antennas   # 342
    freq_in_dim = n_timesteps * n_antennas      # 27

    self.time_proj = nn.Linear(time_in_dim, d_model)
    self.freq_proj = nn.Linear(freq_in_dim, d_model)

    self.time_pos = nn.Parameter(torch.randn(1, n_timesteps, d_model) * 0.02)
    self.freq_pos = nn.Parameter(torch.randn(1, n_subcarriers, d_model) * 0.02)

    self.time_blocks = nn.ModuleList([
      TransformerBlock(d_model, n_heads) for _ in range(n_layers)
    ])
    self.freq_blocks = nn.ModuleList([
      TransformerBlock(d_model, n_heads) for _ in range(n_layers)
    ])

    self.time_cross = CrossAttentionBlock(d_model, n_heads)
    self.freq_cross = CrossAttentionBlock(d_model, n_heads)

    self.time_out = nn.Linear(d_model, time_in_dim)
    self.freq_out = nn.Linear(d_model, freq_in_dim)

  def forward(self, x: torch.Tensor):
    B = x.shape[0]

    time_tokens = x.reshape(B, self.n_timesteps, -1)
    time_tokens = self.time_proj(time_tokens) + self.time_pos

    freq_tokens = x.permute(0, 2, 1, 3).reshape(B, self.n_subcarriers, -1)
    freq_tokens = self.freq_proj(freq_tokens) + self.freq_pos

    for blk in self.time_blocks:
      time_tokens = blk(time_tokens)
    for blk in self.freq_blocks:
      freq_tokens = blk(freq_tokens)

    time_tokens = self.time_cross(time_tokens, freq_tokens)
    freq_tokens = self.freq_cross(freq_tokens, time_tokens)

    time_out = self.time_out(time_tokens)
    freq_out = self.freq_out(freq_tokens)

    time_out = time_out.reshape(B, self.n_timesteps, self.n_subcarriers, self.n_antennas)
    freq_out = freq_out.reshape(B, self.n_subcarriers, self.n_timesteps, self.n_antennas)
    freq_out = freq_out.permute(0, 2, 1, 3)

    out = (time_out + freq_out) / 2.0
    return out
