"""Coherence-based Cross-Antenna Fusion for WiFi CSI.

Inspired by Sonnet's CoherenceAttention (AAAI 2025).
Computes spectral coherence between features from different WiFi antennas
and uses it as physics-informed attention weights for fusion.

Core insight: The 3 MIMO antennas capture correlated multipath components
of the same WiFi signal. Their spectral coherence has physical meaning —
highly coherent antenna pairs should attend more to each other.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoherenceAntennaFusion(nn.Module):
  """Spectral coherence-based cross-antenna feature fusion.

  Splits concatenated 3-antenna ResNet features, computes coherence between
  antenna pairs via FFT, and fuses them with coherence-guided attention.

  Args:
      channels: feature channels (e.g., 512)
      antenna_width: width per antenna in concatenated feature map (e.g., 4)
      n_heads: attention heads (default 8)
      dropout: dropout rate
  """

  def __init__(
    self,
    channels: int = 512,
    antenna_width: int = 4,
    n_heads: int = 8,
    dropout: float = 0.1,
  ):
    super().__init__()
    self.channels = channels
    self.antenna_width = antenna_width
    self.n_heads = n_heads
    self.head_dim = channels // n_heads
    self.scale = self.head_dim ** -0.5

    assert channels % n_heads == 0, f"channels {channels} must be divisible by n_heads {n_heads}"

    # QKV projection (shared across antennas)
    self.qkv = nn.Linear(channels, channels * 3, bias=False)
    self.out_proj = nn.Linear(channels, channels)

    # Variable mixing: learn antenna-to-antenna relationships
    self.var_attn = nn.Parameter(torch.eye(3))

    # Post-fusion refinement
    self.norm = nn.LayerNorm(channels)
    self.mlp = nn.Sequential(
      nn.Linear(channels, channels * 4),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(channels * 4, channels),
      nn.Dropout(dropout),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Fuse 3-antenna features via spectral coherence attention.

    Args:
        x: [B, C, H, W_ant*3] concatenated features from 3 antennas

    Returns:
        out: [B, C, H, W_ant*3] coherence-fused features
    """
    B, C, H, W = x.shape
    W_ant = self.antenna_width
    n_ant = W // W_ant  # should be 3

    # ---- Split into 3 antenna groups ----
    # [B, C, H, W] -> 3 × [B, C, H, W_ant] -> [B, 3, C, H*W_ant]
    ant_features = []
    for a in range(n_ant):
      ant_f = x[:, :, :, a * W_ant : (a + 1) * W_ant]  # [B, C, H, W_ant]
      ant_f = ant_f.flatten(2)                            # [B, C, N_spatial]
      ant_features.append(ant_f)
    x_ant = torch.stack(ant_features, dim=1)               # [B, 3, C, N]

    # ---- Reshape for coherence attention ----
    # Treat: 3 antennas as "variables", C channels as "atoms", N spatial as "time"
    # Format: [B, N_atoms=C, T=N, D=3]
    N = H * W_ant  # spatial positions per antenna
    x_ant = x_ant.permute(0, 2, 3, 1)  # [B, C, N, 3]

    # QKV projection: [B, C, N, 3] -> [B, C, N, 3*dim]
    x_flat = x_ant.permute(0, 2, 3, 1).reshape(B * N, 3, C)  # [B*N, 3, C]
    qkv = self.qkv(x_flat)  # [B*N, 3, 3*C]
    qkv = qkv.reshape(B, N, 3, C, 3).permute(4, 0, 1, 2, 3)  # [3, B, N, 3, C]
    q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, N, 3, C]

    # Multi-head split
    q = q.reshape(B, N, 3, self.n_heads, self.head_dim).permute(0, 3, 1, 2, 4)  # [B, H, N, 3, D]
    k = k.reshape(B, N, 3, self.n_heads, self.head_dim).permute(0, 3, 1, 2, 4)
    v = v.reshape(B, N, 3, self.n_heads, self.head_dim).permute(0, 3, 1, 2, 4)

    # ---- Spectral Coherence Attention ----
    # FFT along the feature dimension (N spatial positions)
    Q_fft = torch.fft.rfft(q, dim=2)  # [B, H, N_fft, 3, D]
    K_fft = torch.fft.rfft(k, dim=2)

    # Cross-spectral density: compute all antenna-pair cross-spectra at once
    # Q_fft: [B, H, N_fft, 3, D], K_fft: [B, H, N_fft, 3, D]
    # Cross-spectrum for all (a,b) pairs:
    # P_ab = mean over freq and head_dim of Q_a * conj(K_b)
    Qr = Q_fft.real  # [B, H, N_fft, 3, D]
    Qi = Q_fft.imag
    Kr = K_fft.real
    Ki = K_fft.imag

    # Compute P_ab for each antenna pair
    # Q[:,:,:,a,:] * conj(K[:,:,:,b,:]) summed over last dim
    # = (Qr_a + i*Qi_a) * (Kr_b - i*Ki_b) = (Qr_a*Kr_b + Qi_a*Ki_b) + i*(Qi_a*Kr_b - Qr_a*Ki_b)
    # Power = |cross|^2 = (Qr_a*Kr_b + Qi_a*Ki_b)^2 + (Qi_a*Kr_b - Qr_a*Ki_b)^2

    # Efficient: compute all 3x3 pairs
    P_list = []
    for a in range(3):
      for b in range(3):
        qr_a = Qr[:, :, :, a, :]  # [B, H, N_fft, D]
        qi_a = Qi[:, :, :, a, :]
        kr_b = Kr[:, :, :, b, :]
        ki_b = Ki[:, :, :, b, :]
        real_part = (qr_a * kr_b + qi_a * ki_b).sum(dim=-1)  # [B, H, N_fft]
        imag_part = (qi_a * kr_b - qr_a * ki_b).sum(dim=-1)
        power = (real_part**2 + imag_part**2).mean(dim=-1)  # [B, H]
        P_list.append(power)

    # Stack into [B, H, 3, 3] coherence matrix
    coherence = torch.stack(P_list, dim=-1).reshape(B, self.n_heads, 3, 3)

    # Normalize: C_ab = P_ab / sqrt(P_aa * P_bb)
    P_aa = coherence.diagonal(dim1=-2, dim2=-1)  # [B, H, 3]
    denom = (P_aa.unsqueeze(-1) * P_aa.unsqueeze(-2)).sqrt().clamp(min=1e-6)  # [B, H, 3, 3]
    coherence = coherence / denom

    attn = F.softmax(coherence * self.scale, dim=-1)  # [B, H, 3, 3]
    attn = F.dropout(attn, p=0.1, training=self.training)

    # Apply antenna-mixing attention to values
    # v: [B, H, N, 3, D], attn: [B, H, 3, 3]
    v_out = torch.einsum("bhij,bhnjd->bhnid", attn, v)  # mix antennas via attention

    # ---- Variable-level mixing (learned antenna relationships) ----
    var_attn = F.softmax(self.var_attn, dim=-1)  # [3, 3]
    v_out = torch.einsum("bhnid,ij->bhnjd", v_out, var_attn)

    # ---- Merge heads ----
    v_out = v_out.permute(0, 2, 3, 1, 4).reshape(B, N, 3, C)  # [B, N, 3, C]

    # ---- Residual + MLP ----
    residual = x_flat.reshape(B, N, 3, C)  # [B, N, 3, C]
    out = self.norm(v_out + residual)
    out = out + self.mlp(out)
    out = self.out_proj(out)  # [B, N, 3, C]

    # ---- Reshape back to [B, C, H, W_ant*3] ----
    out = out.permute(0, 3, 1, 2)          # [B, C, N, 3]
    out = out.reshape(B, C, N * 3)          # [B, C, N*3]
    out = out.reshape(B, C, H, W_ant * 3)   # [B, C, H, W]

    return out
