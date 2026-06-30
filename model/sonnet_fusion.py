"""SonnetFusion: Full SonnetBlock adapted for WiFi CSI multi-antenna fusion.

Based on Sonnet (AAAI 2025): AdaptiveWavelet + CoherenceAttention + KoopmanLayer.

The three components work as a cohesive pipeline:
  1. AdaptiveWavelet: learns time-frequency atoms for spectral decomposition
  2. CoherenceAttention: FFT-based spectral coherence guides cross-variable attention
  3. KoopmanLayer: unitary Koopman matrix for linear evolution in latent space

Adapted for WiPE: the 68 spatial positions per antenna are treated as "time",
the 512 feature channels as "variables", and 32 learned atoms capture
spectral structure in the feature space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveWavelet(nn.Module):
  """Learned time-frequency atoms for spectral decomposition.

  Each channel learns its own set of chirplet atoms:
      atom(t) = exp(-alpha * t^2) * cos(beta * t + gamma * t^2)

  Args:
      n_vars: number of variables (feature channels)
      n_atoms: number of wavelet atoms per variable
      seq_len: sequence length (spatial positions)
  """

  def __init__(self, n_vars: int, n_atoms: int, seq_len: int):
    super().__init__()
    self.n_vars = n_vars
    self.seq_len = seq_len

    # Per-variable atom parameters: [n_vars, n_atoms, 3] for (alpha, beta, gamma)
    self.freq_params = nn.Parameter(torch.randn(n_vars, n_atoms, 3) * 0.02)

  def forward(self, x: torch.Tensor):
    """Decompose signal into wavelet coefficients.

    Args:
        x: [B, T, N] signal (B=batch, T=seq_len, N=n_vars)

    Returns:
        coeffs: [B, n_atoms, T, N] wavelet coefficients
        atoms: [N, n_atoms, T] wavelet basis functions
    """
    B, T, N = x.shape
    device = x.device

    # Time grid
    t = torch.linspace(0, 1, T, device=device)
    t2 = t ** 2

    # Atom parameters
    alpha = self.freq_params[..., 0]  # [N, n_atoms]
    beta = self.freq_params[..., 1]
    gamma = self.freq_params[..., 2]

    # Build atoms: exp(-alpha * t^2) * cos(beta * t + gamma * t^2)
    atoms = torch.exp(-alpha.unsqueeze(-1) * t2) * \
            torch.cos(beta.unsqueeze(-1) * t + gamma.unsqueeze(-1) * t2)
    # atoms: [N, n_atoms, T]

    # Project signal onto atoms
    coeffs = torch.einsum("btn,nkt->bktn", x, atoms)  # [B, n_atoms, T, N]
    return coeffs, atoms


class CoherenceAttention(nn.Module):
  """Cross-variable spectral coherence attention.

  Computes FFT of Q, K projections and derives spectral coherence
  between variables as physics-informed attention weights.

  Args:
      d_model: feature dimension (number of variables)
      n_atoms: number of atoms (used as key dimension)
      hidden_dim: internal hidden dimension
  """

  def __init__(self, d_model: int, n_atoms: int, hidden_dim: int):
    super().__init__()
    self.n_atoms = n_atoms
    self.hidden_dim = hidden_dim
    self.scale = n_atoms ** -0.5

    self.proj = nn.Linear(d_model, hidden_dim * 3, bias=False)
    self.out_proj = nn.Linear(hidden_dim, d_model)

    self.var_attn = nn.Parameter(torch.eye(d_model) * 0.1)
    self.var_mlp = nn.Sequential(
      nn.Linear(hidden_dim, hidden_dim),
      nn.GELU(),
      nn.Linear(hidden_dim, hidden_dim),
    )
    self.dropout = nn.Dropout(0.1)

  def forward(self, coeffs: torch.Tensor):
    """Apply coherence-based attention to wavelet coefficients.

    Args:
        coeffs: [B, n_atoms, T, D] wavelet coefficients

    Returns:
        out: [B, T, n_atoms, d_model] coherence-refined features
    """
    B, N, T, D = coeffs.shape

    # Project to Q, K, V
    qkv = self.proj(coeffs)
    qkv = qkv.reshape(B, N, T, self.hidden_dim, 3)
    q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]  # [B, N, T, H]

    # Rearrange for FFT
    q = q.permute(0, 1, 3, 2)  # [B, N, H, T]
    k = k.permute(0, 1, 3, 2)

    # FFT along time dimension
    Q_fft = torch.fft.rfft(q, dim=-1)  # [B, N, H, T_fft]
    K_fft = torch.fft.rfft(k, dim=-1)

    # Spectral coherence
    P_xy = (Q_fft * K_fft.conj()).mean(dim=1)    # [B, H, T_fft]
    P_xx = (Q_fft * Q_fft.conj()).mean(dim=1)
    P_yy = (K_fft * K_fft.conj()).mean(dim=1)

    coherence = P_xy.abs().pow(2) / (P_xx.abs() * P_yy.abs()).clamp(min=1e-6)

    # Attention
    time_attn = F.softmax(coherence / self.scale, dim=-1)  # [B, H, T_fft]
    time_attn = self.dropout(time_attn)

    # Apply attention to V
    Vr = torch.fft.rfft(v.permute(0, 1, 3, 2), dim=-1)  # [B, N, H, T_fft]
    Vr = Vr * time_attn.unsqueeze(1)                      # weighted
    out_time = torch.fft.irfft(Vr, n=T, dim=-1)            # [B, N, H, T]
    out_time = out_time.permute(0, 3, 1, 2).reshape(B, T, N, self.hidden_dim)

    # Variable mixing
    var_attn = F.softmax(self.var_attn, dim=-1)
    out_var = torch.einsum("b t n h, v h -> b t n v", out_time, var_attn)

    # Residual + MLP
    out = out_var + self.var_mlp(out_var)
    out = self.out_proj(out)
    return out.permute(0, 2, 1, 3)  # [B, n_atoms, T, d_model]


class KoopmanLayer(nn.Module):
  """Unitary Koopman operator for linear time evolution.

  Builds a unitary matrix on the Stiefel manifold via QR decomposition
  and uses it to evolve the latent state linearly.

  Args:
      n_atoms: dimension of Koopman matrix
  """

  def __init__(self, n_atoms: int):
    super().__init__()
    self.U = nn.Parameter(torch.randn(n_atoms, n_atoms, dtype=torch.complex64) * 0.02)
    self.theta = nn.Parameter(torch.zeros(n_atoms))

    with torch.no_grad():
      self.U.data = self.U / (torch.norm(self.U, dim=0, keepdim=True) + 1e-8)

  def forward(self, x: torch.Tensor):
    """Evolve state via Koopman operator.

    Args:
        x: [B, K, T, N] complex input

    Returns:
        out: [B, K, T, N] evolved state
    """
    B, K, T, N = x.shape

    # Build unitary Koopman matrix K = U @ diag(e^{i*theta}) @ U^H
    U, _ = torch.linalg.qr(self.U)
    D = torch.diag(torch.exp(1j * self.theta))
    K_mat = U @ D @ U.conj().T  # [K, K]

    x = x.permute(0, 3, 1, 2).reshape(B * N, K, T)
    K_mat = K_mat.unsqueeze(0).expand(B * N, -1, -1)
    x_evolved = torch.bmm(K_mat, x)
    x_evolved = x_evolved.reshape(B, N, K, T).permute(0, 2, 3, 1)

    return x_evolved


class SonnetFusionBlock(nn.Module):
  """Complete Sonnet processing block for one antenna's feature stream.

  Wavelet → CoherenceAttention → Koopman → Reconstruction.

  Args:
      n_vars: feature channels (512)
      n_atoms: wavelet atoms (32)
      seq_len: spatial positions per antenna (68)
      hidden_dim: attention hidden dim (512)
  """

  def __init__(self, n_vars: int = 512, n_atoms: int = 32,
               seq_len: int = 68, hidden_dim: int = 512):
    super().__init__()
    self.wavelet = AdaptiveWavelet(n_vars, n_atoms, seq_len)
    self.attention = CoherenceAttention(n_vars, n_atoms, hidden_dim)
    self.koopman = KoopmanLayer(n_atoms)

  def forward(self, x: torch.Tensor):
    """Process one antenna stream through Sonnet pipeline.

    Args:
        x: [B, T, N] where T=seq_len, N=n_vars

    Returns:
        out: [B, T, N] reconstructed features
    """
    # Wavelet decomposition
    coeffs, atoms = self.wavelet(x)        # coeffs: [B, n_atoms, T, N]

    # Coherence attention
    z = self.attention(coeffs)              # [B, n_atoms, T, N]

    # Koopman evolution (complex)
    z_koop = self.koopman(z.to(torch.complex64))  # [B, n_atoms, T, N]

    # Reconstruction: sum over atoms weighted by wavelet basis
    recon = torch.einsum("bktn,nkt->btn", z_koop.real, atoms)  # [B, T, N]
    return recon


class SonnetFusion(nn.Module):
  """Multi-antenna Sonnet-based feature fusion for WiFi CSI.

  Processes each antenna's features through shared SonnetFusionBlocks,
  then fuses the outputs.

  Args:
      channels: feature channels (512)
      antenna_width: width per antenna in feature map (4)
      n_atoms: wavelet atoms (32)
  """

  def __init__(self, channels: int = 512, antenna_width: int = 4,
               n_atoms: int = 32):
    super().__init__()
    self.channels = channels
    self.antenna_width = antenna_width
    self.n_antennas = 3
    self.seq_len = 17 * antenna_width  # 68 spatial positions per antenna

    # Shared SonnetBlock for all 3 antennas
    self.block = SonnetFusionBlock(
      n_vars=channels,
      n_atoms=n_atoms,
      seq_len=self.seq_len,
      hidden_dim=channels,
    )

    # Post-fusion normalization
    self.bn = nn.BatchNorm2d(channels)

  def forward(self, x: torch.Tensor):
    """Fuse 3-antenna features via Sonnet pipeline.

    Args:
        x: [B, C, H, W_ant*3] concatenated features

    Returns:
        out: [B, C, H, W_ant*3] fused features
    """
    B, C, H, W = x.shape
    W_ant = self.antenna_width

    outputs = []
    for a in range(self.n_antennas):
      # Extract one antenna: [B, C, H, W_ant] -> [B, C, H*W_ant] -> [B, H*W_ant, C]
      ant_f = x[:, :, :, a * W_ant:(a + 1) * W_ant]
      ant_f = ant_f.flatten(2).transpose(1, 2)  # [B, 68, 512]

      # Pass through SonnetBlock
      out = self.block(ant_f)  # [B, 68, 512]

      # Reshape back: [B, 68, 512] -> [B, 512, 17, 4]
      out = out.transpose(1, 2).reshape(B, C, H, W_ant)
      outputs.append(out)

    # Concatenate antennas along width
    out = torch.cat(outputs, dim=3)  # [B, C, H, W_ant*3]
    out = self.bn(out)
    return out
