"""Weak-gated Evtformer with an SSD-inspired CSI conditioning front-end.

The dataset exposes 15 CSI frames to this model. A short-context Hankel
low-rank reconstruction is applied independently to every subcarrier and
antenna before the center nine frames enter the original Evtformer.
"""

import torch
import torch.nn as nn

from model.evtformer import Evtformer


class ShortContextSSD(nn.Module):
  """Hankel-SVD reconstruction with a bounded per-antenna residual gate.

  This keeps the trajectory-matrix/SVD core of singular-spectrum methods,
  but is intentionally described as SSD-inspired because 15 samples are too
  short for the full iterative, PSD-driven SSD algorithm.
  """

  def __init__(
    self,
    window: int = 7,
    rank: int = 3,
    gate_init: float = -2.2,
    gate_max: float = 0.25,
  ):
    super().__init__()
    self.window = window
    self.rank = rank
    self.gate_logit = nn.Parameter(torch.full((3,), gate_init))
    self.gate_max = gate_max

  def _reconstruct(self, x: torch.Tensor) -> torch.Tensor:
    # Treat every subcarrier-antenna pair as an independent time series.
    batch, timesteps, subcarriers, antennas = x.shape
    series = x.permute(0, 2, 3, 1).reshape(-1, timesteps)
    window = min(self.window, timesteps)
    n_windows = timesteps - window + 1

    hankel = series.unfold(1, window, 1)
    covariance = hankel.transpose(1, 2) @ hankel
    _, eigenvectors = torch.linalg.eigh(covariance)
    basis = eigenvectors[:, :, -min(self.rank, window):]
    hankel_reconstructed = (hankel @ basis) @ basis.transpose(1, 2)

    reconstructed = torch.zeros_like(series)
    counts = torch.zeros(timesteps, device=x.device, dtype=x.dtype)
    for offset in range(window):
      reconstructed[:, offset:offset + n_windows] += hankel_reconstructed[:, :, offset]
      counts[offset:offset + n_windows] += 1
    reconstructed = reconstructed / counts.clamp_min(1)
    return reconstructed.reshape(batch, subcarriers, antennas, timesteps).permute(0, 3, 1, 2)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # CSI is input data, so detaching the deterministic eigendecomposition
    # avoids unstable eigenvector gradients without blocking model learning.
    with torch.no_grad():
      reconstructed = self._reconstruct(x.float()).to(dtype=x.dtype)
    gate = self.gate_max * torch.sigmoid(self.gate_logit).view(1, 1, 1, 3)
    return x + gate * (reconstructed - x)


class EvtSsd(Evtformer):
  """SSD-conditioned variant of the current weak-gated Evtformer."""

  def __init__(self, nInChannel: int, nOutChannel: int):
    super().__init__(nInChannel, nOutChannel)
    self.ssd = ShortContextSSD()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    csi = x[:, :, :, 0, :]
    if csi.shape[1] < 9:
      raise ValueError(f"EvtSsd requires at least 9 CSI frames, got {csi.shape[1]}")

    csi = self.ssd(csi)
    start = (csi.shape[1] - 9) // 2
    csi = csi[:, start:start + 9]
    return super().forward(csi.unsqueeze(3))
