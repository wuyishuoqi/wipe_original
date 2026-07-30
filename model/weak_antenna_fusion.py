"""Lightweight sample-adaptive fusion for CSI antenna features."""

import torch
import torch.nn as nn


class WeakAdaptiveAntennaFusion(nn.Module):
  """Learn sample-level antenna weights without pairwise attention.

  The final confidence layer is zero-initialized, so a new model starts from
  an equal antenna mean and learns only the amount of reweighting supported by
  the data.
  """

  def __init__(self, reference_channels: int, n_antennas: int = 3):
    super().__init__()
    self.reference_channels = reference_channels
    self.n_antennas = n_antennas

    hidden_channels = max(reference_channels // 4, 32)
    self.confidence_head = nn.Sequential(
      nn.LayerNorm(reference_channels),
      nn.Linear(reference_channels, hidden_channels),
      nn.GELU(),
      nn.Linear(hidden_channels, 1),
    )
    self._reset_parameters()

  def _reset_parameters(self):
    nn.init.xavier_uniform_(self.confidence_head[1].weight)
    nn.init.zeros_(self.confidence_head[1].bias)
    nn.init.zeros_(self.confidence_head[3].weight)
    nn.init.zeros_(self.confidence_head[3].bias)

  def antenna_weights(self, reference: torch.Tensor) -> torch.Tensor:
    """Return normalized weights from features shaped ``[B, A, C, H, W]``."""
    self._validate(reference)
    if reference.shape[2] != self.reference_channels:
      raise ValueError(
        f"expected {self.reference_channels} reference channels, "
        f"got {reference.shape[2]}"
      )

    descriptors = reference.mean(dim=(-1, -2))
    logits = self.confidence_head(descriptors).squeeze(-1)
    return logits.softmax(dim=1)

  def fuse(
    self,
    features: torch.Tensor,
    weights: torch.Tensor,
  ) -> torch.Tensor:
    """Apply shared antenna weights to any resolution or channel width."""
    self._validate(features)
    expected_shape = features.shape[:2]
    if weights.shape != expected_shape:
      raise ValueError(
        f"expected antenna weights {expected_shape}, got {weights.shape}"
      )

    weights = weights[:, :, None, None, None]
    return (features * weights).sum(dim=1)

  def forward(self, features: torch.Tensor) -> torch.Tensor:
    weights = self.antenna_weights(features)
    return self.fuse(features, weights)

  def _validate(self, features: torch.Tensor):
    if features.ndim != 5:
      raise ValueError(
        f"expected antenna features [B, A, C, H, W], got {features.shape}"
      )
    if features.shape[1] != self.n_antennas:
      raise ValueError(
        f"expected {self.n_antennas} antennas, got {features.shape[1]}"
      )
