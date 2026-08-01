"""Bounded coordinate refinement for distal COCO joints."""

import torch
import torch.nn as nn


def _logit(probability: float) -> float:
  probability = min(max(probability, 1e-6), 1.0 - 1e-6)
  return float(torch.logit(torch.tensor(probability)))


class BoundedDistalJointHead(nn.Module):
  """Predict small wrist/ankle corrections from joint and parent tokens."""

  def __init__(
    self,
    channels: int = 128,
    hidden_channels: int = 64,
    n_joints: int = 17,
    max_correction: float = 0.5,
  ):
    super().__init__()
    if n_joints != 17:
      raise ValueError("the distal COCO head requires 17 joint tokens")

    self.channels = channels
    self.n_joints = n_joints
    self.max_correction = max_correction

    # COCO: wrists <- elbows, ankles <- knees.
    self.register_buffer(
      "distal_indices",
      torch.tensor((9, 10, 15, 16), dtype=torch.long),
    )
    self.register_buffer(
      "parent_indices",
      torch.tensor((7, 8, 13, 14), dtype=torch.long),
    )

    context_channels = channels * 3
    self.refinement = nn.Sequential(
      nn.LayerNorm(context_channels),
      nn.Linear(context_channels, hidden_channels),
      nn.GELU(),
      nn.Linear(hidden_channels, 2),
    )
    self._reset_parameters()

  def _reset_parameters(self):
    nn.init.xavier_uniform_(self.refinement[1].weight)
    nn.init.zeros_(self.refinement[1].bias)
    nn.init.zeros_(self.refinement[3].weight)
    nn.init.zeros_(self.refinement[3].bias)

  def forward(self, tokens: torch.Tensor) -> torch.Tensor:
    """Return bounded corrections shaped ``[B, 17, 2]``."""
    if tokens.ndim != 3:
      raise ValueError(f"expected joint tokens [B, 17, C], got {tokens.shape}")
    if tokens.shape[1:] != (self.n_joints, self.channels):
      raise ValueError(
        f"expected joint tokens [B, {self.n_joints}, {self.channels}], "
        f"got {tokens.shape}"
      )

    distal = tokens.index_select(1, self.distal_indices)
    parent = tokens.index_select(1, self.parent_indices)
    context = torch.cat((distal, parent, distal - parent), dim=2)
    distal_correction = self.max_correction * torch.tanh(
      self.refinement(context)
    )

    correction = tokens.new_zeros((tokens.shape[0], self.n_joints, 2))
    return correction.index_copy(1, self.distal_indices, distal_correction)


class SoftGatedDistalJointHead(BoundedDistalJointHead):
  """Apply small independently gated corrections without tanh saturation."""

  def __init__(
    self,
    channels: int = 128,
    hidden_channels: int = 64,
    n_joints: int = 17,
    max_correction: float = 0.25,
    gate_init: float = 0.2,
    raw_scale: float = 0.1,
    detach_tokens: bool = True,
  ):
    super().__init__(
      channels=channels,
      hidden_channels=hidden_channels,
      n_joints=n_joints,
      max_correction=max_correction,
    )
    self.raw_scale = raw_scale
    self.detach_tokens = detach_tokens
    self.gate_logits = nn.Parameter(
      torch.full((self.distal_indices.numel(), 1), _logit(gate_init))
    )

  @staticmethod
  def _softsign(x: torch.Tensor) -> torch.Tensor:
    return x / (1.0 + x.abs())

  def effective_gates(self) -> torch.Tensor:
    return torch.sigmoid(self.gate_logits)

  def forward(self, tokens: torch.Tensor) -> torch.Tensor:
    """Return softly bounded wrist/ankle corrections shaped ``[B, 17, 2]``."""
    if tokens.ndim != 3:
      raise ValueError(f"expected joint tokens [B, 17, C], got {tokens.shape}")
    if tokens.shape[1:] != (self.n_joints, self.channels):
      raise ValueError(
        f"expected joint tokens [B, {self.n_joints}, {self.channels}], "
        f"got {tokens.shape}"
      )

    head_tokens = tokens.detach() if self.detach_tokens else tokens
    distal = head_tokens.index_select(1, self.distal_indices)
    parent = head_tokens.index_select(1, self.parent_indices)
    context = torch.cat((distal, parent, distal - parent), dim=2)

    raw_correction = self.raw_scale * self.refinement(context)
    distal_correction = (
      self.max_correction
      * self.effective_gates().unsqueeze(0)
      * self._softsign(raw_correction)
    )

    correction = tokens.new_zeros((tokens.shape[0], self.n_joints, 2))
    return correction.index_copy(1, self.distal_indices, distal_correction)
