"""Bounded non-negative exchange between CSI feature resolutions."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstrainedDualResolutionExchange(nn.Module):
  """Keep both exchange directions weak and prevent subtractive write-back."""

  def __init__(
    self,
    high_channels: int = 64,
    low_channels: int = 512,
    low_to_high_gate_max: float = 0.05,
    high_to_low_gate_max: float = 0.025,
    gate_init: float = -2.0,
  ):
    super().__init__()
    self.low_to_high_gate_max = low_to_high_gate_max
    self.high_to_low_gate_max = high_to_low_gate_max

    self.low_to_high = nn.Sequential(
      nn.Conv2d(low_channels, high_channels, kernel_size=1, bias=False),
      nn.BatchNorm2d(high_channels),
      nn.ReLU(inplace=True),
    )
    self.high_to_low = nn.Sequential(
      nn.Conv2d(
        high_channels, 128, kernel_size=3, stride=2, padding=1, bias=False
      ),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, low_channels, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(low_channels),
      nn.ReLU(inplace=True),
    )

    self.low_to_high_gate_logit = nn.Parameter(torch.tensor(gate_init))
    self.high_to_low_gate_logit = nn.Parameter(torch.tensor(gate_init))

  def effective_gates(self) -> tuple[torch.Tensor, torch.Tensor]:
    return (
      self.low_to_high_gate_max * torch.sigmoid(
        self.low_to_high_gate_logit
      ),
      self.high_to_low_gate_max * torch.sigmoid(
        self.high_to_low_gate_logit
      ),
    )

  def forward(
    self,
    high: torch.Tensor,
    low: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    low_to_high = self.low_to_high(low)
    low_to_high = F.interpolate(
      low_to_high,
      size=high.shape[-2:],
      mode="bilinear",
      align_corners=False,
    )

    high_to_low = self.high_to_low(high)
    if high_to_low.shape[-2:] != low.shape[-2:]:
      high_to_low = F.interpolate(
        high_to_low,
        size=low.shape[-2:],
        mode="bilinear",
        align_corners=False,
      )

    low_to_high_gate, high_to_low_gate = self.effective_gates()
    return (
      high + low_to_high_gate * low_to_high,
      low + high_to_low_gate * high_to_low,
    )
