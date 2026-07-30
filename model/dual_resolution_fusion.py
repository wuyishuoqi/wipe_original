"""Conservative bidirectional exchange between CSI feature resolutions."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualResolutionExchange(nn.Module):
  """Exchange semantic and detail features through bounded residual gates."""

  def __init__(
    self,
    high_channels: int = 64,
    low_channels: int = 512,
    exchange_gate_max: float = 0.25,
  ):
    super().__init__()
    self.exchange_gate_max = exchange_gate_max

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

    # Signed zero gates preserve both original branches at initialization.
    self.low_to_high_gate = nn.Parameter(torch.zeros(()))
    self.high_to_low_gate = nn.Parameter(torch.zeros(()))

  def forward(
    self,
    high: torch.Tensor,
    low: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    low_update = self.low_to_high(low)
    low_update = F.interpolate(
      low_update,
      size=high.shape[-2:],
      mode="bilinear",
      align_corners=False,
    )

    high_update = self.high_to_low(high)
    if high_update.shape[-2:] != low.shape[-2:]:
      high_update = F.interpolate(
        high_update,
        size=low.shape[-2:],
        mode="bilinear",
        align_corners=False,
      )

    low_to_high_gate = self.exchange_gate_max * torch.tanh(
      self.low_to_high_gate
    )
    high_to_low_gate = self.exchange_gate_max * torch.tanh(
      self.high_to_low_gate
    )
    return (
      high + low_to_high_gate * low_update,
      low + high_to_low_gate * high_update,
    )
