"""EvtformerThree: conservative EVT priors with coordinate detail refinement."""

import math

import numpy as np
import torch
import torch.nn as nn

from model.evtformer import Evtformer, _init_weights
from model.evtformer_two import DetailPreservingSonnet, MultiScaleEVTAttention


def _logit(probability: float) -> float:
  return math.log(probability / (1.0 - probability))


class ConservativeEVTStack(nn.Module):
  """Full primary EVT followed by one scalar-gated weak refinement."""

  def __init__(
    self,
    channels: int = 128,
    num_heads: int = 4,
    secondary_gate_init: float = 0.02,
    secondary_gate_max: float = 0.08,
  ):
    super().__init__()
    self.primary = MultiScaleEVTAttention(
      channels=channels,
      num_heads=num_heads,
      gamma_init=(0.002, 0.002, 0.005, 0.040),
      gamma_max=0.05,
      dropout=0.05,
    )
    self.secondary = MultiScaleEVTAttention(
      channels=channels,
      num_heads=num_heads,
      gamma_init=(0.01, 0.05, 0.12, 0.24),
      gamma_max=0.30,
      dropout=0.05,
    )
    probability = secondary_gate_init / secondary_gate_max
    self.secondary_gate_logit = nn.Parameter(
      torch.tensor(_logit(probability))
    )
    self.secondary_gate_max = secondary_gate_max

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.primary(x)
    secondary = self.secondary(x)
    gate = self.secondary_gate_max * torch.sigmoid(
      self.secondary_gate_logit
    )
    return x + gate * (secondary - x)


class CoordinateResidualHead(nn.Module):
  """Predict a bounded correction from features before and after EVT."""

  def __init__(
    self,
    channels: int = 128,
    max_offset: float = 2.0,
    scale_init: float = 0.25,
  ):
    super().__init__()
    self.refine = nn.Sequential(
      nn.Conv2d(channels * 3, 64, kernel_size=1, bias=False),
      nn.BatchNorm2d(64),
      nn.GELU(),
      nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(32),
      nn.GELU(),
      nn.Conv2d(32, 2, kernel_size=1, bias=True),
    )
    self.pool = nn.AdaptiveAvgPool2d((17, 1))
    self.max_offset = max_offset
    probability = scale_init / max_offset
    self.scale_logit = nn.Parameter(torch.tensor(_logit(probability)))

    final = self.refine[-1]
    nn.init.zeros_(final.weight)
    nn.init.zeros_(final.bias)

  def forward(
    self,
    before_evt: torch.Tensor,
    after_evt: torch.Tensor,
  ) -> torch.Tensor:
    features = torch.cat(
      (before_evt, after_evt, after_evt - before_evt),
      dim=1,
    )
    correction = self.pool(self.refine(features)).squeeze(-1)
    correction = correction.transpose(1, 2)
    scale = self.max_offset * torch.sigmoid(self.scale_logit)
    return scale * torch.tanh(correction)


class EvtformerThree(Evtformer):
  """Strict-PCK variant with conservative feature and coordinate refinement."""

  def __init__(self, nInChannel: int, nOutChannel: int):
    super().__init__(nInChannel, nOutChannel)
    self.sonnet = DetailPreservingSonnet(
      self.sonnet,
      gate_init=0.10,
      gate_max=0.50,
    )
    self.evt = ConservativeEVTStack(channels=128, num_heads=4)
    self.coordinate_refine = CoordinateResidualHead(channels=128)
    self.evt.apply(_init_weights)
    self.coordinate_refine.apply(_init_weights)

    # Keep the residual path exactly zero at initialization.
    final = self.coordinate_refine.refine[-1]
    nn.init.zeros_(final.weight)
    nn.init.zeros_(final.bias)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x[:, :, :, 0, :]
    x = self.dual_token(x)
    x = x.permute(0, 3, 2, 1)
    x = x[:, np.newaxis, :, :, :]

    branches = []
    for antenna in range(3):
      branch = self.upsample136x32(x[:, :, antenna, :, :])
      branches.append(self._encode_branch(branch))

    x = torch.cat(branches, dim=3)
    x = self.sonnet(x)
    x = self.channel_reduce(x)
    x = self.spatial_upsample(x)
    before_evt = self.bn2(x)
    after_evt = self.evt(before_evt)

    coordinates = self.decode(after_evt)
    coordinates = self.final_pool(coordinates).squeeze(dim=3)
    coordinates = self.bn1(coordinates).transpose(1, 2)

    correction = self.coordinate_refine(before_evt, after_evt)
    return coordinates + correction
