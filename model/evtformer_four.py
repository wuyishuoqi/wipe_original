"""EvtformerFour: grouped coordinate refinement on the EvtformerThree route."""

import math

import torch
import torch.nn as nn

from model.evtformer import _init_weights
from model.evtformer_three import EvtformerThree


def _logit(probability: float) -> float:
  return math.log(probability / (1.0 - probability))


class GroupedCoordinateResidualHead(nn.Module):
  """Bound coordinate corrections by anatomical group instead of per joint."""

  def __init__(self, channels: int = 128):
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

    # Groups: face, torso, arms, legs.
    group_index = torch.tensor([
      0, 0, 0, 0, 0,
      1, 1, 2, 2, 2, 2,
      1, 1, 3, 3, 3, 3,
    ])
    max_offsets = torch.tensor((2.5, 1.5, 2.5, 2.5))
    initial_offsets = torch.tensor((0.30, 0.20, 0.35, 0.35))
    probabilities = initial_offsets / max_offsets

    self.register_buffer("group_index", group_index)
    self.register_buffer("max_offsets", max_offsets)
    self.scale_logits = nn.Parameter(torch.logit(probabilities))

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

    group_scales = self.max_offsets * torch.sigmoid(self.scale_logits)
    joint_scales = group_scales[self.group_index].view(1, 17, 1)
    return joint_scales * torch.tanh(correction)


class EvtformerFour(EvtformerThree):
  """EvtformerThree with stable anatomical-group coordinate refinement."""

  def __init__(self, nInChannel: int, nOutChannel: int):
    super().__init__(nInChannel, nOutChannel)
    self.coordinate_refine = GroupedCoordinateResidualHead(channels=128)
    self.coordinate_refine.apply(_init_weights)

    # Preserve a zero correction at initialization.
    final = self.coordinate_refine.refine[-1]
    nn.init.zeros_(final.weight)
    nn.init.zeros_(final.bias)
