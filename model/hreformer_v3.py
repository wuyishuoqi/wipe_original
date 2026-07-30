"""HreformerV3: constrained dual resolution with an EVT detail bypass."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.constrained_resolution_fusion import (
  ConstrainedDualResolutionExchange,
)
from model.evtformer import _init_weights
from model.hreformer_v2 import HreformerV2


class HreformerV3(HreformerV2):
  """Preserve high-resolution details through a small coordinate residual."""

  def __init__(self, nInChannel: int, nOutChannel: int):
    super().__init__(nInChannel, nOutChannel)

    self.resolution_exchange = ConstrainedDualResolutionExchange(
      high_channels=64,
      low_channels=512,
      low_to_high_gate_max=0.05,
      high_to_low_gate_max=0.025,
      gate_init=-2.0,
    )

    # Keep pre-EVT detail injection weaker than V2.
    self.detail_gate_max = 0.05

    self.detail_residual_head = nn.Sequential(
      nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True),
      nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0, bias=False),
    )
    self.coordinate_residual_gate_logit = nn.Parameter(torch.tensor(-2.0))
    self.coordinate_residual_gate_max = 0.10

    self.resolution_exchange.apply(_init_weights)
    self.detail_residual_head.apply(_init_weights)
    nn.init.zeros_(self.detail_residual_head[-1].weight)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x[:, :, :, 0, :]
    x = self.dual_token(x)
    x = x.permute(0, 3, 2, 1).unsqueeze(1)

    high_features = []
    low_features = []
    for antenna_idx in range(3):
      branch = x[:, :, antenna_idx, :, :]
      branch = self.upsample136x32(branch)
      high, low = self._encode_dual_resolution(branch)
      high_features.append(high)
      low_features.append(low)

    high_features = torch.stack(high_features, dim=1)
    low_features = torch.stack(low_features, dim=1)
    antenna_weights = self.antenna_fusion.antenna_weights(low_features)
    high = self.antenna_fusion.fuse(high_features, antenna_weights)
    low = self.antenna_fusion.fuse(low_features, antenna_weights)

    high, low = self.resolution_exchange(high, low)

    semantic = self.channel_reduce(low)
    semantic = self.spatial_upsample(semantic)
    detail = self.detail_to_evt(high)
    detail = F.interpolate(
      detail,
      size=semantic.shape[-2:],
      mode="bilinear",
      align_corners=False,
    )

    detail_gate = self.detail_gate_max * torch.sigmoid(
      self.detail_gate_logit
    )
    evt_features = self.bn2(semantic + detail_gate * detail)
    evt_features = self.evt(evt_features)

    coordinates = self.decode(evt_features)
    coordinates = self.final_pool(coordinates)
    coordinates = coordinates.squeeze(dim=3)
    coordinates = self.bn1(coordinates)
    coordinates = torch.transpose(coordinates, 1, 2)

    detail_residual = self.detail_residual_head(detail)
    detail_residual = self.final_pool(detail_residual)
    detail_residual = detail_residual.squeeze(dim=3)
    detail_residual = torch.transpose(detail_residual, 1, 2)
    residual_gate = self.coordinate_residual_gate_max * torch.sigmoid(
      self.coordinate_residual_gate_logit
    )
    return coordinates + residual_gate * detail_residual
