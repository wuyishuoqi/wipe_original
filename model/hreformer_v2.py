"""HreformerV2: weak antenna fusion with dual-resolution CSI features."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.dual_resolution_fusion import DualResolutionExchange
from model.evtformer import Evtformer, _init_weights
from model.weak_antenna_fusion import WeakAdaptiveAntennaFusion


class HreformerV2(Evtformer):
  """Add a conservative high-resolution detail path to Hreformer V1."""

  def __init__(self, nInChannel: int, nOutChannel: int):
    super().__init__(nInChannel, nOutChannel)

    detail_channels = 64
    evt_channels = 128
    self.sonnet = nn.Identity()
    self.antenna_fusion = WeakAdaptiveAntennaFusion(
      reference_channels=512,
      n_antennas=3,
    )

    self.detail_lateral = nn.Sequential(
      nn.Conv2d(128, detail_channels, kernel_size=1, bias=False),
      nn.BatchNorm2d(detail_channels),
      nn.ReLU(inplace=True),
    )
    self.resolution_exchange = DualResolutionExchange(
      high_channels=detail_channels,
      low_channels=512,
      exchange_gate_max=0.25,
    )
    self.detail_to_evt = nn.Sequential(
      nn.Conv2d(detail_channels, evt_channels, kernel_size=1, bias=False),
      nn.BatchNorm2d(evt_channels),
      nn.ReLU(inplace=True),
    )

    self.detail_gate_logit = nn.Parameter(torch.tensor(-2.0))
    self.detail_gate_max = 0.25

    for module in (
      self.detail_lateral,
      self.resolution_exchange,
      self.detail_to_evt,
    ):
      module.apply(_init_weights)

  def _encode_dual_resolution(
    self,
    x: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    x = self.encoder_conv1(x)
    x = self.encoder_bn1(x)
    x = self.encoder_relu(x)
    x = self.encoder_layer1(x)
    x = self.encoder_layer2(x)

    high = self.detail_lateral(x)
    low = self.encoder_layer3(x)
    low = self.encoder_layer4(low)
    return high, low

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
    detail_gate = self.detail_gate_max * torch.sigmoid(self.detail_gate_logit)
    x = semantic + detail_gate * detail

    x = self.bn2(x)
    x = self.evt(x)
    x = self.decode(x)
    x = self.final_pool(x)
    x = x.squeeze(dim=3)
    x = self.bn1(x)
    return torch.transpose(x, 1, 2)
