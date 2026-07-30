"""HreformerV4: V1 trunk with one post-EVT detail residual."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.evtformer import Evtformer, _init_weights
from model.weak_antenna_fusion import WeakAdaptiveAntennaFusion


class HreformerV4(Evtformer):
  """Add high-resolution features after EVT and keep one shared decoder."""

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
    self.detail_to_evt = nn.Sequential(
      nn.Conv2d(detail_channels, evt_channels, kernel_size=1, bias=False),
      nn.BatchNorm2d(evt_channels),
      nn.ReLU(inplace=True),
    )

    self.detail_gate_logit = nn.Parameter(torch.tensor(-2.0))
    self.detail_gate_max = 0.05

    self.detail_lateral.apply(_init_weights)
    self.detail_to_evt.apply(_init_weights)

  def _encode_dual_resolution(
    self,
    x: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    x = self.encoder_conv1(x)
    x = self.encoder_bn1(x)
    x = self.encoder_relu(x)
    x = self.encoder_layer1(x)
    x = self.encoder_layer2(x)

    detail = self.detail_lateral(x)
    semantic = self.encoder_layer3(x)
    semantic = self.encoder_layer4(semantic)
    return detail, semantic

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x[:, :, :, 0, :]
    x = self.dual_token(x)
    x = x.permute(0, 3, 2, 1).unsqueeze(1)

    detail_features = []
    semantic_features = []
    for antenna_idx in range(3):
      branch = x[:, :, antenna_idx, :, :]
      branch = self.upsample136x32(branch)
      detail, semantic = self._encode_dual_resolution(branch)
      detail_features.append(detail)
      semantic_features.append(semantic)

    detail_features = torch.stack(detail_features, dim=1)
    semantic_features = torch.stack(semantic_features, dim=1)
    antenna_weights = self.antenna_fusion.antenna_weights(semantic_features)
    detail = self.antenna_fusion.fuse(detail_features, antenna_weights)
    semantic = self.antenna_fusion.fuse(semantic_features, antenna_weights)

    semantic = self.channel_reduce(semantic)
    semantic = self.spatial_upsample(semantic)
    evt_features = self.bn2(semantic)
    evt_features = self.evt(evt_features)

    detail = self.detail_to_evt(detail)
    detail = F.interpolate(
      detail,
      size=evt_features.shape[-2:],
      mode="bilinear",
      align_corners=False,
    )
    detail_gate = self.detail_gate_max * torch.sigmoid(
      self.detail_gate_logit
    )
    x = evt_features + detail_gate * detail

    x = self.decode(x)
    x = self.final_pool(x)
    x = x.squeeze(dim=3)
    x = self.bn1(x)
    return torch.transpose(x, 1, 2)
