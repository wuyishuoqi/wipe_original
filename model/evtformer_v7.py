"""EvtformerV7: DualToken + SonnetFusion + EVT Refinement.

Architecture:
  - DualToken (single-stage cross-attn): time-freq tokenization
  - SonnetFusion: Wavelet+Coh+Koopman per-antenna processing
  - EVT spatial attention x2: geometry-aware refinement

Flow:
  CSI -> DualToken(single-stage) -> 3×ResNet34 -> concat
    -> SonnetFusion -> upsample(17×24) -> EVT×2 -> decode -> [B,17,2]
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision

from model.dual_token_v7 import TimeFreqDualTokenV7
from model.sonnet_fusion import SonnetFusion
from model.evt import EVTSpatialAttention


def _make_evt_stack(channels, heads, gamma, layers):
  return nn.Sequential(*[
    EVTSpatialAttention(channels=channels, num_heads=heads, gamma_init=gamma)
    for _ in range(layers)
  ])


class EvtformerV7(nn.Module):

  def __init__(self, nInChannel: int, nOutChannel: int):
    super().__init__()

    evt_channels: int = 128
    evt_heads: int = 4
    gamma_init: float = 0.15
    num_evt_layers: int = 2
    n_atoms: int = 32
    up_h: int = 17
    up_w: int = 24

    self.dual_token = TimeFreqDualTokenV7(
      n_timesteps=9, n_subcarriers=114, n_antennas=3,
      d_model=128, n_heads=4, n_layers=2,
    )

    self.upsample136x32 = nn.Upsample((136, 32))

    resnet = torchvision.models.resnet34(pretrained=True)
    self.encoder_conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    self.encoder_bn1 = resnet.bn1
    self.encoder_relu = resnet.relu
    self.encoder_maxpool = resnet.maxpool
    self.encoder_layer1 = resnet.layer1
    self.encoder_layer2 = resnet.layer2
    self.encoder_layer3 = resnet.layer3
    self.encoder_layer4 = resnet.layer4

    self.sonnet = SonnetFusion(
      channels=512, antenna_width=4, n_atoms=n_atoms,
    )

    self.channel_reduce = nn.Sequential(
      nn.Conv2d(512, evt_channels, kernel_size=1, bias=False),
      nn.BatchNorm2d(evt_channels),
      nn.ReLU(inplace=True),
    )
    self.spatial_upsample = nn.Upsample((up_h, up_w), mode="bilinear")
    self.evt = _make_evt_stack(evt_channels, evt_heads, gamma_init, num_evt_layers)
    self.bn2 = nn.BatchNorm2d(evt_channels)

    self.decode = nn.Sequential(
      nn.Conv2d(evt_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True),
      nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0, bias=False),
      nn.BatchNorm2d(2),
      nn.ReLU(inplace=True),
    )
    self.final_pool = nn.AvgPool2d((up_h // 17, up_w))
    self.bn1 = nn.BatchNorm1d(2)

  def _encode_branch(self, x: torch.Tensor) -> torch.Tensor:
    x = self.encoder_conv1(x)
    x = self.encoder_bn1(x)
    x = self.encoder_relu(x)
    x = self.encoder_layer1(x)
    x = self.encoder_layer2(x)
    x = self.encoder_layer3(x)
    x = self.encoder_layer4(x)
    return x

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x[:, :, :, 0, :]
    x = self.dual_token(x)
    x = x.permute(0, 3, 2, 1)
    x = x[:, np.newaxis, :, :, :]

    x1 = x[:, :, 0, :, :]
    x2 = x[:, :, 1, :, :]
    x3 = x[:, :, 2, :, :]

    x1 = self.upsample136x32(x1)
    x2 = self.upsample136x32(x2)
    x3 = self.upsample136x32(x3)

    x1 = self._encode_branch(x1)
    x2 = self._encode_branch(x2)
    x3 = self._encode_branch(x3)

    x = torch.cat([x1, x2, x3], dim=3)
    x = self.sonnet(x)
    x = self.channel_reduce(x)
    x = self.spatial_upsample(x)
    x = self.bn2(x)
    x = self.evt(x)
    x = self.decode(x)

    x = self.final_pool(x)
    x = x.squeeze(dim=3)
    x = self.bn1(x)
    x = torch.transpose(x, 1, 2)
    return x
