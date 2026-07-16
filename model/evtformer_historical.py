"""Shared architecture for the historical Evtformer v8 and B1 variants."""

import numpy as np
import torch
import torch.nn as nn
import torchvision

try:
  from torchvision.models import ResNet34_Weights
except ImportError:
  ResNet34_Weights = None

from model.dual_token import TimeFreqDualToken
from model.evt import EVTSpatialAttention
from model.evtformer import StabilizedEVTSpatialAttention, _init_weights
from model.sonnet_fusion import SonnetFusion


class HistoricalEvtformer(nn.Module):
  """Common v8 route with a selectable historical EVT configuration."""

  def __init__(
    self,
    nInChannel: int,
    nOutChannel: int,
    *,
    num_evt_layers: int,
    stabilize_evt: bool,
    stabilize_dual_token: bool,
  ):
    super().__init__()

    evt_channels = 128
    evt_heads = 4
    gamma_init = 0.15
    n_atoms = 32
    up_h = 17
    up_w = 24

    self.dual_token = TimeFreqDualToken(
      n_timesteps=9,
      n_subcarriers=114,
      n_antennas=3,
      d_model=128,
      n_heads=4,
      n_layers=2,
      stabilize_residual=stabilize_dual_token,
    )

    self.upsample136x32 = nn.Upsample((136, 32))

    if ResNet34_Weights is None:
      resnet = torchvision.models.resnet34(pretrained=True)
    else:
      resnet = torchvision.models.resnet34(
        weights=ResNet34_Weights.IMAGENET1K_V1
      )
    self.encoder_conv1 = nn.Conv2d(
      1, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    self.encoder_bn1 = resnet.bn1
    self.encoder_relu = resnet.relu
    self.encoder_maxpool = resnet.maxpool
    self.encoder_layer1 = resnet.layer1
    self.encoder_layer2 = resnet.layer2
    self.encoder_layer3 = resnet.layer3
    self.encoder_layer4 = resnet.layer4

    self.sonnet = SonnetFusion(
      channels=512,
      antenna_width=4,
      n_atoms=n_atoms,
    )

    self.channel_reduce = nn.Sequential(
      nn.Conv2d(512, evt_channels, kernel_size=1, bias=False),
      nn.BatchNorm2d(evt_channels),
      nn.ReLU(inplace=True),
    )
    self.spatial_upsample = nn.Upsample((up_h, up_w), mode="bilinear")

    evt_type = StabilizedEVTSpatialAttention if stabilize_evt else EVTSpatialAttention
    self.evt = nn.Sequential(*[
      evt_type(
        channels=evt_channels,
        num_heads=evt_heads,
        gamma_init=gamma_init,
      )
      for _ in range(num_evt_layers)
    ])

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

    self._init_new_layers()

  def _init_new_layers(self):
    modules = (
      self.dual_token,
      self.encoder_conv1,
      self.sonnet,
      self.channel_reduce,
      self.evt,
      self.bn2,
      self.decode,
      self.bn1,
    )
    for module in modules:
      module.apply(_init_weights)

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

    x1 = self.upsample136x32(x[:, :, 0, :, :])
    x2 = self.upsample136x32(x[:, :, 1, :, :])
    x3 = self.upsample136x32(x[:, :, 2, :, :])

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
    return torch.transpose(x, 1, 2)
