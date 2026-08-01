"""Standalone HreformerV10 model definition.

Flow:
  CSI -> DualToken -> shared ResNet34 antenna branches
    -> relative antenna fusion -> weak-gated EVT
    -> RMS-limited joint graph refinement -> coordinate decoder

The complete architecture is declared here, while its late learning-rate
schedule and coordinate BatchNorm freeze remain in TrainerHreformerV10.
"""

import math

import torch
import torch.nn as nn
import torchvision

try:
  from torchvision.models import ResNet34_Weights
except ImportError:
  ResNet34_Weights = None

from model.dual_token import TimeFreqDualToken
from model.evt import EVTSpatialAttention
from model.joint_graph_refinement import (
  RmsNormalizedJointTokenGraphRefinement,
)
from model.relative_antenna_fusion import RelativeAntennaFusion


def _init_weights(module: nn.Module):
  if isinstance(module, nn.Conv2d):
    nn.init.xavier_normal_(module.weight.data)
  elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
    nn.init.constant_(module.weight, 1)
    nn.init.constant_(module.bias, 0)
  elif isinstance(module, nn.Linear):
    nn.init.xavier_uniform_(module.weight.data)
    if module.bias is not None:
      nn.init.constant_(module.bias, 0)


class StabilizedEVTSpatialAttention(EVTSpatialAttention):
  """EVT attention with a bounded spatial-prior strength."""

  def __init__(
    self,
    channels: int,
    num_heads: int = 8,
    gamma_init: float = 0.15,
    gamma_max: float = 0.6,
  ):
    super().__init__(
      channels=channels,
      num_heads=num_heads,
      gamma_init=gamma_init,
    )
    self.gamma_max = gamma_max
    self.gamma.register_hook(lambda grad: grad.clamp(-0.05, 0.05))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = x.shape
    n_tokens = height * width

    tokens = x.flatten(2).transpose(1, 2)
    shortcut = tokens
    normalized = self.norm1(tokens)

    qkv = self.qkv(normalized).reshape(
      batch, n_tokens, 3, self.num_heads, self.head_dim
    ).permute(2, 0, 3, 1, 4)
    query, key, value = qkv[0], qkv[1], qkv[2]

    attention = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
    distance = self._build_spatial_decay(
      height, width, x.device, x.dtype
    )
    gamma = self.gamma.clamp(0.0, self.gamma_max)
    attention = attention - gamma * distance.unsqueeze(0).unsqueeze(0)
    attention = attention.softmax(dim=-1)

    out = attention @ value
    out = out.transpose(1, 2).reshape(batch, n_tokens, channels)
    out = self.proj(out)
    out = shortcut + out
    out = out + self.mlp(self.norm2(out))
    return out.transpose(1, 2).reshape(batch, channels, height, width)


class WeakGatedEVTStack(nn.Module):
  """One full EVT block followed by a bounded weak second block."""

  def __init__(
    self,
    channels: int,
    heads: int,
    gamma_init: float,
    second_gate_init: float = -3.0,
    second_gate_max: float = 0.25,
  ):
    super().__init__()
    self.primary = StabilizedEVTSpatialAttention(
      channels=channels,
      num_heads=heads,
      gamma_init=gamma_init,
    )
    self.secondary = StabilizedEVTSpatialAttention(
      channels=channels,
      num_heads=heads,
      gamma_init=gamma_init,
    )
    self.second_gate_logit = nn.Parameter(torch.tensor(second_gate_init))
    self.second_gate_max = second_gate_max

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    primary = self.primary(x)
    secondary = self.secondary(primary)
    gate = self.second_gate_max * torch.sigmoid(self.second_gate_logit)
    return primary + gate * (secondary - primary)


class HreformerV10(nn.Module):
  """HreformerV10 without dependencies on earlier Hreformer versions."""

  def __init__(self, nInChannel: int, nOutChannel: int):
    super().__init__()

    # The input/output arguments are retained for the project's model API.
    del nInChannel, nOutChannel

    evt_channels = 128
    evt_heads = 4
    gamma_init = 0.15
    up_h = 17
    up_w = 24

    self.dual_token = TimeFreqDualToken(
      n_timesteps=9,
      n_subcarriers=114,
      n_antennas=3,
      d_model=128,
      n_heads=4,
      n_layers=2,
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

    self.antenna_fusion = RelativeAntennaFusion(
      channels=512,
      n_antennas=3,
      attention_dim=128,
      num_heads=4,
      relative_gate_init=-3.0,
      relative_gate_max=0.25,
    )

    self.channel_reduce = nn.Sequential(
      nn.Conv2d(512, evt_channels, kernel_size=1, bias=False),
      nn.BatchNorm2d(evt_channels),
      nn.ReLU(inplace=True),
    )
    self.spatial_upsample = nn.Upsample(
      (up_h, up_w), mode="bilinear"
    )
    self.bn2 = nn.BatchNorm2d(evt_channels)

    self.evt = WeakGatedEVTStack(
      channels=evt_channels,
      heads=evt_heads,
      gamma_init=gamma_init,
    )

    self.joint_graph = RmsNormalizedJointTokenGraphRefinement(
      channels=evt_channels,
      n_joints=17,
      residual_gate_init=-2.0,
      residual_gate_max=0.005,
    )

    self.decode = nn.Sequential(
      nn.Conv2d(
        evt_channels, 32, kernel_size=3, stride=1, padding=1, bias=False
      ),
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
    x = x.permute(0, 3, 2, 1).unsqueeze(1)

    antenna_features = []
    for antenna_idx in range(3):
      branch = x[:, :, antenna_idx, :, :]
      branch = self.upsample136x32(branch)
      antenna_features.append(self._encode_branch(branch))

    x = torch.stack(antenna_features, dim=1)
    x = self.antenna_fusion(x)
    x = self.channel_reduce(x)
    x = self.spatial_upsample(x)
    x = self.bn2(x)
    x = self.evt(x)

    joint_tokens = x.mean(dim=3).transpose(1, 2)
    graph_residual = self.joint_graph(joint_tokens)
    graph_residual = graph_residual.transpose(1, 2).unsqueeze(3)
    x = x + graph_residual

    x = self.decode(x)
    x = self.final_pool(x)
    x = x.squeeze(dim=3)
    x = self.bn1(x)
    return x.transpose(1, 2)
