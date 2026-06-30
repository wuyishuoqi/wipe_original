from torch import Tensor

import torch.nn as nn


def convLayer(nChannels, kernelSize, is3d: bool) -> nn.Module:
  if is3d:
    Conv = nn.Conv3d
  else:
    Conv = nn.Conv2d
  return Conv(nChannels, nChannels, kernelSize, padding="same", bias=False)


def batchNormLayer(nChannels, is3d: bool) -> nn.Module:
  if is3d:
    BatchNorm = nn.BatchNorm3d
  else:
    BatchNorm = nn.BatchNorm2d
  return BatchNorm(nChannels)


class BasicBlock(nn.Module):
  def __init__(
    self,
    nChannels: int,
    kernelSize: int = 3,
    activation=nn.ReLU(inplace=True),
    is3d: bool = False,
  ) -> None:
    super().__init__()

    self.conv1 = nn.Sequential(
      convLayer(nChannels, kernelSize, is3d),
      batchNormLayer(nChannels, is3d),
    )
    self.activation = activation
    self.conv2 = nn.Sequential(
      convLayer(nChannels, kernelSize, is3d),
      batchNormLayer(nChannels, is3d),
    )

  def forward(self, x: Tensor) -> Tensor:
    residual = x

    out = self.conv1(x)
    out = self.activation(out)
    out = self.conv2(out)

    out += residual
    out = self.activation(out)

    return out
