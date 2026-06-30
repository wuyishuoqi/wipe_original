from model.unet import UNet
from model import resnet

from torch import Tensor
import torch.nn as nn


class Piw(nn.Module):
  def __init__(self, nInChannel: int, nOutChannel1: int, nOutChannel2: int):
    super().__init__()

    self.stage0 = nn.Sequential(
      nn.Upsample((120, 104), mode="bilinear"),
      resnet.BasicBlock(nInChannel),
    )

    self.branch1 = Branch(nInChannel, nOutChannel1)
    self.branch2 = Branch(nInChannel, nOutChannel2)

  def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
    shape = x.shape
    x = x.reshape((shape[0], -1, 3, 3))

    out = self.stage0(x)
    branch1out = self.branch1(out)
    branch2out = self.branch2(out)

    return branch1out, branch2out


class Branch(nn.Module):
  def __init__(self, nInChannel: int, nOutChannel: int):
    super().__init__()

    # nMidChannels = 3 * 30
    nMidChannels = 128

    self.branch = nn.Sequential(
      UNet(nInChannel, nMidChannels),
      nn.AvgPool2d((2, 1)),
      nn.Conv2d(nMidChannels, nMidChannels, 3, bias=False),
      nn.BatchNorm2d(nMidChannels),
      nn.ReLU(),
      nn.Conv2d(nMidChannels, nMidChannels, 3, bias=False),
      nn.BatchNorm2d(nMidChannels),
      nn.ReLU(),
      nn.Conv2d(nMidChannels, nOutChannel, 1, bias=False),
      nn.BatchNorm2d(nOutChannel),
      nn.ReLU(),
      nn.Conv2d(nOutChannel, nOutChannel, 1),
      nn.Sigmoid(),
    )

  def forward(self, x: Tensor) -> tuple[Tensor]:
    out = self.branch(x)
    return out
