from model import resnet

import torch
import torch.nn as nn


class Wpnet(nn.Module):
  def __init__(self, nInChannel: int, nOutChannel: int):
    super().__init__()

    self.upsample = nn.Upsample((136, 136))

    self.block1 = nn.Sequential(
      nn.Conv2d(1, 64, 3, padding="same"),
      nn.BatchNorm2d(64),
      nn.ReLU(),
    )
    self.block2 = nn.Sequential(
      resnet.BasicBlock(64),
      nn.Conv2d(64, 64, 3, padding="same"),
      nn.BatchNorm2d(64),
      nn.ReLU(),
    )
    self.block3 = nn.Sequential(
      nn.Conv2d(64, 128, 1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      resnet.BasicBlock(128),
      resnet.BasicBlock(128),
      nn.AvgPool2d(2),
    )
    self.block4 = nn.Sequential(
      nn.Conv2d(128, 256, 1),
      nn.BatchNorm2d(256),
      nn.ReLU(),
      resnet.BasicBlock(256),
      resnet.BasicBlock(256),
      resnet.BasicBlock(256),
      nn.AvgPool2d(2),
    )
    self.block5 = nn.Sequential(
      nn.Conv2d(256, 512, 1),
      nn.BatchNorm2d(512),
      nn.ReLU(),
      resnet.BasicBlock(512),
      nn.Conv2d(512, 512, 1),
      nn.BatchNorm2d(512),
      nn.ReLU(),
      nn.AvgPool2d(2),
    )
    self.bottleneck = nn.Sequential(
      nn.Conv2d(512, 2, 1),
      nn.BatchNorm2d(2),
      nn.ReLU(),
      # output
      nn.AvgPool2d((1, 17)),
    )

  def forward(self, x: torch.Tensor):
    shape = x.shape
    x = x.reshape((shape[0], 1, -1, 9))
    y = self.upsample(x)

    y = self.block1(y)
    y = self.block2(y)
    y = self.block3(y)
    y = self.block4(y)
    y = self.block5(y)
    y = self.bottleneck(y)
    y = y.squeeze(-1).transpose(-1, -2)
    return y
