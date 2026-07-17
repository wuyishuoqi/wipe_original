from model.piw import Piw
from model.wpnet import Wpnet
from model.wisppn import Wisppn
from model.wpformer import Wpformer
from model.evtformer import Evtformer
from model.evtformer_b1 import EvtformerB1
from model.evtformer_no_dual import EvtformerNoDual
from model.evtformer_no_evt import EvtformerNoEVT
from model.evtformer_no_sonnet import EvtformerNoSonnet
from model.evtformer_v8 import EvtformerV8
from model.evtformer_v4 import EvtformerV4
from model.evtformer_v7 import EvtformerV7

from model.unet import UNet
from model import resnet

from torch import Tensor
import torch.nn as nn


class Unet3x3(nn.Module):
  def __init__(self, nInChannel: int, nOutChannel: int):
    super().__init__()

    block: list[nn.Module] = []

    block += [
      nn.Upsample((120, 104), mode="bilinear"),
      resnet.BasicBlock(nInChannel),
    ]

    # nMidChannels = 3 * 30
    nMidChannels = 128
    block += [
      UNet(nInChannel, nMidChannels),
      nn.AvgPool2d((2, 1)),
      nn.Conv2d(nMidChannels, nMidChannels, 3, bias=False),
      nn.BatchNorm2d(nMidChannels),
      nn.ReLU(),
      nn.Conv2d(nMidChannels, nMidChannels, 3, bias=False),
      nn.BatchNorm2d(nMidChannels),
      nn.ReLU(),
    ]

    block += [
      nn.Conv2d(nMidChannels, nOutChannel, 1, bias=False),
      nn.BatchNorm2d(nOutChannel),
      nn.ReLU(),
      nn.Conv2d(nOutChannel, nOutChannel, 1),
      nn.Sigmoid(),
    ]

    self.f = nn.Sequential(*block)

  def forward(self, x: Tensor) -> Tensor:
    shape = x.shape
    x = x.reshape((shape[0], -1, 3, 3))

    y = self.f(x)
    return y
