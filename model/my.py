from model.piw import Piw
from model.wpnet import Wpnet
from model.wisppn import Wisppn
from model.wpformer import Wpformer
from model.hreformer_v10 import HreformerV10
from model.hreformer_v10_no_dual import HreformerV10NoDual
from model.hreformer_v10_no_raf import HreformerV10NoRAF
from model.hreformer_v10_no_graph import HreformerV10NoGraph

# Historical Evtformer/Hreformer versions are optional. HreformerV10 remains
# available when those archived model files are removed from the project.
try:
  from model.evtformer import Evtformer
  from model.evtformer_b1 import EvtformerB1
  from model.evtformer_no_dual import EvtformerNoDual
  from model.evtformer_no_evt import EvtformerNoEVT
  from model.evtformer_no_sonnet import EvtformerNoSonnet
  from model.evt_ssd import EvtSsd
  from model.evtformer_two import EvtformerTwo
  from model.evtformer_three import EvtformerThree
  from model.evtformer_four import EvtformerFour
  from model.three_test_one import ThreeTestOne
  from model.evtformer_five import EvtformerFive
  from model.evtformer_six import EvtformerSix
  from model.hreformer import Hreformer
  from model.hreformer_v2 import HreformerV2
  from model.hreformer_v3 import HreformerV3
  from model.hreformer_v4 import HreformerV4
  from model.hreformer_v5 import HreformerV5
  from model.hreformer_v6 import HreformerV6
  from model.hreformer_v7 import HreformerV7
  from model.hreformer_v8 import HreformerV8
  from model.hreformer_v9 import HreformerV9
  from model.evtformer_v8 import EvtformerV8
  from model.evtformer_v4 import EvtformerV4
  from model.evtformer_v7 import EvtformerV7
except ModuleNotFoundError as error:
  optional_modules = {
    "model.evtformer",
    "model.hreformer",
    "model.hreformer_v2",
    "model.hreformer_v3",
    "model.hreformer_v4",
    "model.hreformer_v5",
    "model.hreformer_v6",
    "model.hreformer_v7",
    "model.hreformer_v8",
    "model.hreformer_v9",
  }
  if error.name not in optional_modules:
    raise

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
