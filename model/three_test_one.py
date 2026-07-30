"""ThreeTestOne: EvtformerThree with only grouped residual refinement."""

import torch.nn as nn

from model.evtformer import _init_weights
from model.evtformer_four import GroupedCoordinateResidualHead
from model.evtformer_three import EvtformerThree


class ThreeTestOne(EvtformerThree):
  """Ablate the grouped residual head while retaining the Three loss."""

  def __init__(self, nInChannel: int, nOutChannel: int):
    super().__init__(nInChannel, nOutChannel)
    self.coordinate_refine = GroupedCoordinateResidualHead(channels=128)
    self.coordinate_refine.apply(_init_weights)

    # Match Three/Four: residual correction starts exactly at zero.
    final = self.coordinate_refine.refine[-1]
    nn.init.zeros_(final.weight)
    nn.init.zeros_(final.bias)
