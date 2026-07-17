"""Weak-gated Evtformer ablation without the complete EVT refinement stack."""

import torch.nn as nn

from model.evtformer import Evtformer


class EvtformerNoEVT(Evtformer):
  """Keep feature adaptation and decoding while bypassing both EVT blocks."""

  def __init__(self, nInChannel: int, nOutChannel: int):
    super().__init__(nInChannel, nOutChannel)
    self.evt = nn.Identity()
