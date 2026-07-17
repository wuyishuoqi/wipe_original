"""Weak-gated Evtformer ablation without SonnetFusion."""

import torch.nn as nn

from model.evtformer import Evtformer


class EvtformerNoSonnet(Evtformer):
  """Keep the full weak-gated model while bypassing only SonnetFusion."""

  def __init__(self, nInChannel: int, nOutChannel: int):
    super().__init__(nInChannel, nOutChannel)
    self.sonnet = nn.Identity()
