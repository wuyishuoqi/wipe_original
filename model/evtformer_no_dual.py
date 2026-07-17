"""Weak-gated Evtformer ablation without DualToken enhancement."""

import torch.nn as nn

from model.evtformer import Evtformer


class EvtformerNoDual(Evtformer):
  """Keep the full weak-gated model while bypassing only DualToken."""

  def __init__(self, nInChannel: int, nOutChannel: int):
    super().__init__(nInChannel, nOutChannel)
    self.dual_token = nn.Identity()
