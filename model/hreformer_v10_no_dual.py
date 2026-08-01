"""HreformerV10 ablation without time-frequency DualToken."""

import torch.nn as nn

from model.hreformer_v10 import HreformerV10


class HreformerV10NoDual(HreformerV10):
  """Pass raw CSI features directly to the shared antenna encoder."""

  def __init__(self, nInChannel: int, nOutChannel: int):
    super().__init__(nInChannel, nOutChannel)
    self.dual_token = nn.Identity()
