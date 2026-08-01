"""HreformerV10 ablation without relative antenna fusion."""

import torch
import torch.nn as nn

from model.hreformer_v10 import HreformerV10


class MeanAntennaFusion(nn.Module):
  """Fuse the three antenna branches using a fixed equal-weight mean."""

  def forward(self, features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 5:
      raise ValueError(
        f"expected antenna features [B, A, C, H, W], got {features.shape}"
      )
    return features.mean(dim=1)


class HreformerV10NoRAF(HreformerV10):
  """Replace learned antenna reliability and attention with a fixed mean."""

  def __init__(self, nInChannel: int, nOutChannel: int):
    super().__init__(nInChannel, nOutChannel)
    self.antenna_fusion = MeanAntennaFusion()
