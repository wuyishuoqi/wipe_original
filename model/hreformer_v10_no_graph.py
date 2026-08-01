"""HreformerV10 ablation without COCO joint-graph refinement."""

import torch
import torch.nn as nn

from model.hreformer_v10 import HreformerV10


class ZeroJointGraphRefinement(nn.Module):
  """Return a zero residual while preserving the V10 forward contract."""

  def forward(self, joint_tokens: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(joint_tokens)


class HreformerV10NoGraph(HreformerV10):
  """Decode weak-gated EVT features without skeleton propagation."""

  def __init__(self, nInChannel: int, nOutChannel: int):
    super().__init__(nInChannel, nOutChannel)
    self.joint_graph = ZeroJointGraphRefinement()
