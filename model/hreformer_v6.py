"""HreformerV6: Hreformer V1 with joint-token graph refinement."""

import torch

from model.hreformer import Hreformer
from model.joint_graph_refinement import JointTokenGraphRefinement


class HreformerV6(Hreformer):
  """Refine post-EVT joint features through a directed COCO graph."""

  def __init__(self, nInChannel: int, nOutChannel: int):
    super().__init__(nInChannel, nOutChannel)
    self.joint_graph = JointTokenGraphRefinement(
      channels=128,
      n_joints=17,
      residual_gate_init=-3.0,
      residual_gate_max=0.05,
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x[:, :, :, 0, :]
    x = self.dual_token(x)
    x = x.permute(0, 3, 2, 1).unsqueeze(1)

    antenna_features = []
    for antenna_idx in range(3):
      branch = x[:, :, antenna_idx, :, :]
      branch = self.upsample136x32(branch)
      antenna_features.append(self._encode_branch(branch))

    x = torch.stack(antenna_features, dim=1)
    x = self.antenna_fusion(x)
    x = self.channel_reduce(x)
    x = self.spatial_upsample(x)
    x = self.bn2(x)
    x = self.evt(x)

    joint_tokens = x.mean(dim=3).transpose(1, 2)
    graph_residual = self.joint_graph(joint_tokens)
    graph_residual = graph_residual.transpose(1, 2).unsqueeze(3)
    x = x + graph_residual

    x = self.decode(x)
    x = self.final_pool(x)
    x = x.squeeze(dim=3)
    x = self.bn1(x)
    return torch.transpose(x, 1, 2)
