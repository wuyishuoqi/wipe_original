"""HreformerV9: V7 with softly gated wrist/ankle refinement."""

import torch

from model.distal_joint_refinement import SoftGatedDistalJointHead
from model.hreformer_v7 import HreformerV7


class HreformerV9(HreformerV7):
  """Add non-saturating distal corrections while preserving the V7 trunk."""

  def __init__(self, nInChannel: int, nOutChannel: int):
    super().__init__(nInChannel, nOutChannel)
    self.distal_head = SoftGatedDistalJointHead(
      channels=128,
      hidden_channels=64,
      n_joints=17,
      max_correction=0.25,
      gate_init=0.2,
      raw_scale=0.1,
      detach_tokens=True,
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
    refined_joint_tokens = joint_tokens + graph_residual
    x = x + graph_residual.transpose(1, 2).unsqueeze(3)

    x = self.decode(x)
    x = self.final_pool(x)
    x = x.squeeze(dim=3)
    x = self.bn1(x)
    coordinates = torch.transpose(x, 1, 2)
    return coordinates + self.distal_head(refined_joint_tokens)
