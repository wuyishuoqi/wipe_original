"""Hreformer V1: DualToken + relative antenna fusion + weak-gated EVT."""

import torch
import torch.nn as nn

from model.evtformer import Evtformer
from model.relative_antenna_fusion import RelativeAntennaFusion


class Hreformer(Evtformer):
  """First-stage Sonnet-free Evtformer with adaptive antenna fusion.

  V1 intentionally keeps the proven Evtformer encoder, weak-gated EVT stack,
  coordinate decoder, and output contract unchanged. Only SonnetFusion and
  antenna concatenation are replaced by RelativeAntennaFusion.
  """

  def __init__(self, nInChannel: int, nOutChannel: int):
    super().__init__(nInChannel, nOutChannel)
    self.sonnet = nn.Identity()
    self.antenna_fusion = RelativeAntennaFusion(
      channels=512,
      n_antennas=3,
      attention_dim=128,
      num_heads=4,
      relative_gate_init=-3.0,
      relative_gate_max=0.25,
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

    x = self.decode(x)
    x = self.final_pool(x)
    x = x.squeeze(dim=3)
    x = self.bn1(x)
    return torch.transpose(x, 1, 2)
