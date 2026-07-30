"""HreformerV5: RMS-aligned post-EVT high-resolution residual."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.hreformer_v4 import HreformerV4


class HreformerV5(HreformerV4):
  """Make the detail gate represent a bounded feature-energy ratio."""

  def __init__(self, nInChannel: int, nOutChannel: int):
    super().__init__(nInChannel, nOutChannel)
    self.detail_norm = nn.GroupNorm(1, 128, affine=False)
    self.detail_gate_max = 0.03

  def _align_detail_scale(
    self,
    detail: torch.Tensor,
    evt_features: torch.Tensor,
  ) -> torch.Tensor:
    detail = self.detail_norm(detail)
    evt_rms = evt_features.detach().square().mean(
      dim=(1, 2, 3), keepdim=True
    ).add(1e-6).sqrt()
    return detail * evt_rms

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x[:, :, :, 0, :]
    x = self.dual_token(x)
    x = x.permute(0, 3, 2, 1).unsqueeze(1)

    detail_features = []
    semantic_features = []
    for antenna_idx in range(3):
      branch = x[:, :, antenna_idx, :, :]
      branch = self.upsample136x32(branch)
      detail, semantic = self._encode_dual_resolution(branch)
      detail_features.append(detail)
      semantic_features.append(semantic)

    detail_features = torch.stack(detail_features, dim=1)
    semantic_features = torch.stack(semantic_features, dim=1)
    antenna_weights = self.antenna_fusion.antenna_weights(semantic_features)
    detail = self.antenna_fusion.fuse(detail_features, antenna_weights)
    semantic = self.antenna_fusion.fuse(semantic_features, antenna_weights)

    semantic = self.channel_reduce(semantic)
    semantic = self.spatial_upsample(semantic)
    evt_features = self.bn2(semantic)
    evt_features = self.evt(evt_features)

    detail = self.detail_to_evt(detail)
    detail = F.interpolate(
      detail,
      size=evt_features.shape[-2:],
      mode="bilinear",
      align_corners=False,
    )
    detail = self._align_detail_scale(detail, evt_features)
    detail_gate = self.detail_gate_max * torch.sigmoid(
      self.detail_gate_logit
    )
    x = evt_features + detail_gate * detail

    x = self.decode(x)
    x = self.final_pool(x)
    x = x.squeeze(dim=3)
    x = self.bn1(x)
    return torch.transpose(x, 1, 2)
