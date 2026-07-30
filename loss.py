import torch
import torch.nn as nn
import torch.nn.functional as F


class MyLoss(nn.Module):
  def __init__(
    self,
    device: torch.device,
    mWeightJhmsParam: tuple[float, float],
    maskWeight: float,
    jhmsRegularNrom: int,
    jhmsRegularWeight: float,
    maskRegularNrom: int,
    maskRegularWeight: float,
  ):
    super().__init__()
    self.device = device

    self.jhmMweightK = mWeightJhmsParam[0]
    self.jhmMweightB = mWeightJhmsParam[1]
    self.jhmsRegularNrom = jhmsRegularNrom
    self.jhmsRegularWeight = jhmsRegularWeight

    self.maskWeight = maskWeight
    self.maskRegularNrom = maskRegularNrom
    self.maskRegularWeight = maskRegularWeight

  def forward(
    self,
    jhms: torch.Tensor,
    jhmsTarget: torch.Tensor,
    mask: torch.Tensor,
    maskTarget: torch.Tensor,
  ) -> tuple[torch.Tensor, tuple]:
    jhmsLoss = F.smooth_l1_loss(jhms, jhmsTarget, reduction="none")
    if self.jhmMweightK and self.jhmMweightB:
      thr = 0.5
      jhmsII = (
        torch.zeros(jhmsTarget.shape, device=self.device)
        .masked_fill(jhmsTarget >= thr, 1)
        .masked_fill(jhmsTarget < thr, 0)
      )
      mWeightJhm = self.jhmMweightK * jhmsTarget + self.jhmMweightB * jhmsII
      jhmsLoss = mWeightJhm * F.smooth_l1_loss(jhms, jhmsTarget, reduction="none")

    jhmsLoss = jhmsLoss.mean()

    if self.jhmsRegularWeight:
      jhmsRegularDim = tuple(range(1, jhms.dim()))
      jhmsRegular = jhms.norm(self.jhmsRegularNrom, jhmsRegularDim)
      jhmsLoss = jhmsLoss + self.jhmsRegularWeight * jhmsRegular.mean()

    if self.maskWeight:
      maskLoss = F.binary_cross_entropy(mask, maskTarget)
      if self.maskRegularWeight:
        maskRegularDim = tuple(range(1, mask.dim()))
        maskRegular = mask.norm(self.maskRegularNrom, maskRegularDim)
        maskLoss = maskLoss + self.maskRegularWeight * maskRegular.mean()
    else:
      maskLoss = torch.zeros(1, device=self.device)

    totalLoss = jhmsLoss + self.maskWeight * maskLoss
    lossVal = (totalLoss.item(), jhmsLoss.item(), maskLoss.item())
    return totalLoss, lossVal


class Piw(nn.Module):
  def __init__(self, device: torch.device):
    super().__init__()
    self.device = device
    self.smWeight = 0.1
    self.jhmWeight = 1
    self.pafWeight = 1

    self.mWeightJhmK = 1
    self.mWeightJhmB = 1

    self.mWeightPafK = 1
    self.mWeightPafB = 0.3

  def forward(
    self,
    jhms: torch.Tensor,
    jhmsTarget: torch.Tensor,
    mask: torch.Tensor,
    maskTarget: torch.Tensor,
    pafs: torch.Tensor,
    pafsTarget: torch.Tensor,
  ) -> tuple:
    thr = 0.5
    jhmII = (
      torch.zeros(jhmsTarget.shape)
      .to(self.device)
      .masked_fill(jhmsTarget >= thr, 1)
      .masked_fill(jhmsTarget < thr, 0)
    )
    pafII = (
      torch.zeros(pafsTarget.shape)
      .to(self.device)
      .masked_fill(pafsTarget >= thr, 1)
      .masked_fill(pafsTarget < thr, 0)
    )

    mWeightJhm = self.mWeightJhmK * jhmsTarget + self.mWeightJhmB * jhmII
    mWeightPaf = self.mWeightPafK * pafsTarget + self.mWeightPafB * pafII

    maskLoss = F.binary_cross_entropy(mask, maskTarget, reduction="none")
    jhmLoss = mWeightJhm * F.mse_loss(jhms, jhmsTarget, reduction="none")
    pafLoss = mWeightPaf * F.mse_loss(pafs, pafsTarget, reduction="none")

    totalLoss = (
      self.smWeight * maskLoss.mean()
      + self.jhmWeight * jhmLoss.mean()
      + self.pafWeight * pafLoss.mean()
    )
    lossValues = (
      totalLoss.item(),
      jhmLoss.mean().item(),
      maskLoss.mean().item(),
    )
    return totalLoss, lossValues


class Wpnet(nn.Module):
  def __init__(self, device: torch.device):
    super().__init__()
    self.device = device

  def forward(self, keypoints: torch.Tensor, keypointsT: torch.Tensor) -> tuple:
    loss = F.mse_loss(keypoints, keypointsT)
    lossValue = loss.item()
    return loss, lossValue


class Wpformer(nn.Module):
  def __init__(self, device: torch.device):
    super().__init__()
    self.device = device

  def forward(self, keypoints: torch.Tensor, keypointsT: torch.Tensor) -> tuple:
    loss = F.mse_loss(keypoints, keypointsT)
    lossValue = loss.item()
    return loss, lossValue


class EvtformerThree(nn.Module):
  """Coordinate loss that retains gradients for strict localization errors."""

  def __init__(self, device: torch.device):
    super().__init__()
    joint_weights = torch.tensor([
      1.05, 1.05, 1.05, 1.05, 1.05,
      1.00, 1.00, 1.10, 1.10, 1.25, 1.25,
      1.00, 1.00, 1.10, 1.10, 1.25, 1.25,
    ])
    joint_weights = joint_weights / joint_weights.mean()
    self.register_buffer(
      "joint_weights",
      joint_weights.view(1, 17, 1).to(device),
    )
    self.detail_weight = 0.15
    self.detail_beta = 0.5

  def forward(self, keypoints: torch.Tensor, keypointsT: torch.Tensor) -> tuple:
    squared_error = (keypoints - keypointsT).square()
    mse = (self.joint_weights * squared_error).mean()
    detail = F.smooth_l1_loss(
      keypoints,
      keypointsT,
      reduction="none",
      beta=self.detail_beta,
    )
    detail = (self.joint_weights * detail).mean()
    loss = mse + self.detail_weight * detail
    return loss, loss.item()


class EvtformerFour(nn.Module):
  """Strict coordinate loss with light kinematic-vector consistency."""

  def __init__(self, device: torch.device):
    super().__init__()
    joint_weights = torch.tensor([
      1.15, 1.15, 1.15, 1.10, 1.10,
      0.95, 0.95, 1.12, 1.12, 1.35, 1.35,
      0.95, 0.95, 1.12, 1.12, 1.35, 1.35,
    ])
    joint_weights = joint_weights / joint_weights.mean()
    self.register_buffer(
      "joint_weights",
      joint_weights.view(1, 17, 1).to(device),
    )
    self.register_buffer(
      "bone_parents",
      torch.tensor(
        (0, 0, 1, 2, 5, 7, 6, 8, 5, 6, 5, 11, 11, 13, 12, 14),
        device=device,
      ),
    )
    self.register_buffer(
      "bone_children",
      torch.tensor(
        (1, 2, 3, 4, 7, 9, 8, 10, 6, 12, 11, 12, 13, 15, 14, 16),
        device=device,
      ),
    )
    self.detail_weight = 0.15
    self.detail_beta = 0.5
    self.bone_weight = 0.03

  def forward(self, keypoints: torch.Tensor, keypointsT: torch.Tensor) -> tuple:
    squared_error = (keypoints - keypointsT).square()
    mse = (self.joint_weights * squared_error).mean()

    detail = F.smooth_l1_loss(
      keypoints,
      keypointsT,
      reduction="none",
      beta=self.detail_beta,
    )
    detail = (self.joint_weights * detail).mean()

    predicted_bones = (
      keypoints[:, self.bone_children] - keypoints[:, self.bone_parents]
    )
    target_bones = (
      keypointsT[:, self.bone_children] - keypointsT[:, self.bone_parents]
    )
    bone = F.smooth_l1_loss(
      predicted_bones,
      target_bones,
      beta=1.0,
    )

    loss = mse + self.detail_weight * detail + self.bone_weight * bone
    return loss, loss.item()


class EvtformerFive(nn.Module):
  """Three loss plus curriculum PCK@0.05/0.10 failure surrogates."""

  def __init__(self, device: torch.device):
    super().__init__()
    joint_weights = torch.tensor([
      1.05, 1.05, 1.05, 1.05, 1.05,
      1.00, 1.00, 1.10, 1.10, 1.25, 1.25,
      1.00, 1.00, 1.10, 1.10, 1.25, 1.25,
    ])
    joint_weights = joint_weights / joint_weights.mean()
    self.register_buffer(
      "joint_weights",
      joint_weights.view(1, 17, 1).to(device),
    )
    self.detail_weight = 0.15
    self.detail_beta = 0.5
    self.pck05_weight = 0.12
    self.pck10_weight = 0.025
    self.temperature = 0.01
    self.curriculum_scale = 0.0

  def set_epoch(self, epoch: int):
    if epoch < 10:
      self.curriculum_scale = 0.0
    elif epoch == 10:
      self.curriculum_scale = 0.25
    elif epoch == 11:
      self.curriculum_scale = 0.50
    elif epoch == 12:
      self.curriculum_scale = 0.75
    else:
      self.curriculum_scale = 1.0

  def _soft_failure(self, distance: torch.Tensor, threshold: float):
    return torch.sigmoid(
      (distance - threshold) / self.temperature
    ).mean()

  def forward(
    self,
    keypoints: torch.Tensor,
    keypointsT: torch.Tensor,
    bbox: torch.Tensor,
  ) -> tuple:
    squared_error = (keypoints - keypointsT).square()
    mse = (self.joint_weights * squared_error).mean()

    detail = F.smooth_l1_loss(
      keypoints,
      keypointsT,
      reduction="none",
      beta=self.detail_beta,
    )
    detail = (self.joint_weights * detail).mean()
    loss = mse + self.detail_weight * detail

    if self.curriculum_scale:
      bbox = bbox.to(device=keypoints.device, dtype=keypoints.dtype)
      bbox_diagonal = torch.linalg.vector_norm(
        bbox[:, 0] - bbox[:, 1],
        dim=1,
      ).clamp_min(1e-6)
      distance = torch.linalg.vector_norm(
        keypoints - keypointsT,
        dim=-1,
      ) / bbox_diagonal[:, None]
      pck05 = self._soft_failure(distance, 0.05)
      pck10 = self._soft_failure(distance, 0.10)
      loss = loss + self.curriculum_scale * (
        self.pck05_weight * pck05
        + self.pck10_weight * pck10
      )

    return loss, loss.item()


class EvtformerSix(EvtformerFive):
  """A delayed and lower-weight PCK curriculum for stable fine tuning."""

  def __init__(self, device: torch.device):
    super().__init__(device)
    self.pck05_weight = 0.04
    self.pck10_weight = 0.01
    self.temperature = 0.02

  def set_epoch(self, epoch: int):
    if epoch < 14:
      self.curriculum_scale = 0.0
    elif epoch == 14:
      self.curriculum_scale = 0.25
    elif epoch == 15:
      self.curriculum_scale = 0.50
    elif epoch == 16:
      self.curriculum_scale = 0.75
    else:
      self.curriculum_scale = 1.0


class Wisppn(nn.Module):
  def __init__(self, device: torch.device):
    super().__init__()
    self.device = device

  def forward(self, ppam: torch.Tensor, ppamT: torch.Tensor) -> tuple:
    cPpamT = ppamT[:, [0]]
    xyPpamT = ppamT[:, 1:]

    loss1 = cPpamT * F.mse_loss(ppam[:, 0], xyPpamT[:, 0], reduction="none")
    loss2 = F.mse_loss(ppam[:, 1], xyPpamT[:, 1], reduction="none")

    loss = (loss1 + loss2).mean()
    lossValue = loss.item()
    return loss, lossValue
