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
