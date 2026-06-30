from libs import poseutils
import config

from sklearn import decomposition
from torch.utils.data import Dataset

import numpy as np

from pathlib import Path
import pickle
import json

dtype = np.float32


class Infer(Dataset):
  def __init__(
    self,
    datasetDir: Path,
    nTimestep: int,
    nSubcarrier: int,
    csiSubDir: str,
    fusion: str,
  ):
    self.nSubcarriers = nSubcarrier
    self.nTimestep = nTimestep
    self.fusion = fusion

    self._csiPathList: list[Path] = []

    for subdir in sorted(datasetDir.iterdir()):
      if subdir.is_dir() and ";" in subdir.name:
        csiDir = subdir / csiSubDir
        subCsiPathList = sorted(csiDir.iterdir())
        self._csiPathList += subCsiPathList

  def __len__(self):
    return len(self._csiPathList)

  def __getitem__(self, idx) -> np.ndarray:
    csiNp = self._getCsi(idx)
    return csiNp

  def _getCsi(self, idx: int) -> np.ndarray:
    csiNp = np.load(self._csiPathList[idx])
    nFrame = csiNp.shape[0]
    low = (nFrame - self.nTimestep) // 2
    hig = low + self.nTimestep

    csiNp = csiNp[low:hig, 0 : self.nSubcarriers, :, :]
    if self.nTimestep > 1:
      if self.fusion == "mean":
        fusedCsi = csiNp.mean(0, keepdims=True)
      elif self.fusion == "max":
        fusedCsi = csiNp.max(0, keepdims=True)
      elif self.fusion == "min":
        fusedCsi = csiNp.min(0, keepdims=True)
      elif self.fusion == "pca":
        shape = csiNp.shape
        tmp = csiNp.reshape(shape[0], -1).transpose()
        y = decomposition.PCA(n_components=1).fit(tmp).transform(tmp)
        fusedCsi = y.transpose().reshape((1,) + shape[1:])
      elif self.fusion == "ica":
        shape = csiNp.shape
        tmp = csiNp.reshape(shape[0], -1).transpose()
        y = decomposition.FastICA(n_components=1).fit(tmp).transform(tmp)
        fusedCsi = y.transpose().reshape((1,) + shape[1:])
      elif self.fusion == "random":
        fusedCsi = csiNp[np.random.randint(0, nFrame)]
      else:
        fusedCsi = csiNp
      out = fusedCsi
    else:
      out = csiNp
    return out.astype(dtype)

  def getCsiPath(self, idx) -> Path:
    return self._csiPathList[idx]


class Train(Infer):
  def __init__(
    self,
    datasetDir: Path,
    nTimestep: int,
    nSubcarrier: int,
    csiSubDir: str,
    fusion: str,
  ):
    self.nSubcarriers = nSubcarrier
    self.nTimestep = nTimestep
    self.fusion = fusion

    self.outRes = config.Global.outRes
    self.originRes = config.Global.originRes

    self.scaleFactor: tuple[float, float] = (
      self.outRes[0] / self.originRes[0],
      self.outRes[1] / self.originRes[1],
    )

    self._csiPathList: list[Path] = []

    for subdir in sorted(datasetDir.iterdir()):
      if subdir.is_dir() and ";" in subdir.name:
        csiDir = subdir / csiSubDir
        subCsiPathList = sorted(csiDir.iterdir())
        self._csiPathList += subCsiPathList

        # Check length
        mmposeDir = subdir / config.Annot.mmposeDir
        detectron2Dir = subdir / config.Annot.detectron2Dir
        # pafsDir = subdir / config.Annot.pafsDir

        csiLen = len(subCsiPathList)
        mmposeLen = sum(1 for x in mmposeDir.iterdir())
        detectron2Len = sum(1 for x in detectron2Dir.iterdir())
        # subPafsLen = sum(1 for x in pafsDir.iterdir())

        assert (
          csiLen == mmposeLen == detectron2Len
        ), f"{csiLen}, {mmposeLen}, {detectron2Len}"

        # assert (
        #   csiLen == mmposeLen == detectron2Len == subPafsLen
        # ), f"{csiLen}, {mmposeLen}, {detectron2Len}, {subPafsLen}"

  def __getitem__(
    self, idx
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    csi = super()._getCsi(idx)
    bbox, mask = self._getDetectron2(idx)
    keypoints = self._getKeypointsFromMmpose(idx)
    jhms = poseutils.keypoints2Jhms(
      keypoints,
      config.Global.outRes,
      config.Global.scaleFactor,
      config.Train.mmposeThr,
      config.Train.gaussianCov,
    )

    factor = np.array(((1,) + config.Global.scaleFactor))
    scaledKeypoints = keypoints * factor[np.newaxis, :]
    return (
      csi.astype(dtype),
      jhms.astype(dtype),
      mask.astype(dtype),
      bbox,
      scaledKeypoints,
    )

  def _getMmposePath(self, idx):
    csiPath = self._csiPathList[idx]
    dir = csiPath.parent.parent / config.Annot.mmposeDir
    filename = csiPath.with_suffix(".json").name
    return dir / filename

  def _getJhmsPath(self, idx):
    csiPath = self._csiPathList[idx]
    dir = csiPath.parent.parent / config.Annot.jhmsDir
    filename = csiPath.with_suffix(".npy").name
    return dir / filename

  def _getDetectron2Path(self, idx):
    csiPath = self._csiPathList[idx]
    dir = csiPath.parent.parent / config.Annot.detectron2Dir
    filename = csiPath.with_suffix(".pkl").name
    return dir / filename

  def _getPafsPath(self, idx):
    csiPath = self._csiPathList[idx]
    dir = csiPath.parent.parent / config.Annot.pafsDir
    filename = csiPath.with_suffix(".npy").name
    return dir / filename

  def _getKeypointsFromMmpose(self, idx: int) -> np.ndarray:
    with open(self._getMmposePath(idx), "rt") as f:
      rawMmpose = json.load(f)
    return poseutils.mmpose2np(rawMmpose)

  def _getDetectron2(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
    path = self._getDetectron2Path(idx)
    with open(path, "rb") as f:
      detectron2: dict[str, np.ndarray] = pickle.load(f)
    bbox = detectron2["zoomedBbox"]
    mask = detectron2["zoomedMask"]
    return (bbox, mask)


class TrainPiw(Train):
  def __init__(
    self,
    datasetDir: Path,
    nTimestep: int,
    nSubcarrier: int,
    csiSubDir: str,
    fusion: str,
  ):
    super().__init__(datasetDir, nTimestep, nSubcarrier, csiSubDir, fusion)

  def __getitem__(
    self, idx
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    csi = super()._getCsi(idx)
    bbox, mask = self._getDetectron2(idx)
    keypoints = self._getKeypointsFromMmpose(idx)
    jhms = poseutils.keypoints2Jhms(
      keypoints,
      config.Global.outRes,
      config.Global.scaleFactor,
      config.Train.mmposeThr,
      config.Train.gaussianCov,
    )
    pafs = np.load(self._getPafsPath(idx))

    factor = np.array(((1,) + config.Global.scaleFactor))
    scaledKeypoints = keypoints * factor[np.newaxis, :]
    return (
      csi.astype(dtype),
      jhms.astype(dtype),
      mask.astype(dtype),
      pafs.astype(dtype),
      bbox,
      scaledKeypoints,
    )


class TrainWpnet(Train):
  def __init__(
    self,
    datasetDir: Path,
    nTimestep: int,
    nSubcarrier: int,
    csiSubDir: str,
    fusion: str,
  ):
    super().__init__(datasetDir, nTimestep, nSubcarrier, csiSubDir, fusion)

  def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    csi = super()._getCsi(idx)

    print(csi.shape)
    bbox, _ = self._getDetectron2(idx)
    keypoints = self._getKeypointsFromMmpose(idx)
    factor = np.array(((1,) + config.Global.scaleFactor))
    scaledKeypoints = keypoints * factor[np.newaxis, :]
    return (
      csi.astype(dtype),
      scaledKeypoints[:, 1:].astype(dtype),
      bbox,
    )


class TrainWpformer(Train):
  def __init__(
    self,
    datasetDir: Path,
    nTimestep: int,
    nSubcarrier: int,
    csiSubDir: str,
    fusion: str,
  ):
    super().__init__(datasetDir, nTimestep, nSubcarrier, csiSubDir, fusion)

  def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    csi = super()._getCsi(idx)
    bbox, _ = self._getDetectron2(idx)
    keypoints = self._getKeypointsFromMmpose(idx)
    factor = np.array(((1,) + config.Global.scaleFactor))
    scaledKeypoints = keypoints * factor[np.newaxis, :]
    return (
      csi.astype(dtype),
      scaledKeypoints[:, 1:].astype(dtype),
      bbox,
    )


class TrainWisppn(Train):
  def __init__(
    self,
    datasetDir: Path,
    nTimestep: int,
    nSubcarrier: int,
    csiSubDir: str,
    fusion: str,
  ):
    super().__init__(datasetDir, nTimestep, nSubcarrier, csiSubDir, fusion)

  def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    csi = super()._getCsi(idx)
    bbox, _ = self._getDetectron2(idx)
    keypoints = self._getKeypointsFromMmpose(idx)
    factor = np.array(((1,) + config.Global.scaleFactor))
    scaledKeypoints = keypoints * factor[np.newaxis, :]
    added = (scaledKeypoints[5] + scaledKeypoints[6]) / 2
    scaledKeypoints = np.append(scaledKeypoints, added[np.newaxis, :], axis=0)
    ppam = self._keypoints2ppam(scaledKeypoints)

    return (
      csi.astype(dtype),
      ppam.astype(dtype),
      bbox,
    )

  def _keypoints2ppam(self, keypoints: np.ndarray) -> np.ndarray:
    cPpam = np.diag(keypoints[:, 0])
    xPpam = np.diag(keypoints[:, 1])
    yPpam = np.diag(keypoints[:, 2])
    return np.stack((cPpam, xPpam, yPpam))
