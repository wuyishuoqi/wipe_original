from libs import utils, figutils, poseutils, torchutils
import config
import dataset as MyDateset
import loss as loss_
import model.my

from torch.utils.data import DataLoader, Subset, random_split
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np

from pathlib import Path
import os
import pickle

nSubThreads = config.Train.nSubThreads
torch.backends.cudnn.benchmark = True
plt.switch_backend("Agg")


class Trainer:
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  def __init__(self, conf: config.Train):
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.Train.gpus

    # Init model
    self.conf = conf
    nInChannel = conf.nSubcarrier * 1  # conf.nTimestep
    nOutChannel = config.Global.nJoints + 1

    if conf.model == "Piw":
      self.model = self._getModel(conf.model, nInChannel, nOutChannel, 38)
    else:
      self.model = self._getModel(conf.model, nInChannel, nOutChannel)

    self.model = torch.nn.DataParallel(self.model).to(self.device)
    self.optimizer = torch.optim.Adam(params=self.model.parameters())

    # Init train stats save
    self.epochIdxList: list[int] = []  # Iter
    self.trainLossList: list = []  # Iter, Loss
    self.valIdxList: list[int] = []  # Iter

    self.valBatchLossList: list = []  # Idx, Batch, Loss
    self.valBatchPckList: dict[float, list] = {
      k: [] for k in conf.pckThrList
    }  # Thr: (Idx, Batch, Pck)
    self.valBatchIouList: list = []  # Idx, Batch

  def _getModel(self, name: str, *args) -> nn.Module:
    return getattr(model.my, name)(*args)

  def _calIou(
    self, mask: np.ndarray, maskT: np.ndarray, thr: float = 0.5
  ) -> np.ndarray:
    maskResult = mask > thr
    i = maskResult & maskT
    u = maskResult | maskT
    iou = i.sum() / u.sum()
    return iou

  def _saveTrain(self, outDir: Path):
    # Save
    outDir.mkdir(parents=True, exist_ok=True)
    torch.save(self.model.state_dict(), outDir / "weights.pt")

    # Save loss and metric figure
    trainLossNp = np.array(self.trainLossList)  # N, Type
    valLossNp = np.array(self.valBatchLossList)  # N, Batch, Type
    valPckNp = {
      k: np.array(v) for k, v in self.valBatchPckList.items()
    }  # Thr: (N, Batch, Keypoint)

    np.save(outDir / "train.loss.npy", trainLossNp)
    np.save(outDir / "val.loss.npy", valLossNp)
    with open(outDir / "val.pck.pkl", "wb") as f:
      pickle.dump(valPckNp, f)

    pckThr = 0.2
    x = range(len(self.trainLossList))
    figutils.lossWithMetric(
      Path(outDir / "mpck.loss.pdf"),
      "Iteration",
      (x, trainLossNp[:, 0]),
      (self.valIdxList, valLossNp[:, :, 0].mean(axis=1)),
      f"mPCK@{pckThr}",
      (self.valIdxList, valPckNp[pckThr][:, :, 0].mean(axis=1)),
    )
    figutils.lossWithMetric(
      Path(outDir / "mpck.jhms-loss.pdf"),
      "Iteration",
      (x, trainLossNp[:, 1]),
      (self.valIdxList, valLossNp[:, :, 1].mean(axis=1)),
      f"mPCK@{pckThr}",
      (self.valIdxList, valPckNp[pckThr][:, :, 0].mean(axis=1)),
    )

    # IoU
    iou = True if self.valBatchIouList else False
    if iou:
      valIouNp = np.array(self.valBatchIouList)  # N, Batch
      np.save(outDir / "val.iou.npy", valIouNp)
      figutils.lossWithMetric(
        Path(outDir / "iou.loss.pdf"),
        "Iteration",
        (x, trainLossNp[:, 0]),
        (self.valIdxList, valLossNp[:, :, 0].mean(axis=1)),
        "IoU",
        (self.valIdxList, valIouNp.mean(axis=1)),
      )
      figutils.lossWithMetric(
        Path(outDir / "iou.mask-loss.pdf"),
        "Iteration",
        (x, trainLossNp[:, 2]),
        (self.valIdxList, valLossNp[:, :, 2].mean(axis=1)),
        "IoU",
        (self.valIdxList, valIouNp.mean(axis=1)),
      )

  def inference(
    self,
    valIndex: bool = False,
  ):
    outDir = Path(config.Global.outDir) / self.conf.outDir
    outDir.mkdir(parents=True, exist_ok=True)

    weightsPath = Path(self.conf.weightsFile)

    dataset = MyDateset.Infer(
      Path(self.conf.datasetDir),
      self.conf.nTimestep,
      self.conf.nSubcarrier,
      self.conf.csiSubDir,
      self.conf.fusion,
    )

    if valIndex and self.conf.idxFile:
      valIndexList = np.load(self.conf.idxFile)
      subset = Subset(dataset, valIndexList)
    else:
      subset = dataset
      valIndexList = np.arange(len(dataset))
    dataLoader = DataLoader(
      dataset=subset,
      batch_size=1,
      shuffle=False,
      num_workers=nSubThreads,
    )

    self.model.load_state_dict(torch.load(weightsPath))
    self.model.eval()
    with torch.no_grad():
      for idx, data in enumerate(tqdm(dataLoader, leave=False)):
        x = data
        x = x.to(self.device)

        y = self.model(x)
        mask = torchutils.toNp(y[:, 0, :, :])
        jhms = torchutils.toNp(y[:, 1:, :, :])

        # Save
        outStem = f"{valIndexList[idx]}".zfill(8)
        outJhmsPath = outDir / f"{outStem}.jhms.npy"
        outMaskPath = outDir / f"{outStem}.mask.npy"
        np.save(outJhmsPath, jhms.reshape(jhms.shape[1:]))
        np.save(outMaskPath, mask.reshape(mask.shape[1:]))

  def inference_ret(self, input_data, valIndex: bool = False):
    # 定义输出目录
    outDir = Path(config.Global.outDir) / self.conf.outDir
    outDir.mkdir(parents=True, exist_ok=True)

    # 定义权重文件路径
    weightsPath = Path(self.conf.weightsFile)

    # 加载模型权重
    self.model.load_state_dict(torch.load(weightsPath))
    self.model.eval()
    input_data=input_data.mean(0, keepdims=True)
    # 将输入数据转换为张量并移动到设备
    input_tensor = torch.tensor(input_data).to(self.device)

    with torch.no_grad():
        # 进行模型推理
        y = self.model(input_tensor)
        
        # 提取结果
        jhms = torchutils.toNp(y[:, 1:, :, :])
        
        # 保存推理结果
        outJhmsPath = outDir / "output.jhms.npy"
        np.save(outJhmsPath, jhms.reshape(jhms.shape[1:]))
        
        return jhms.reshape(jhms.shape[1:])


  def _initTrain(self):
    self.outDir = Path(config.Global.outDir) / self.conf.outDir
    self.outDir.mkdir(parents=True, exist_ok=True)

    datasetPath = Path(self.conf.datasetDir)
    if self.conf.model == "Piw":
      dataset = MyDateset.TrainPiw(
        datasetPath,
        self.conf.nTimestep,
        self.conf.nSubcarrier,
        self.conf.csiSubDir,
        self.conf.fusion,
      )
    elif self.conf.model == "Wpnet":
      dataset = MyDateset.TrainWpnet(
        datasetPath,
        self.conf.nTimestep,
        self.conf.nSubcarrier,
        self.conf.csiSubDir,
        self.conf.fusion,
      )
    elif self.conf.model == "Wisppn":
      dataset = MyDateset.TrainWisppn(
        datasetPath,
        self.conf.nTimestep,
        self.conf.nSubcarrier,
        self.conf.csiSubDir,
        self.conf.fusion,
      )
    elif self.conf.model in ("Wpformer", "Evtformer", "EvtformerV4", "EvtformerV7"):
      dataset = MyDateset.TrainWpformer(
        datasetPath,
        self.conf.nTimestep,
        self.conf.nSubcarrier,
        self.conf.csiSubDir,
        self.conf.fusion,
      )
    else:
      dataset = MyDateset.Train(
        datasetPath,
        self.conf.nTimestep,
        self.conf.nSubcarrier,
        self.conf.csiSubDir,
        self.conf.fusion,
      )
    trainSize = int(self.conf.trainRatio * len(dataset))
    valSize = len(dataset) - trainSize
    trainSet, valSet = random_split(
      dataset, [trainSize, valSize], torch.Generator().manual_seed(3)
    )

    # Save indices
    np.save(self.outDir / "trainset.idx.npy", trainSet.indices)
    np.save(self.outDir / "valset.idx.npy", valSet.indices)

    self.trainLoader = DataLoader(
      dataset=trainSet,
      batch_size=self.conf.batchSize,
      shuffle=True,
      pin_memory=True,
      num_workers=nSubThreads,
      persistent_workers=True,
    )
    self.valLoader = DataLoader(
      dataset=valSet,
      batch_size=self.conf.batchSize,
      shuffle=False,
      pin_memory=True,
      num_workers=nSubThreads,
      persistent_workers=True,
    )

    self.nBatch = len(self.trainLoader)
    self.valBatchIdx = frozenset(
      int((self.nBatch - 1) * x) for x in self.conf.valEpochMilestone
    )

  def train(self):
    self._initTrain()

    self.criterion = loss_.MyLoss(
      self.device,
      self.conf.mWeightJhmsParam,
      self.conf.maskWeight,
      self.conf.jhmsRegularNrom,
      self.conf.jhmsRegularWeight,
      self.conf.maskRegularNrom,
      self.conf.maskRegularWeight,
    )

    self.model.train()
    iter = 0
    for epoch in trange(self.conf.nEpoch, desc="Epoch", leave=False):
      self.epochIdxList.append(iter)
      np.save(self.outDir / "epoch.idx.npy", self.epochIdxList)

      # Train
      with tqdm(total=self.nBatch, desc="Train", leave=False) as pbar:
        for i, batch in enumerate(self.trainLoader):
          x, jhmsT, maskT, _, _ = batch
          x, jhmsT, maskT = (
            x.to(self.device),
            jhmsT.to(self.device),
            maskT.to(self.device),
          )

          y = self.model(x)
          maskY = y[:, 0, :, :]
          jhmsY = y[:, 1:, :, :]
          loss, lossVal = self.criterion(jhmsY, jhmsT, maskY, maskT)

          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()

          self.trainLossList.append(lossVal)
          pbar.set_description(f"Train l = {lossVal[0]:.2e}")
          pbar.update(1)

          # Validate
          if i in self.valBatchIdx or iter == 0:
            valBatchLoss, valBatchPck, valBatchIou = self._eval(
              self.valLoader, self.conf.pckThrList
            )
            self.model.train()

            self.valIdxList.append(iter)
            self.valBatchLossList.append(valBatchLoss)
            utils.dictAppend(self.valBatchPckList, valBatchPck)
            self.valBatchIouList.append(valBatchIou)

            self._saveTrain(self.outDir / f"epoch-{epoch}")

          iter += 1

  def _eval(
    self,
    dataLoader: DataLoader,
    pckThrList: tuple,
  ) -> tuple[np.ndarray, dict[float, np.ndarray], np.ndarray]:
    batchLossList: list = []
    batchPckList: dict[float, list] = {k: [] for k in pckThrList}
    batchIouList: list = []

    nBatch = len(dataLoader)
    self.model.eval()
    with tqdm(total=nBatch, desc="Eval", leave=False) as pbar:
      with torch.no_grad():
        for i, batch in enumerate(dataLoader):
          x, jhmsT, maskT, bbox, keypointsT = batch
          x, jhmsT, maskT = (
            x.to(self.device),
            jhmsT.to(self.device),
            maskT.to(self.device),
          )

          y = self.model(x)
          maskY = y[:, 0, :, :]
          jhmsY = y[:, 1:, :, :]
          _, lossVal = self.criterion(jhmsY, jhmsT, maskY, maskT)
          batchLossList.append(lossVal)

          batchSize = bbox.shape[0]
          norm = np.linalg.norm(bbox[:, 0] - bbox[:, 1], axis=1)
          norm = np.ones((batchSize, config.Global.nJoints)) * norm[:, np.newaxis]
          norm[norm == 0] = -1

          keypoints = poseutils.batchedJhms2xys(
            torchutils.toNp(jhmsY), self.conf.decode
          )
          pck = poseutils.batchedXyAccuracy(
            keypoints[:, :, 1:],
            keypointsT.numpy()[:, :, 1:],
            pckThrList,
            norm,
          )
          utils.dictAppend(batchPckList, pck)

          iou = self._calIou(
            torchutils.toNp(maskY), torchutils.toNp(maskT).astype(bool)
          )
          batchIouList.append(iou)

          pbar.set_description(f"Val l = {lossVal[0]:.2e}")
          pbar.update(1)
          i += 1

    return (
      np.array(batchLossList),
      {k: np.array(v) for k, v in batchPckList.items()},
      np.array(batchIouList),
    )


class TrainerPiw(Trainer):
  def train(self):
    self._initTrain()

    self.criterion = loss_.Piw(self.device)

    self.model.train()
    iter = 0
    for epoch in trange(self.conf.nEpoch, desc="Epoch", leave=False):
      self.epochIdxList.append(iter)
      np.save(self.outDir / "epoch.idx.npy", self.epochIdxList)

      # Train
      with tqdm(total=self.nBatch, desc="Train", leave=False) as pbar:
        for i, batch in enumerate(self.trainLoader):
          x, jhmsT, maskT, pafsT, _, _ = batch
          x, jhmsT, maskT, pafsT = (
            x.to(self.device),
            jhmsT.to(self.device),
            maskT.to(self.device),
            pafsT.to(self.device),
          )

          y, pafsY = self.model(x)
          maskY = y[:, 0, :, :]
          jhmsY = y[:, 1:, :, :]
          loss, lossVal = self.criterion(jhmsY, jhmsT, maskY, maskT, pafsY, pafsT)

          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()

          self.trainLossList.append(lossVal)
          pbar.set_description(f"Train l = {lossVal[0]:.2e}")
          pbar.update(1)

          # Validate
          if i in self.valBatchIdx or iter == 0:
            valBatchLoss, valBatchPck, valBatchIou = self._eval(
              self.valLoader, self.conf.pckThrList
            )
            self.model.train()

            self.valIdxList.append(iter)
            self.valBatchLossList.append(valBatchLoss)
            utils.dictAppend(self.valBatchPckList, valBatchPck)
            self.valBatchIouList.append(valBatchIou)

            self._saveTrain(self.outDir / f"epoch-{epoch}")

          iter += 1

  def _eval(
    self,
    dataLoader: DataLoader,
    pckThrList: tuple,
  ) -> tuple[np.ndarray, dict[float, np.ndarray], np.ndarray]:
    batchLossList: list = []
    batchPckList: dict[float, list] = {k: [] for k in pckThrList}
    batchIouList: list = []

    nBatch = len(dataLoader)
    self.model.eval()
    with tqdm(total=nBatch, desc="Eval", leave=False) as pbar:
      with torch.no_grad():
        for i, batch in enumerate(dataLoader):
          x, jhmsT, maskT, pafsT, bbox, keypointsT = batch
          x, jhmsT, maskT, pafsT = (
            x.to(self.device),
            jhmsT.to(self.device),
            maskT.to(self.device),
            pafsT.to(self.device),
          )

          y, pafsY = self.model(x)
          maskY = y[:, 0, :, :]
          jhmsY = y[:, 1:, :, :]
          _, lossVal = self.criterion(jhmsY, jhmsT, maskY, maskT, pafsY, pafsT)
          batchLossList.append(lossVal)

          batchSize = bbox.shape[0]
          norm = np.linalg.norm(bbox[:, 0] - bbox[:, 1], axis=1)
          norm = np.ones((batchSize, config.Global.nJoints)) * norm[:, np.newaxis]
          norm[norm == 0] = -1

          keypoints = poseutils.batchedJhms2xys(
            torchutils.toNp(jhmsY), self.conf.decode
          )
          pck = poseutils.batchedXyAccuracy(
            keypoints,
            keypointsT.numpy(),
            pckThrList,
            norm,
          )
          utils.dictAppend(batchPckList, pck)

          iou = self._calIou(
            torchutils.toNp(maskY), torchutils.toNp(maskT).astype(bool)
          )
          batchIouList.append(iou)

          pbar.set_description(f"Val l = {lossVal:.2e}")
          pbar.update(1)
          i += 1

    return (
      np.array(batchLossList),
      {k: np.array(v) for k, v in batchPckList.items()},
      np.array(batchIouList),
    )


class TrainerWpnet(Trainer):
  def _saveTrain(self, outDir: Path):
    # Save
    outDir.mkdir(parents=True, exist_ok=True)
    torch.save(self.model.state_dict(), outDir / "weights.pt")

    # Save loss and metric figure
    trainLossNp = np.array(self.trainLossList)  # N, Type
    valLossNp = np.array(self.valBatchLossList)  # N, Batch, Type
    valPckNp = {
      k: np.array(v) for k, v in self.valBatchPckList.items()
    }  # Thr: (N, Batch, Keypoint)

    np.save(outDir / "train.loss.npy", trainLossNp)
    np.save(outDir / "val.loss.npy", valLossNp)
    with open(outDir / "val.pck.pkl", "wb") as f:
      pickle.dump(valPckNp, f)

    pckThr = 0.2
    x = range(len(self.trainLossList))
    figutils.lossWithMetric(
      Path(outDir / "mpck.loss.pdf"),
      "Iteration",
      (x, trainLossNp),
      (self.valIdxList, valLossNp.mean(axis=1)),
      f"mPCK@{pckThr}",
      (self.valIdxList, valPckNp[pckThr][:, :, 0].mean(axis=1)),
    )

  def train(self):
    self._initTrain()

    self.criterion = loss_.Wpnet(self.device)

    self.model.train()
    iter = 0
    for epoch in trange(self.conf.nEpoch, desc="Epoch", leave=False):
      self.epochIdxList.append(iter)
      np.save(self.outDir / "epoch.idx.npy", self.epochIdxList)

      # Train
      with tqdm(total=self.nBatch, desc="Train", leave=False) as pbar:
        for i, batch in enumerate(self.trainLoader):
          x, keypointsT, _ = batch
          x, keypointsT = (
            x.to(self.device),
            keypointsT.to(self.device),
          )

          y = self.model(x)
          loss, lossVal = self.criterion(y, keypointsT)

          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()

          self.trainLossList.append(lossVal)
          pbar.set_description(f"Train l = {lossVal:.2e}")
          pbar.update(1)

          # Validate
          if i in self.valBatchIdx or iter == 0:
            valBatchLoss, valBatchPck = self._eval(self.valLoader, self.conf.pckThrList)
            self.model.train()

            self.valIdxList.append(iter)
            self.valBatchLossList.append(valBatchLoss)
            utils.dictAppend(self.valBatchPckList, valBatchPck)

            suffix = str(epoch).zfill(0)
            self._saveTrain(self.outDir / f"epoch-{suffix}")

          iter += 1

  def _eval(
    self,
    dataLoader: DataLoader,
    pckThrList: tuple,
  ) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    batchLossList: list = []
    batchPckList: dict[float, list] = {k: [] for k in pckThrList}

    nBatch = len(dataLoader)
    self.model.eval()
    with tqdm(total=nBatch, desc="Eval", leave=False) as pbar:
      with torch.no_grad():
        for i, batch in enumerate(dataLoader):
          x, keypointsT, bbox = batch
          x, keypointsT = (
            x.to(self.device),
            keypointsT.to(self.device),
          )

          y = self.model(x)
          _, lossVal = self.criterion(y, keypointsT)
          batchLossList.append(lossVal)

          batchSize = bbox.shape[0]
          norm = np.linalg.norm(bbox[:, 0] - bbox[:, 1], axis=1)
          norm = np.ones((batchSize, config.Global.nJoints)) * norm[:, np.newaxis]
          norm[norm == 0] = -1

          pck = poseutils.batchedXyAccuracy(
            torchutils.toNp(y),
            torchutils.toNp(keypointsT),
            pckThrList,
            norm,
          )
          utils.dictAppend(batchPckList, pck)

          pbar.set_description(f"Val l = {lossVal:.2e}")
          pbar.update(1)
          i += 1

    return (
      np.array(batchLossList),
      {k: np.array(v) for k, v in batchPckList.items()},
    )


class TrainerWpformer(Trainer):
  def _saveTrain(self, outDir: Path):
    # Save
    outDir.mkdir(parents=True, exist_ok=True)
    torch.save(self.model.state_dict(), outDir / "weights.pt")

    # Save loss and metric figure
    trainLossNp = np.array(self.trainLossList)  # N, Type
    valLossNp = np.array(self.valBatchLossList)  # N, Batch, Type
    valPckNp = {
      k: np.array(v) for k, v in self.valBatchPckList.items()
    }  # Thr: (N, Batch, Keypoint)

    np.save(outDir / "train.loss.npy", trainLossNp)
    np.save(outDir / "val.loss.npy", valLossNp)
    with open(outDir / "val.pck.pkl", "wb") as f:
      pickle.dump(valPckNp, f)

    pckThr = 0.2
    x = range(len(self.trainLossList))
    figutils.lossWithMetric(
      Path(outDir / "mpck.loss.pdf"),
      "Iteration",
      (x, trainLossNp),
      (self.valIdxList, valLossNp.mean(axis=1)),
      f"mPCK@{pckThr}",
      (self.valIdxList, valPckNp[pckThr][:, :, 0].mean(axis=1)),
    )

  def train(self):
    self._initTrain()

    self.criterion = loss_.Wpformer(self.device)

    # AMP disabled due to NaN on this dataset
    use_amp = False

    self.model.train()
    iter = 0
    for epoch in trange(self.conf.nEpoch, desc="Epoch", leave=True, position=0):
      self.epochIdxList.append(iter)
      np.save(self.outDir / "epoch.idx.npy", self.epochIdxList)

      # Train
      with tqdm(total=self.nBatch, desc="Train", leave=False, position=1) as pbar:
        for i, batch in enumerate(self.trainLoader):
          x, keypointsT, _ = batch
          x, keypointsT = (
            x.to(self.device),
            keypointsT.to(self.device),
          )

          self.optimizer.zero_grad()
          y = self.model(x)
          loss, lossVal = self.criterion(y, keypointsT)

          loss.backward()
          torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
          self.optimizer.step()

          self.trainLossList.append(lossVal)
          pbar.set_description(f"Train l = {lossVal:.2e}")
          pbar.update(1)

          # Validate
          if i in self.valBatchIdx or iter == 0:
            valBatchLoss, valBatchPck = self._eval(self.valLoader, self.conf.pckThrList)
            self.model.train()

            self.valIdxList.append(iter)
            self.valBatchLossList.append(valBatchLoss)
            utils.dictAppend(self.valBatchPckList, valBatchPck)

            suffix = str(epoch).zfill(0)
            self._saveTrain(self.outDir / f"epoch-{suffix}")

          iter += 1

  def _eval(
    self,
    dataLoader: DataLoader,
    pckThrList: tuple,
  ) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    batchLossList: list = []
    batchPckList: dict[float, list] = {k: [] for k in pckThrList}

    nBatch = len(dataLoader)
    self.model.eval()
    with tqdm(total=nBatch, desc="Eval", leave=False) as pbar:
      with torch.no_grad():
        for i, batch in enumerate(dataLoader):
          x, keypointsT, bbox = batch
          x, keypointsT = (
            x.to(self.device),
            keypointsT.to(self.device),
          )

          y = self.model(x)
          _, lossVal = self.criterion(y, keypointsT)
          batchLossList.append(lossVal)

          batchSize = bbox.shape[0]
          norm = np.linalg.norm(bbox[:, 0] - bbox[:, 1], axis=1)
          norm = np.ones((batchSize, config.Global.nJoints)) * norm[:, np.newaxis]
          norm[norm == 0] = -1

          pck = poseutils.batchedXyAccuracy(
            torchutils.toNp(y),
            torchutils.toNp(keypointsT),
            pckThrList,
            norm,
          )
          utils.dictAppend(batchPckList, pck)

          pbar.set_description(f"Val l = {lossVal:.2e}")
          pbar.update(1)
          i += 1

    return (
      np.array(batchLossList),
      {k: np.array(v) for k, v in batchPckList.items()},
    )

  def inference(self, valIndex: bool = False):
    outDir = Path(config.Global.outDir) / self.conf.outDir
    outDir.mkdir(parents=True, exist_ok=True)

    weightsPath = Path(self.conf.weightsFile)

    dataset = MyDateset.Infer(
      Path(self.conf.datasetDir),
      self.conf.nTimestep,
      self.conf.nSubcarrier,
      self.conf.csiSubDir,
      self.conf.fusion,
    )

    if valIndex and self.conf.idxFile:
      valIndexList = np.load(self.conf.idxFile)
      subset = Subset(dataset, valIndexList)
    else:
      subset = dataset
      valIndexList = np.arange(len(dataset))
    dataLoader = DataLoader(
      dataset=subset,
      batch_size=1,
      shuffle=False,
      num_workers=nSubThreads,
    )

    self.model.load_state_dict(torch.load(weightsPath))
    self.model.eval()
    with torch.no_grad():
      for idx, data in enumerate(tqdm(dataLoader, leave=False)):
        x = data
        x = x.to(self.device)

        y = self.model(x)  # [1, 17, 2] 关键点坐标
        keypoints = torchutils.toNp(y).reshape(-1, 2)  # [17, 2]

        # Save
        outStem = f"{valIndexList[idx]}".zfill(8)
        outKeypointsPath = outDir / f"{outStem}.keypoints.npy"
        np.save(outKeypointsPath, keypoints)


class TrainerWisppn(Trainer):
  def _saveTrain(self, outDir: Path):
    # Save
    outDir.mkdir(parents=True, exist_ok=True)
    torch.save(self.model.state_dict(), outDir / "weights.pt")

    # Save loss and metric figure
    trainLossNp = np.array(self.trainLossList)  # N
    valLossNp = np.array(self.valBatchLossList)  # N, Batch
    valPckNp = {
      k: np.array(v) for k, v in self.valBatchPckList.items()
    }  # Thr: (N, Batch, Keypoint)

    np.save(outDir / "train.loss.npy", trainLossNp)
    np.save(outDir / "val.loss.npy", valLossNp)
    with open(outDir / "val.pck.pkl", "wb") as f:
      pickle.dump(valPckNp, f)

    pckThr = 0.2
    x = range(len(self.trainLossList))
    figutils.lossWithMetric(
      Path(outDir / "mpck.loss.pdf"),
      "Iteration",
      (x, trainLossNp),
      (self.valIdxList, valLossNp.mean(axis=1)),
      f"mPCK@{pckThr}",
      (self.valIdxList, valPckNp[pckThr][:, :, 0].mean(axis=1)),
    )

  def train(self):
    self._initTrain()

    self.criterion = loss_.Wisppn(self.device)

    use_amp = False  # disabled due to NaN on this dataset

    self.model.train()
    iter = 0
    for epoch in trange(self.conf.nEpoch, desc="Epoch", leave=True, position=0):
      self.epochIdxList.append(iter)
      np.save(self.outDir / "epoch.idx.npy", self.epochIdxList)

      # Train
      with tqdm(total=self.nBatch, desc="Train", leave=False, position=1) as pbar:
        for i, batch in enumerate(self.trainLoader):
          x, ppamT, _ = batch
          x, ppamT = (
            x.to(self.device),
            ppamT.to(self.device),
          )

          self.optimizer.zero_grad()
          y = self.model(x)
          loss, lossVal = self.criterion(y, ppamT)

          loss.backward()
          torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
          self.optimizer.step()

          self.trainLossList.append(lossVal)
          pbar.set_description(f"Train l = {lossVal:.2e}")
          pbar.update(1)

          # Validate
          if i in self.valBatchIdx or iter == 0:
            valBatchLoss, valBatchPck = self._eval(self.valLoader, self.conf.pckThrList)
            self.model.train()

            self.valIdxList.append(iter)
            self.valBatchLossList.append(valBatchLoss)
            utils.dictAppend(self.valBatchPckList, valBatchPck)

            self._saveTrain(self.outDir / f"epoch-{epoch}")

          iter += 1

  def _eval(
    self,
    dataLoader: DataLoader,
    pckThrList: tuple,
  ) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    batchLossList: list = []
    batchPckList: dict[float, list] = {k: [] for k in pckThrList}

    nBatch = len(dataLoader)
    self.model.eval()
    with tqdm(total=nBatch, desc="Eval", leave=False) as pbar:
      with torch.no_grad():
        for i, batch in enumerate(dataLoader):
          x, ppamT, bbox = batch
          x, ppamT = (
            x.to(self.device),
            ppamT.to(self.device),
          )

          y = self.model(x)
          _, lossVal = self.criterion(y, ppamT)
          batchLossList.append(lossVal)

          keypoints = torch.diagonal(y, dim1=-2, dim2=-1).transpose(-2, -1)
          keypointsT = torch.diagonal(ppamT[:, 1:], dim1=-2, dim2=-1).transpose(-2, -1)

          batchSize = bbox.shape[0]
          norm = np.linalg.norm(bbox[:, 0] - bbox[:, 1], axis=1)
          norm = np.ones((batchSize, config.Global.nJoints + 1)) * norm[:, np.newaxis]
          norm[norm == 0] = -1

          pck = poseutils.batchedXyAccuracy(
            torchutils.toNp(keypoints),
            torchutils.toNp(keypointsT),
            pckThrList,
            norm,
          )
          utils.dictAppend(batchPckList, pck)

          pbar.set_description(f"Val l = {lossVal:.2e}")
          pbar.update(1)
          i += 1

    return (
      np.array(batchLossList),
      {k: np.array(v) for k, v in batchPckList.items()},
    )

  def inference(self, valIndex: bool = False):
    outDir = Path(config.Global.outDir) / self.conf.outDir
    outDir.mkdir(parents=True, exist_ok=True)

    weightsPath = Path(self.conf.weightsFile)

    dataset = MyDateset.Infer(
      Path(self.conf.datasetDir),
      self.conf.nTimestep,
      self.conf.nSubcarrier,
      self.conf.csiSubDir,
      self.conf.fusion,
    )

    if valIndex and self.conf.idxFile:
      valIndexList = np.load(self.conf.idxFile)
      subset = Subset(dataset, valIndexList)
    else:
      subset = dataset
      valIndexList = np.arange(len(dataset))
    dataLoader = DataLoader(
      dataset=subset,
      batch_size=1,
      shuffle=False,
      num_workers=nSubThreads,
    )

    self.model.load_state_dict(torch.load(weightsPath))
    self.model.eval()
    with torch.no_grad():
      for idx, data in enumerate(tqdm(dataLoader, leave=False)):
        x = data
        x = x.to(self.device)

        y = self.model(x)  # [1, 2, 18, 18] PPAM matrix
        ppam = torchutils.toNp(y).reshape(2, 18, 18)

        # Save
        outStem = f"{valIndexList[idx]}".zfill(8)
        outPpamPath = outDir / f"{outStem}.ppam.npy"
        np.save(outPpamPath, ppam)


class TrainerEvtformer(TrainerWpformer):
  """Trainer for Evtformer model (v8, latest).

  Inherits from TrainerWpformer since Evtformer shares the same
  input/output format: CSI -> [N, 17, 2] keypoint coordinates.
  Uses Wpformer dataset and loss (MSE on keypoints).
  """

  pass


class TrainerEvtformerV4(TrainerWpformer):
  """Trainer for EvtformerV4 model (CT + EVT)."""
  pass


class TrainerEvtformerV7(TrainerWpformer):
  """Trainer for EvtformerV7 model (DualToken v1 + SonnetFusion + EVT)."""
  pass
