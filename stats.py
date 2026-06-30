#!/usr/bin/env python3

from libs import utils, map_
import config

from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from dataclasses import dataclass
from pathlib import Path
import pickle
import re

percentFormatter0 = mtick.FuncFormatter(lambda x, _: f"{x * 100:.0f}")
percentFormatter1 = mtick.FuncFormatter(lambda x, _: f"{x * 100:.1f}")
percentFormatter2 = mtick.FuncFormatter(lambda x, _: f"{x * 100:.2f}")


linewidth = config.Fig.lineWidth
lineColormap = plt.get_cmap(config.Fig.lineColormap)

fillAlpha = config.Fig.errorFillAlpha
fillColormap = plt.get_cmap(config.Fig.fillColormap)


def naturalSortKey(s):
  sub = re.search(r"\d+", s).group(0)
  return int(sub)


class Perf:
  def __init__(self, path: Path, pckThrList: tuple) -> None:
    self.path: Path = path

    self.pckThrList: tuple = pckThrList

    self.name: str = ""
    self.trainLossNp: np.ndarray = None  # Model, Iter, Loss
    self.valLossNp: np.ndarray = None  # Model, Epoch, Batch, Loss

    self.valThrPckNp = {}  # Thr: (Model, Epoch, Batch, Pck)
    self.valThrPckMaxEpochList = {}  # Thr: (Model, Epoch)

    self.iou = True
    self.valIouNp: np.ndarray = None  # Model, Epoch, Batch

    self.name = self.path.name
    if self.name in config.Stats.order:
      self.order = config.Stats.order[self.name]
    else:
      self.order = 1_000_000

    # trainLossList: list[np.ndarray] = []  # Model, Iter, Loss
    # valLossList: list[np.ndarray] = []  # Model, Epoch, Batch, Loss

    valThrPckList = {
      thr: [] for thr in self.pckThrList
    }  # Thr: (Model, Epoch, Batch, Pck)
    valIouList: list[np.ndarray] = []  # Model, Epoch, Batch

    for sub in sorted(self.path.iterdir(), key=lambda x: naturalSortKey(x.name)):
      epochDirList = sorted(sub.glob("epoch-*"), key=lambda x: naturalSortKey(x.name))

      # trainLossList.append(
      #   np.load(epochDirList[-1] / "train.loss.npy")
      # )  #  Epoch, Batch, Pck
      # valLossList.append(
      #   np.load(epochDirList[-1] / "val.loss.npy")
      # )  #  Epoch, Batch, Pck

      with open(epochDirList[-1] / "val.pck.pkl", "rb") as f:
        utils.dictAppend(valThrPckList, pickle.load(f))

      if self.iou:
        ioUpath = epochDirList[-1] / "val.iou.npy"
        if not ioUpath.exists():
          self.iou = False
        else:
          valIouList.append(np.load(epochDirList[-1] / "val.iou.npy"))

    self.valThrPckNp = {k: np.array(v) for k, v in valThrPckList.items()}
    for thr in self.pckThrList:
      mpck = self.valThrPckNp[thr][:, :, :, 0].mean(2)  # Model, Epoch
      self.valThrPckMaxEpochList[thr] = mpck.argmax(1)

    if self.iou:
      self.valIouNp = np.array(valIouList)  # Model, Epoch, Batch

  def __str__(self) -> str:
    return self.name


def saveCsv(outPath: Path, perfList: list[Perf], thr):
  dfList = []
  for perf in perfList:
    name = str(perf)

    pcks = perf.valThrPckNp[thr]  # Model, Epoch, Batch, Pck
    pcks = pcks.mean(2)  # Model, Epoch, Pck

    maxEpochList = perf.valThrPckMaxEpochList[thr]
    pcks = pcks[np.arange(maxEpochList.size), maxEpochList].mean(0)

    if perf.iou:
      iou = perf.valIouNp  # Model, Epoch, Batch
      iou = iou.mean(-1).max(-1).mean()
    else:
      iou = -1

    columns = ("IoU", "Mean") + tuple(range(0, pcks.size - 1))
    data = np.append([iou], pcks)

    dfList.append(
      pd.DataFrame(
        index=[name],
        data=data[np.newaxis, :],
        columns=columns,
      )
    )

  pd.concat(dfList).to_csv(outPath)


def plotBox(
  outPath: Path,
  data: tuple | list | np.ndarray,
  labelList: tuple | list,
  yLabel: str,
  title: str = "",
  legend: str = "",
  xTick: bool = True,
  yLimit: tuple = (),
):
  fig, ax = plt.subplots(figsize=(7, 4))

  meanProps = dict(marker="x", markeredgecolor="black")

  bplotList = []
  for i, label in enumerate(labelList):
    kwargs = dict(
      x=data[i],
      labels=[label],
      positions=[i + 1],
      vert=True,
      showmeans=True,
      meanprops=meanProps,
      patch_artist=True,
      widths=0.4,
    )
    bplotList.append(ax.boxplot(**kwargs))

  nBox = len(bplotList)
  cm = plt.get_cmap(config.Fig.boxColormap, nBox)
  for i in range(nBox):
    bplotList[i]["boxes"][0].set_facecolor(cm(i))
    bplotList[i]["medians"][0].set_color("red")

  if legend:
    legendList = [bplot["boxes"][0] for bplot in bplotList]
    ax.legend(legendList, labelList, loc=legend)

  if not xTick:
    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

  if title:
    ax.set_title(title, fontsize=config.Fig.titleSize)

  if yLimit:
    ax.set_ylim(yLimit)

  ax.yaxis.set_major_formatter(percentFormatter1)
  ax.set_ylabel(yLabel, fontsize=config.Fig.axisLabelSize)
  ax.tick_params(axis="x", labelsize=config.Fig.tickSize)
  ax.tick_params(axis="y", labelsize=config.Fig.tickSize)

  ax.yaxis.grid(True)

  ax.set_xlabel("Length of CSI Frames", fontsize=config.Fig.axisLabelSize)

  fig.savefig(str(outPath), bbox_inches="tight")
  plt.close(fig)


def plotBarWithError(
  outPath: Path,
  data: tuple | list | np.ndarray,
  labelList: tuple | list,
  yLabel: str,
  title: str = "",
  yLimit: tuple = (),
):
  fig, ax = plt.subplots()

  length = len(labelList)
  cm = plt.get_cmap(config.Fig.boxColormap, length)
  for i in range(length):
    mean = data[i].mean()
    yerr = [
      [mean - data[i].min()],
      [data[i].max() - mean],
    ]
    ax.bar(i, mean, yerr=yerr, color=cm(i))
  ax.set_xticks(range(length), labels=labelList)

  if title:
    ax.set_title(title, fontsize=config.Fig.titleSize)

  if yLimit:
    ax.set_ylim(yLimit)

  ax.yaxis.set_major_formatter(percentFormatter1)
  ax.set_ylabel(yLabel, fontsize=config.Fig.axisLabelSize)
  ax.tick_params(axis="x", labelsize=config.Fig.tickSize)
  ax.tick_params(axis="y", labelsize=config.Fig.tickSize)

  ax.yaxis.grid(True)

  fig.savefig(str(outPath), bbox_inches="tight")
  plt.close(fig)


def plotPckBar(outPath: Path, perfList: list[Perf], thr, pck):
  dataList: list[np.ndarray] = []
  labelList: list[str] = []

  for perf in perfList:
    label = str(perf)
    labelList.append(label)

    data = perf.valThrPckNp[thr].mean(2)  # Model, Epoch, Pck
    maxEpochList = perf.valThrPckMaxEpochList[thr]
    data = data[np.arange(maxEpochList.size), maxEpochList, pck]  # Model, Epoch
    dataList.append(data)

  plotBarWithError(outPath, dataList, labelList, f"PCK@{thr} (%)")


def plotPckLineWithError(outPath: Path, perfList: list[Perf], thr, pck):
  fig, ax = plt.subplots(figsize=(7, 4))
  for i, perf in enumerate(perfList):
    pcks = perf.valThrPckNp[thr]  # Model, Epoch, Pck
    pcks = pcks.mean(2)[:, :, pck]  # Model, Epoch

    y = pcks.mean(0)  #  Epoch
    y1 = y + pcks.std(0)  # Epoch
    y2 = y - pcks.std(0)  # Epoch

    lineColor = lineColormap(i)
    fillColor = fillColormap(i)

    x = range(y.size)
    label = str(perf)
    if label in map_.legend:
      label = map_.legend[label]
    ax.plot(x, y, label=label, linewidth=linewidth, color=lineColor)
    ax.fill_between(x, y1, y2, alpha=fillAlpha, color=fillColor)

  ax.yaxis.set_major_formatter(percentFormatter0)
  ax.set_xlabel("Epoch", fontsize=config.Fig.axisLabelSize)
  ax.set_ylabel(f"PCK@{thr} (%)", fontsize=config.Fig.axisLabelSize)
  ax.tick_params(axis="x", labelsize=config.Fig.tickSize)
  ax.tick_params(axis="y", labelsize=config.Fig.tickSize)

  ax.legend(fontsize=config.Fig.legendSize)
  ax.yaxis.grid(True)

  fig.savefig(str(outPath), bbox_inches="tight")
  plt.close(fig)


def plotPckBox(outPath: Path, perfList: list[Perf], thr, pck):
  dataList: list[np.ndarray] = []
  labelList: list[str] = []

  for perf in perfList:
    label = str(perf)
    labelList.append(label)

    data = perf.valThrPckNp[thr].mean(2)  # Model, Epoch, Pck
    maxEpochList = perf.valThrPckMaxEpochList[thr]
    data = data[np.arange(maxEpochList.size), maxEpochList, pck]  # Model, Epoch
    dataList.append(data)

  plotBox(outPath, dataList, labelList, f"PCK@{thr} (%)")


def plotJointPckBox(outDir: Path, perfList: list[Perf], thr):
  outDir.mkdir(parents=True, exist_ok=True)

  for n, perf in enumerate(perfList):
    name = str(perf)

    dataList: list[np.ndarray] = []
    lableList: list[str] = []

    for n in range(config.Global.nJoints + 1):
      data = perf.valThrPckNp[thr].mean(2)  # Model, Epoch, Pck
      maxEpochList = perf.valThrPckMaxEpochList[thr]
      data = data[np.arange(maxEpochList.size), maxEpochList, n]  # Model, Epoch, Pck
      dataList.append(data)

      if n == 0:
        lableList.append("Mean")
      else:
        lableList.append(str(n - 1))

    plotBox(
      outDir / f"{name}.pdf",
      dataList,
      lableList,
      f"PCK@{thr} (%)",
    )


def saveAll(outDir: Path, orderedPerfList: list[Perf], thrList: tuple):
  for thr in tqdm(thrList, desc="Threshold"):
    subOutDir = outDir / f"pck@{thr:.2f}"
    subOutDir.mkdir(parents=True, exist_ok=True)
    saveCsv(subOutDir / "pck.csv", orderedPerfList, thr)
    plotJointPckBox(subOutDir / "joints.box", orderedPerfList, thr)
    for n in range(config.Global.nJoints + 1):
      prefix = f"{n}".zfill(2)
      plotPckLineWithError(subOutDir / f"{prefix}.line.pdf", orderedPerfList, thr, n)
      plotPckBar(subOutDir / f"{prefix}.bar.pdf", orderedPerfList, thr, n)
      plotPckBox(subOutDir / f"{prefix}.box.pdf", orderedPerfList, thr, n)


confList = [config.Stats()]

if __name__ == "__main__":
  inDir = Path(config.Stats.inDir)
  outDir = Path(config.Stats.outDir)

  outDir.mkdir(parents=True, exist_ok=True)

  perfList: list[Perf] = []
  for inSubDir in sorted(inDir.iterdir()):
    if inSubDir.name not in config.Stats.exceptSubDirList:
      perfList.append(Perf(inSubDir, pckThrList=config.Stats.pckThrList))

  sortedList = sorted(perfList, key=lambda x: x.order)
  saveAll(outDir, sortedList, config.Stats.pckThrList)
