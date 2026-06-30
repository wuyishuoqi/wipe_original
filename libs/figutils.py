from sklearn import decomposition

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path


def pca(outPath: Path, inNp: np.ndarray, nComponents: int):
  reshapedArray = inNp.reshape([inNp.shape[0], -1])
  pca = decomposition.PCA().fit(reshapedArray)
  explainedVar = pca.explained_variance_ratio_[0:nComponents]

  x = range(1, explainedVar.size + 1)
  fig, ax = plt.subplots()
  ax.bar(x, explainedVar)
  ax.set_xticks(x, x)

  ax.set_title("PCA")
  ax.set_xlabel("Pricipal Component")
  ax.set_ylabel("Explained Variance Ratio")
  ax.grid(axis="y")

  plt.savefig(outPath, bbox_inches="tight")
  plt.close(fig)


def jitter(
  outPath: Path,
  inNp: np.ndarray,
  HistBins: int,
  xLabel: str = "",
  title: str = "",
  density=True,
):
  fig, ax = plt.subplots()
  ax.hist(inNp, HistBins, density=density)

  ax.set_title(title)
  ax.set_xlabel(xLabel)
  if density:
    ax.set_ylabel("Probability Density")
  else:
    ax.set_ylabel("Probability density function")
  ax.grid()

  fig.savefig(str(outPath), bbox_inches="tight")
  plt.close(fig)


def lossWithMetric(
  outPath: Path,
  xlabel: str,
  trainLoss: tuple,
  valLoss: tuple,
  metricName: str,
  metric: tuple,
  titile: str = "",
  lossName: str = "Loss",
):
  fig, lossAx = plt.subplots()
  metricAx = lossAx.twinx()

  lossAx.plot(trainLoss[0], trainLoss[1], color="tab:blue", label="Train Loss")
  lossAx.plot(valLoss[0], valLoss[1], color="tab:orange", label="Val Loss")
  metricAx.plot(metric[0], metric[1], color="tab:red", label=metricName)

  lossAx.set_xlabel(xlabel)
  lossAx.set_ylabel(lossName)
  metricAx.set_ylabel(metricName)
  metricAx.set_title(titile)

  fig.legend(loc="upper center", ncol=3)

  fig.savefig(str(outPath), bbox_inches="tight")
  plt.close(fig)
