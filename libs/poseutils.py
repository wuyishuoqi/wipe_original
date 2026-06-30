from libs.dark import taylor, gaussian_blur

from scipy.stats import multivariate_normal
from sklearn.preprocessing import minmax_scale
import numpy as np


def mmpose2np(raw: dict) -> np.ndarray:
  firstResult = raw[0]
  xyNp = np.array(firstResult["keypoints"])
  confidenceNp = np.array(firstResult["keypoint_scores"])
  return np.concatenate((confidenceNp[:, np.newaxis], xyNp), axis=1)


# def mmpose2np(raw: list[dict]) -> np.ndarray:
#   resultList: list[np.ndarray] = []
#   for p in raw:
#     xyNp = np.array(p["keypoints"])
#     confidenceNp = np.array(p["keypoint_scores"])
#     pResult = np.concatenate((confidenceNp[:, np.newaxis], xyNp), axis=1)
#     resultList.append(pResult)
#   return np.array(resultList)


def keypoints2Jhms(
  keypoints: np.ndarray,
  outRes: tuple[int, int],
  scaleFactor: tuple[float, float],
  thr: float,
  gaussian: float,
) -> np.ndarray:
  heatmapList = []
  for point in keypoints:
    if point[0] > thr:
      x = point[1] * scaleFactor[1]
      y = point[2] * scaleFactor[0]
      heatmap = xy2Heatmap((x, y), outRes, gaussian)
      shape = heatmap.shape
      heatmap = minmax_scale(heatmap.reshape((1, -1)), axis=1).reshape(shape)
    else:
      heatmap = np.zeros(outRes)
    heatmapList.append(heatmap)

  return np.array(heatmapList)


def batchedJhmsMaxXys(jhms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  batchSize = jhms.shape[0]
  numJoints = jhms.shape[1]
  w = jhms.shape[3]
  reshaped = jhms.reshape((batchSize, numJoints, -1))

  maxIdx = np.argmax(reshaped, 2)
  maxVals = np.amax(reshaped, 2)

  maxVals = maxVals.reshape((batchSize, numJoints, 1))
  maxIdx = maxIdx.reshape((batchSize, numJoints, 1))

  maxXys = np.tile(maxIdx, (1, 1, 2)).astype(float)
  maxXys[:, :, 0] = (maxXys[:, :, 0]) % w
  maxXys[:, :, 1] = np.floor((maxXys[:, :, 1]) / w)

  maxMask = np.tile(np.greater(maxVals, 0.0), (1, 1, 2))
  maxMask = maxMask.astype(float)
  maxXys *= maxMask

  return maxVals, maxXys


def batchedJhms2xys(jhms: np.ndarray, mode: str = "") -> np.ndarray:
  maxVals, maxPreds = batchedJhmsMaxXys(jhms)

  if mode == "max":
    preds = maxPreds

  elif mode == "dark":
    preds = maxPreds.copy()
    hm = gaussian_blur(jhms, 5)
    hm = np.maximum(hm, 1e-10)
    hm = np.log(hm)
    for n in range(preds.shape[0]):
      for p in range(preds.shape[1]):
        preds[n, p] = taylor(hm[n][p], preds[n][p])

  else:
    maxRemoved = jhms
    for n in range(maxPreds.shape[0]):
      for j in range(maxPreds.shape[1]):
        maxRemoved[n, j, :] = np.where(
          maxRemoved[n, j, :] == maxVals[n, j], 0, maxRemoved[n, j, :]
        )
    _, secondPreds = batchedJhmsMaxXys(maxRemoved)

    preds = maxPreds.copy()
    for n in range(preds.shape[0]):
      for j in range(preds.shape[1]):
        m = maxPreds[n, j]
        s = secondPreds[n, j]
        if not np.array_equal(m, s):
          preds[n, j] = m + 0.25 * (s - m)

  out = np.concatenate((maxVals, preds), axis=2)
  return out


# https://stackoverflow.com/questions/44945111/how-to-efficiently-compute-the-heat-map-of-two-gaussian-distribution-in-python
def xy2Heatmap(
  xy: tuple[float, float],
  shape: tuple[int, int],
  gaussian: float,
) -> np.ndarray:
  if gaussian:
    cov = np.eye(2) * gaussian
    k = multivariate_normal(mean=xy, cov=cov)

    x = np.arange(0, shape[1])
    y = np.arange(0, shape[0])
    xx, yy = np.meshgrid(x, y)

    xxyy = np.stack([xx.ravel(), yy.ravel()]).T
    heatmap = k.pdf(xxyy).reshape(shape)

  else:
    heatmap = np.zeros(shape)
    heatmap[xy[1], xy[0]] = 1

  return heatmap


def batchedXyAccuracy(
  keypoints: np.ndarray,
  keypointsT: np.ndarray,
  thrList: tuple,
  norm,
) -> dict[int, np.ndarray]:
  dists = np.linalg.norm(keypoints - keypointsT, axis=2) / norm
  mask = dists >= 0

  result = {}
  for thr in thrList:
    correct = mask & (dists < thr)
    total = mask.sum(axis=0)
    sub = correct.sum(axis=0) * 1.0 / total
    sub = np.concatenate(([sub.mean()], sub))
    result[thr] = sub

  return result
