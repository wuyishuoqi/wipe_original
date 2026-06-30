from sklearn import decomposition
import numpy as np
import scipy
import xarray as xr

from datetime import datetime, timezone
import bisect


def iso2datetimeNp(iso: str) -> np.datetime64:
  dt = datetime.fromisoformat(iso)
  return datetime2datetimeNp(dt)


def datetime2datetimeNp(dt: datetime) -> np.datetime64:
  withOutTz = dt.astimezone(timezone.utc).replace(tzinfo=None)
  return np.datetime64(withOutTz)


def datetimeNp2iso(dt64: np.datetime64) -> str:
  dt: datetime = dt64.item()
  return dt.astimezone().isoformat()


def alignTimestampList(
  mainTimestampList: np.ndarray, subTimestampList: np.ndarray
) -> tuple[int, int]:
  alignedSubIdx = np.argmax(
    (subTimestampList - mainTimestampList[0]) > np.timedelta64()
  )

  alignedSubEnd = np.argmax(
    (subTimestampList - mainTimestampList[-1]) > np.timedelta64()
  )

  idx = int(alignedSubIdx)
  length = int(alignedSubEnd) - idx

  return (idx, length)


def getSyncedNpIdx(
  lo: int,
  centerT: np.datetime64,
  inTimestampNp: np.ndarray,
  n: int,
  interval: np.timedelta64,
) -> list[int]:
  startTimestamp = centerT - interval * n / 2
  startIdx = bisect.bisect_left(inTimestampNp, startTimestamp, lo)
  trimDataIdxList: list[int] = []

  searchInterval = np.timedelta64()
  for idx, inTimestamp in enumerate(inTimestampNp[startIdx:]):
    searchTimestamp = startTimestamp + searchInterval

    if inTimestamp >= searchTimestamp:
      trimDataIdxList.append(startIdx + idx)
      searchInterval += interval

      if len(trimDataIdxList) >= n:
        break

  return trimDataIdxList


def pcaReduce(inNp: np.ndarray, nComponents: int) -> np.ndarray:
  inShape = inNp.shape
  reshapedArray = inNp.reshape([inShape[0], -1])
  pca = decomposition.PCA(n_components=nComponents).fit(reshapedArray)
  reduced = pca.transform(reshapedArray)
  outArray = pca.inverse_transform(reduced)
  return outArray.reshape(inShape)


def interpolateCsi(
  inNp: np.ndarray, timestamp: np.ndarray, resampleRule: str, method="linear"
) -> tuple[np.ndarray, np.ndarray]:
  assert method in (
    "linear",
    "nearest",
    "zero",
    "slinear",
    "quadratic",
    "cubic",
    "polynomial",
    "barycentric",
    "krogh",
    "pchip",
    "spline",
    "akima",
  )

  dataAraay = xr.DataArray(
    inNp, coords={"time": timestamp}, dims=["time", "subcarrier", "tx", "rx"]
  )

  # Resample and interpolate
  resampled = dataAraay.resample(time=resampleRule).asfreq()
  resampledTimestamp: np.ndarray = resampled.coords["time"].to_numpy()
  interpolated = resampled.interpolate_na("time", method=method, use_coordinate=False)
  interpolatedNp = interpolated.to_numpy()

  return interpolatedNp, resampledTimestamp


def zoomNdarray(inNp: np.ndarray, outShape: tuple, **kwargs) -> np.ndarray:
  inShape = inNp.shape
  if inShape == outShape:
    return inNp
  else:
    factors = np.array(outShape) / np.array(inShape)
    return scipy.ndimage.zoom(inNp, factors, **kwargs)
