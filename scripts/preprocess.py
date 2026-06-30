#!/usr/bin/env python3

import figutils, nputils, utils
import config

from sklearn.preprocessing import scale
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

confList: list[config.Pre] = [
  # config.Pre()
  config.Pre(
    rawDir=r"/home/teacher2/下载/desk-chair-350-raw-preprocess/",
    parsedDir=r"/home/teacher2/下载/desk-chair-350-parse-preprocess/",
  ),
  # config.Pre(
  #   rawDir=r"/media/zxx/新加卷/CSI/FramesRaw/free-small-2.4g",
  #   parsedDir=r"/media/zxx/新加卷/CSI/parsed/free-small-2.4g",
  # ),
]

nSubThreads = config.Global.nSubThreads
plt.switch_backend("Agg")


class Preprocessor:
  def __init__(self, conf: config.Pre, rawDir, parsedDir) -> None:
    self.conf = conf
    self.rawDir = rawDir
    self.parsedDir = parsedDir
    self.statsDir = parsedDir / conf.parsedStatsDir

    self.rgbVideoPath = tuple(rawDir.glob("*.rgb.*"))[0]
    self.rgbTimestampPath = rawDir / conf.rawFrameTimestampFile
    self.alignedFrameTimestampPath = (
      self.parsedDir / conf.parsedAlignedFrameTimestampFile
    )

    self.magPath = tuple(rawDir.glob("*.mag.npy"))[0]
    #self.phasePath = tuple(rawDir.glob("*.phase.npy"))[0]
    self.csiTimestampPath = tuple(rawDir.glob("*.timestamp.npy"))[0]

    self.syncedMagDir = self.parsedDir / conf.parsedSyncedMagDir
    #self.syncedPhaseDir = self.parsedDir / conf.parsedSyncedPhaseDir
    self.framesDir = self.parsedDir / conf.parsedFramesDir

  def extractAlignedFrames(self):
    outDir = self.framesDir
    outDir.mkdir(parents=True, exist_ok=True)

    csiTimestampNp = np.load(self.csiTimestampPath)
    with open(self.rgbTimestampPath, "rt") as rgbTimestampFile:
      frameTimestampNp = np.array(
        list(map(nputils.iso2datetimeNp, rgbTimestampFile.read().splitlines()))
      )

    # Align frame and CSI
    self.alignedFrameIdx, self.frameLength = nputils.alignTimestampList(
      csiTimestampNp, frameTimestampNp
    )
    self.alignedFrameIdx, self.frameLength = (
      self.alignedFrameIdx + 1,
      self.frameLength - 2,
    )
    newFrameTimestampNp = frameTimestampNp[
      self.alignedFrameIdx : self.alignedFrameIdx + self.frameLength
    ]
    np.save(self.parsedDir / "frame.timestamp.npy", newFrameTimestampNp)

    capture = cv2.VideoCapture(str(self.rgbVideoPath))
    frameEnd = self.alignedFrameIdx + self.frameLength

    # Extract aligned frames
    with tqdm(total=self.frameLength, desc="Extract frames", leave=False) as pbar:
      with ThreadPoolExecutor(max_workers=nSubThreads) as executor:
        argsQueue = Queue(maxsize=nSubThreads * 2)
        futureSet = set()

        frameIdx = 0
        saveIdx = 0
        exist, frame = capture.read()
        while exist:
          if frameIdx >= self.alignedFrameIdx:
            savePath = outDir / f"{str(saveIdx).zfill(8)}.jpg"
            saveIdx += 1

            argsQueue.put({"filename": str(savePath), "img": frame})
            future = executor.submit(utils.funFromQ, cv2.imwrite, argsQueue, pbar)
            futureSet.add(future)

          exist, frame = capture.read()
          frameIdx += 1
          if frameIdx >= frameEnd:
            break

          if len(futureSet) >= nSubThreads * 10:
            utils.clearFutureSet(futureSet)
        utils.clearFutureSet(futureSet)

  def extractSyncedCsi(self, frameRate: int, csiN: int, resampleRule: str = "1ms"):
    csiIntervalUs = int(1e6 / (frameRate * csiN))
    csiInterval = np.timedelta64(csiIntervalUs, "us")

    csiTimestampNp = np.load(self.csiTimestampPath)

    # Analyse CSI data
    deltaMsNp = np.diff(csiTimestampNp).astype("timedelta64[ms]").astype("int")

    rawMagNp: np.ndarray = np.load(self.magPath)
    #rawPhaseNp: np.ndarray = np.load(self.phasePath)

    # PCA stats on raw
    statsDir = self.statsDir
    statsDir.mkdir(parents=True, exist_ok=True)
    # figutils.pca(statsDir / "raw.pca.pdf", rawMagNp, 20)

    # Resample and interpolate
    resampledMagNp, resampledTimestampNp = nputils.interpolateCsi(
      rawMagNp, csiTimestampNp, resampleRule
    )
    #resampledPhaseNp, _ = nputils.interpolateCsi(
      #rawPhaseNp, 
      #csiTimestampNp, resampleRule
    #)

    interpolationRatio = 1 - (csiTimestampNp.size / resampledTimestampNp.size)
    interpolationPercent = f"{interpolationRatio * 101:.3f}%"

    # Z-score normalizate
    shape = resampledMagNp.shape
    normedMagNp = scale(resampledMagNp.reshape((shape[0], -1))).reshape(shape)

    # PCA stats on interpolated data
    # figutils.pca(statsDir / "interpolated.pca.pdf", resampledMagNp, 20)

    # Jitter stats
    meanDeltaMs = deltaMsNp.mean()
    stdDeltaMs = np.std(deltaMsNp)
    maxDeltaMs = deltaMsNp.max()
    minDeltaMs = deltaMsNp.min()

    jitterFigTitle = f"Interpolation Ratio = {interpolationPercent}\nMean = {meanDeltaMs:.3f}ms, STD = {stdDeltaMs:.3f}ms\nMax = {maxDeltaMs:.3f}ms, Min = {minDeltaMs:.3f}ms"
    figutils.jitter(
      self.statsDir / "jitter.pdf",
      deltaMsNp,
      self.conf.deltaHistBins,
      "∆t (ms)",
      jitterFigTitle,
    )

    # Extract synced mag
    frameTimestamp = np.load(self.alignedFrameTimestampPath)
    lo = 0

    outMagDir = self.syncedMagDir
    #outPhaseDir = self.syncedPhaseDir
    outMagDir.mkdir(parents=True, exist_ok=True)
    #outPhaseDir.mkdir(parents=True, exist_ok=True)
    for idx, frameTimestamp in enumerate(
      tqdm(frameTimestamp, desc="Extract CSI", leave=False)
    ):
      syncedMagIdxList = nputils.getSyncedNpIdx(
        lo, frameTimestamp, resampledTimestampNp, csiN, csiInterval
      )
      lo = syncedMagIdxList[-1]

      np.save(
        outMagDir / f"{str(idx).zfill(8)}.npy",
        normedMagNp[syncedMagIdxList].astype(np.float32),
      )
      #np.save(
      #  outPhaseDir / f"{str(idx).zfill(8)}.npy",
      #  resampledPhaseNp[syncedMagIdxList].astype(np.float32),
      #)


def preprocess(conf: config.Pre):
  rawDir = Path(conf.rawDir)
  parsedDir = Path(conf.parsedDir)

  for rawSubDir in tqdm(tuple(rawDir.iterdir()), desc="Subdir", leave=False):
    if rawSubDir.is_dir() and ";" in rawSubDir.name:
      parsedSubDir = parsedDir / rawSubDir.name
      parsedSubDir.mkdir(parents=True, exist_ok=True)

      preprocessor = Preprocessor(conf, rawSubDir, parsedSubDir)
      preprocessor.extractAlignedFrames()
      preprocessor.extractSyncedCsi(conf.fps, conf.nCsi)


if __name__ == "__main__":
  for conf in tqdm(confList, desc="Conf"):
    preprocess(conf)
