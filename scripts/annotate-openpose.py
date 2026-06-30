#!/usr/bin/env python3

from libs import utils, nputils
import config

from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import minmax_scale

import numpy as np

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

nSubThreads = config.Global.nSubThreads

confList: list[config.Annot] = [config.Annot()]


def savePafs(outPath: Path, rawPath: Path):
  raw = np.array(Image.open(rawPath))
  nHeatmap = int(raw.shape[1] / 1360)
  splited = np.split(raw, nHeatmap, axis=1)
  # 56 x 768 x 1360, 18 joints, 38 PAFs
  pafs = np.array(splited[18:])
  outShape = (pafs.shape[0],) + config.Global.outRes
  zoomedPafs = nputils.zoomNdarray(pafs, outShape, order=0)
  scaledPafs = minmax_scale(zoomedPafs.reshape((1, -1)), axis=1).reshape(outShape)

  np.save(outPath, scaledPafs.astype(np.float32))


def annotate(conf: config.Annot):
  inDir = Path(conf.datasetDir)
  for subdir in tqdm(tuple(inDir.iterdir())):
    rawDir = subdir / "openpose-raw-heatmap"
    outDir = subdir / "openpose-paf"
    outDir.mkdir(exist_ok=True, parents=True)

    rawPathList = tuple(rawDir.iterdir())
    total = len(rawPathList)

    with tqdm(total=total, desc="Processing", leave=False) as pbar:
      with ThreadPoolExecutor(max_workers=nSubThreads) as executor:
        argsQueue = Queue(maxsize=nSubThreads * 2)
        futureSet = set()
        for rawPath in rawPathList:
          filename = rawPath.with_suffix(".npy").name.replace("_pose_heatmaps", "")
          argsQueue.put(
            {
              "outPath": outDir / filename,
              "rawPath": rawPath,
            }
          )
          future = executor.submit(utils.funFromQ, savePafs, argsQueue, pbar)
          futureSet.add(future)

          if len(futureSet) >= nSubThreads * 10:
            utils.clearFutureSet(futureSet)
        utils.clearFutureSet(futureSet)


if __name__ == "__main__":
  for conf in tqdm(confList, desc="Conf"):
    annotate(conf)
