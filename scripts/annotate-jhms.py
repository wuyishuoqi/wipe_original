#!/usr/bin/env python3

from libs import utils, poseutils
import config

from tqdm import tqdm
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
import json

confList: list[config.Annot] = [
  # config.Annot()
  # config.Annot(
  #   datasetDir=r"/root/docker/data/CSI with frames (2023-11)/parsed/block-small",
  # ),
  # config.Annot(
  #   datasetDir=r"/root/docker/data/CSI with frames (2023-11)/parsed/free-big",
  # ),
  config.Annot(
    datasetDir=r"/root/docker/data/CSI with frames (2023-11)/parsed/free-small",
  ),
  config.Annot(
    datasetDir=r"/root/docker/data/CSI with frames (2023-11)/parsed/free-small-2.4g",
  ),
  # config.Annot(
  #   datasetDir=r"/root/docker/data/CSI with frames (2023-11)/parsed/point-1",
  # ),
  # config.Annot(
  #   datasetDir=r"/root/docker/data/CSI with frames (2023-11)/parsed/point-2",
  # ),
  # config.Annot(
  #   datasetDir=r"/root/docker/data/CSI with frames (2023-11)/parsed/point-3",
  # ),
  # config.Annot(
  #   datasetDir=r"/root/docker/data/CSI with frames (2023-11)/parsed/point-4",
  # ),
  # config.Annot(
  #   datasetDir=r"/root/docker/data/CSI with frames (2023-11)/parsed/straight-1",
  # ),
  # config.Annot(
  #   datasetDir=r"/root/docker/data/CSI with frames (2023-11)/parsed/straight-2",
  # ),
  # config.Annot(
  #   datasetDir=r"/root/docker/data/CSI with frames (2023-11)/parsed/vertical",
  # ),
]

nSubThreads = config.Global.nSubThreads


def saveJhmsFromMmpose(outPath: Path, raw: dict):
  jhms = poseutils.keypoints2Jhms(
    poseutils.mmpose2np(raw),
    config.Global.outRes,
    config.Global.scaleFactor,
    config.Train.mmposeThr,
    config.Train.gaussianCov,
  )
  np.save(outPath, jhms.astype(np.float32))


def saveAll(conf: config.Annot):
  for subdir in tqdm(
    tuple(Path(conf.datasetDir).iterdir()), desc="Subdir", leave=False
  ):
    outDir = subdir / conf.jhmsDir
    outDir.mkdir(parents=True, exist_ok=True)
    mmposePathList = tuple((subdir / conf.mmposeDir).iterdir())

    total = len(mmposePathList)
    with tqdm(total=total, desc="Saving", leave=False) as pbar:
      with ThreadPoolExecutor(max_workers=nSubThreads) as executor:
        argsQueue = Queue(maxsize=nSubThreads * 2)
        futureSet = set()
        for mmposePath in mmposePathList:
          with open(mmposePath, "rt") as f:
            argsQueue.put(
              {
                "outPath": outDir / mmposePath.with_suffix(".npy").name,
                "raw": json.load(f),
              }
            )
          future = executor.submit(utils.funFromQ, saveJhmsFromMmpose, argsQueue, pbar)
          futureSet.add(future)

          if len(futureSet) >= nSubThreads * 10:
            utils.clearFutureSet(futureSet)
        utils.clearFutureSet(futureSet)


if __name__ == "__main__":
  for conf in tqdm(confList, desc="Conf"):
    saveAll(conf)
