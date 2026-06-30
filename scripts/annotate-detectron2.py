#!/usr/bin/env python3

from libs import utils, nputils, torchutils
import config

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

from PIL import Image
from tqdm import tqdm
import torch
import numpy as np
import pickle

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def saveDetectron2(outPath: Path, detectron2raw: dict):
  instances = detectron2raw["instances"]
  resultList = instances[instances.pred_classes == 0]

  if resultList:
    result = resultList[0]

    # https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.Boxes
    bbox = torchutils.toNp(result.pred_boxes[0].tensor).reshape((2, 2))
    zoomedBbox = bbox * np.array(config.Global.scaleFactor[::-1])[np.newaxis, :]
    mask = torchutils.toNp(result.pred_masks[0])
    zoomedMask = nputils.zoomNdarray(mask, config.Global.outRes, order=0)
  else:
    bbox = -1 * np.ones((2, 2))
    zoomedBbox = bbox
    mask = np.zeros(config.Global.originRes, dtype=bool)
    zoomedMask = np.zeros(config.Global.outRes, dtype=bool)

  output = {
    "bbox": bbox,
    "zoomedBbox": zoomedBbox,
    "zoomedMask": zoomedMask,
  }
  with open(outPath.with_suffix(".pkl"), "wb") as f:
    pickle.dump(output, f)


def annotate(conf: config.Annot):
  inDir = Path(conf.datasetDir)

  detectron2ConfFile = "Misc/panoptic_fpn_R_101_dconv_cascade_gn_3x.yaml"

  with utils.HideStdout():
    detectron2Conf = get_cfg()
    detectron2Conf.merge_from_file(model_zoo.get_config_file(detectron2ConfFile))
    detectron2Conf.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(detectron2ConfFile)
    detectron2Conf.INPUT.FORMAT = "RGB"
    predictor = DefaultPredictor(detectron2Conf)

  for subdir in tqdm(tuple(inDir.iterdir()), desc="Subdir", leave=False):
    if subdir.is_dir() and ";" in subdir.name:
      inDir = subdir / "frames"

      outDir = subdir / conf.detectron2Dir
      outDir.mkdir(parents=True, exist_ok=True)

      imgPathList = tuple(inDir.iterdir())
      with tqdm(total=len(imgPathList), desc="Inference", leave=False) as pbar:
        with ThreadPoolExecutor(max_workers=nSubThreads) as executor:
          argsQueue = Queue(maxsize=nSubThreads)
          futureSet = set()
          for imgPath in imgPathList:
            if imgPath.suffix == ".jpg":
              argsQueue.put(
                {
                  "outPath": outDir / imgPath.stem,
                  "detectron2raw": predictor(Image.open(imgPath)),
                }
              )
              future = executor.submit(utils.funFromQ, saveDetectron2, argsQueue, pbar)
              futureSet.add(future)

            if len(futureSet) >= nSubThreads:
              utils.clearFutureSet(futureSet)
          utils.clearFutureSet(futureSet)


if __name__ == "__main__":
  for conf in tqdm(confList, desc="Conf"):
    annotate(conf)
