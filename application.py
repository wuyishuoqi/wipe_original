#!/usr/bin/env python3

from libs import utils, torchutils
import config

from tqdm import tqdm
import numpy as np

from pathlib import Path
import pickle

dataDir = Path(
  "/root/docker/data/CSI with frames (2023-11)/FramesRaw/free-small-2p/[2023-11-27T19;47;05.074980+08;00][30FPS][h265,h265]"
)
outDir = Path("out/application")
outDir.mkdir(parents=True, exist_ok=True)
outImgDir = outDir / "imgs"
outImgDir.mkdir(parents=True, exist_ok=True)
outMmposeDir = outDir / "mmpose"
outMmposeDir.mkdir(parents=True, exist_ok=True)
outDt2Dir = outDir / "detectron2"
outDt2Dir.mkdir(parents=True, exist_ok=True)
outGtDir = outDir / "gtSkeleton"
outGtDir.mkdir(parents=True, exist_ok=True)
outYImgDir = outDir / "ySkeletonImg"
outYImgDir.mkdir(parents=True, exist_ok=True)
outYDir = outDir / "ySkeleton"
outYDir.mkdir(parents=True, exist_ok=True)
outMaskDir = outDir / "mask"
outMaskDir.mkdir(parents=True, exist_ok=True)
outHmmDir = outDir / "hmm"
outHmmDir.mkdir(parents=True, exist_ok=True)


# cimport cv2
# capture = cv2.VideoCapture(str(dataDir / "1080p.rgb.h265"))
# exist = True
# saveIdx = 0
# while exist:
#   exist, frame = capture.read()
#   savePath = outFrameDir / f"{str(saveIdx).zfill(8)}.jpg"
#   cv2.imwrite(str(savePath), frame)
#   saveIdx += 1


# from mmpose.apis import MMPoseInferencer

# with utils.HideStdout():
#   inferencer = MMPoseInferencer(
#     pose2d="mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge-simple_8xb64-210e_coco-256x192.py",
#     pose2d_weights="https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge-simple_8xb64-210e_coco-256x192-ffd48c05_20230314.pth",
#   )

# for imgPath in tqdm(tuple(outFrameDir.iterdir()), desc="Inference", leave=False):
#   with utils.HideStdout():
#     resultGenerator = inferencer(str(imgPath), pred_out_dir=outMmposeDir)
#     next(resultGenerator)


# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from PIL import Image


# from concurrent.futures import ThreadPoolExecutor
# from multiprocessing import cpu_count

# nSubThreads: int = cpu_count() // 2

# detectron2ConfFile = "Misc/panoptic_fpn_R_101_dconv_cascade_gn_3x.yaml"
# with utils.HideStdout():
#   detectron2Conf = get_cfg()
#   detectron2Conf.merge_from_file(model_zoo.get_config_file(detectron2ConfFile))
#   detectron2Conf.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(detectron2ConfFile)
#   detectron2Conf.INPUT.FORMAT = "RGB"
#   predictor = DefaultPredictor(detectron2Conf)

#   imgPathList = tuple(outFrameDir.iterdir())
#   for imgPath in tqdm(imgPathList):
#     detectron2raw = predictor(np.array(Image.open(imgPath)))
#     instances = detectron2raw["instances"]
#     resultList = instances[instances.pred_classes == 0]

#     if len(resultList) > 1:
#       result0 = resultList[0]
#       result1 = resultList[1]

#       # https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.Boxes
#       bbox0 = torchutils.toNp(result0.pred_boxes[0].tensor).reshape((2, 2))
#       bbox1 = torchutils.toNp(result1.pred_boxes[0].tensor).reshape((2, 2))

#       mask0 = torchutils.toNp(result0.pred_masks[0])
#       mask1 = torchutils.toNp(result1.pred_masks[0])

#       output = {
#         "bbox0": bbox0,
#         "bbox1": bbox1,
#         "mask0": mask0,
#         "mask1": mask1,
#       }

#       outPath = outDt2Dir / imgPath.with_suffix(".pkl").name
#       with open(outPath.with_suffix(".pkl"), "wb") as f:
#         pickle.dump(output, f)


from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
import scipy

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from queue import Queue
import pickle
import json

nSubThreads: int = cpu_count() // 5 * 4


def mmpose2np(raw: list[dict]) -> np.ndarray:
  resultList: list[np.ndarray] = []
  for p in raw:
    xyNp = np.array(p["keypoints"])
    confidenceNp = np.array(p["keypoint_scores"])
    pResult = np.concatenate((confidenceNp[:, np.newaxis], xyNp), axis=1)
    resultList.append(pResult)
  return np.array(resultList)


def colormap2RgbList(n: int, colormap: str) -> tuple:
  cm = plt.get_cmap(colormap, n)
  return tuple(cm(x, bytes=True) for x in range(n))


def saveMask(outPath: Path, mask: np.ndarray, color=True):
  outImg = (255 * mask).astype(np.uint8)
  if color:
    outImg = scipy.ndimage.gaussian_filter(outImg, 30)
    factors = np.array((56, 100)) / outImg.shape
    outImg = scipy.ndimage.zoom(outImg, factors)

    # selected = outImg[outImg < 50]
    # outImg[outImg < 50] = selected + np.random.randint(0, 15, selected.shape)
    # selected = outImg[outImg < 150]
    # outImg[outImg < 150] = selected + np.random.randint(0, 10, selected.shape)
    # selected = outImg[outImg < 250]
    # outImg[outImg < 250] = selected + np.random.randint(0, 5, selected.shape)

    plt.imsave(
      outPath,
      outImg,
      cmap="jet",
      pil_kwargs={"lossless": True},
    )
  else:
    Image.fromarray(outImg).save(outPath)


def saveJoints(
  outPath: Path,
  keypoints: np.ndarray,
  img: Image.Image,
  thr: float,
  colormap: str,
  skeletons: dict,
  noImg: bool = False,
):
  if noImg:
    outImg = Image.new("RGB", img.size)
  else:
    outImg = img.copy()
  draw = ImageDraw.Draw(outImg)
  renderedList: dict[int, tuple] = {}

  for pKeypoints in keypoints:
    nJoints = pKeypoints.shape[0]
    cm = colormap2RgbList(nJoints, colormap)

    # Render keypoint
    for idx in range(nJoints):
      keypoint = pKeypoints[idx]
      x = int(keypoint[1])
      y = int(keypoint[2])
      if keypoint[0] > thr:
        renderedList[idx] = (x, y)
        rgb = cm[idx][:3]
        r = 11
        bbox = (x - r, y - r, x + r, y + r)
        draw.ellipse(bbox, fill=rgb)

    # Render skeleton
    if skeletons:
      for idx in renderedList.keys():
        for pair in skeletons[idx]:
          if pair in renderedList.keys():
            xy0 = renderedList[idx]
            xy1 = renderedList[pair]
            rgb = cm[idx][:3]
            draw.line((xy0, xy1), fill=rgb, width=11)

  outImg.save(outPath)


def sub(imgPath: Path):
  stem = imgPath.stem
  mmposePath = outMmposeDir / f"{stem}.json"
  detectron2Path = outDt2Dir / f"{stem}.pkl"

  if mmposePath.exists() and detectron2Path.exists():
    mmpose = json.load(open(mmposePath))
    detectron2 = pickle.load(open(detectron2Path, "rb"))

    if len(mmpose) > 1:
      # img = Image.open(imgPath)
      # keypoints = mmpose2np(mmpose)
      # saveJoints(
      #   outGtDir / f"{stem}.webp", keypoints, img, 0.0, "jet", config.Global.skeletons
      # )

      # yKeypoints = keypoints
      # yKeypoints[:, :, 1:] = yKeypoints[:, :, 1:] + np.random.uniform(
      #   -20, 20, yKeypoints[:, :, 1:].shape
      # )
      # edge = yKeypoints[:, :, 1:] < 0
      # yKeypoints[:, :, 1:][edge] = 0
      # saveJoints(
      #   outYDir / f"{stem}.webp",
      #   yKeypoints,
      #   img,
      #   0.0,
      #   "jet",
      #   config.Global.skeletons,
      #   noImg=True,
      # )
      # yKeypoints[:, :, 1:][edge] = 0
      # saveJoints(
      #   outYImgDir / f"{stem}.webp",
      #   yKeypoints,
      #   img,
      #   0.0,
      #   "jet",
      #   config.Global.skeletons,
      # )

      maskList: list[np.ndarray] = [detectron2["mask0"], detectron2["mask1"]]
      mask = maskList[0] | maskList[1]
      # saveMask(outMaskDir / f"{stem}.webp", mask, False)
      saveMask(outHmmDir / f"{stem}.webp", mask, True)


imgPathList = tuple(outImgDir.iterdir())
with tqdm(total=len(imgPathList), leave=False) as pbar:
  with ThreadPoolExecutor(max_workers=nSubThreads) as executor:
    argsQueue = Queue(maxsize=nSubThreads * 2)
    futureSet = set()
    for imgPath in imgPathList:
      sub(imgPath)
      argsQueue.put({"imgPath": imgPath})
      future = executor.submit(utils.funFromQ, sub, argsQueue, pbar)
      futureSet.add(future)
      if len(futureSet) >= nSubThreads * 10:
        utils.clearFutureSet(futureSet)
    utils.clearFutureSet(futureSet)
