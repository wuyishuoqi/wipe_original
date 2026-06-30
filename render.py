#!/usr/bin/env python3

from libs import utils
import config
import dataset as MyDateset
import libs.poseutils as poseutils

from PIL import Image, ImageDraw
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
import json
import pickle


confList: list[config.Render] = [
  config.Render(
    inDir="infer/best",
    outDir="render/best",
  ),
  # config.Render(
  #   datasetDir="/root/docker/data/CSI with frames (2023-11)/parsed/block-small",
  #   inDir="infer/block",
  #   outDir="render/block",
  # ),
]

nSubThreads = config.Global.nSubThreads


def jhms2ScaledXy(jhms: np.ndarray, outRes: tuple[int, int], decode):
  jhmsRes = jhms.shape[-2:]
  factors = (outRes[0] / jhmsRes[0], outRes[1] / jhmsRes[1])
  jhms = jhms[np.newaxis, :, :, :]
  keypoints = poseutils.batchedJhms2xys(jhms, decode)
  keypoints = keypoints.squeeze()
  keypoints[:, 1] *= factors[1]
  keypoints[:, 2] *= factors[0]

  return keypoints


def colormap2RgbList(n: int, colormap: str) -> tuple:
  cm = plt.get_cmap(colormap, n)
  return tuple(cm(x, bytes=True) for x in range(n))


def saveJhms(outDir: Path, jhms: np.ndarray, colormap: str, suffix: str = "jhms"):
  for idx, jhm in enumerate(jhms):
    scaled = (255 * jhm).astype(np.uint8)

    filename = f"{suffix}.{idx}.webp"
    plt.imsave(
      outDir / filename,
      scaled,
      cmap=colormap,
      pil_kwargs={"lossless": True},
    )

  fused = (255 * jhms.sum(axis=0)).astype(np.uint8)
  filename = f"{suffix}.webp"
  plt.imsave(
    outDir / filename,
    fused,
    cmap=colormap,
    pil_kwargs={"lossless": True},
  )

#画骨骼关键点
def saveJoints(
  outPath: Path,
  keypoints: np.ndarray,
  img: Image.Image,
  thr: float,
  colormap: str,
  skeletons: dict,
):
  outImg = img.copy()
  draw = ImageDraw.Draw(outImg)
  renderedList: dict[int, tuple] = {}

  midX=int((keypoints[5][1]+keypoints[6][1])/2)
  midY=int((keypoints[5][2]+keypoints[6][2])/2)
  # keypoints=np.append(keypoints,[[1,midX,midY]],axis=0)
  keypoints=np.insert(keypoints,7,[1,midX,midY],axis=0)
  nJoints = keypoints.shape[0]
  cm = colormap2RgbList(nJoints, colormap)
  # Render keypoint
  for idx in range(nJoints):
    keypoint = keypoints[idx]
    x = int(keypoint[1])
    y = int(keypoint[2])
    print(str(idx)+":"+str(x)+" "+str(y))
    if keypoint[0] > thr:
      renderedList[idx] = (x, y)
      rgb = cm[idx][:3]
      r = 8
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
          draw.line((xy0, xy1), fill=rgb, width=8)

  outImg.save(outPath)


def saveMask(outPath: Path, mask: np.ndarray, color=True):
  outImg = (255 * mask).astype(np.uint8)
  if color:
    plt.imsave(
      outPath,
      outImg,
      cmap="jet",
      pil_kwargs={"lossless": True},
    )
  else:
    Image.fromarray(outImg).save(outPath)


def saveMaskWithImg(outPath: Path, mask: np.ndarray, img: Image.Image, thr: float):
  outRes = img.size[:2]
  result = np.where(mask > thr, 0, 255).astype(np.uint8)
  resizedRgbMask = (
    Image.fromarray(result).resize(outRes, Image.Resampling.NEAREST).convert("RGB")
  )
  Image.blend(img, resizedRgbMask, 0.3).save(outPath)


def saveRender(
  outDir: Path,
  imgPath: Path,
  jhmsPath: Path,
  maskPath: Path,
  mmposePath: Path,
  detectron2Path: Path,
  conf: config.Render,
):
  outDir.mkdir(parents=True, exist_ok=True)
  img = Image.open(imgPath)
  img.save(outDir / "img.webp")

  jhms: np.ndarray = np.load(jhmsPath)
  saveJhms(outDir, jhms, conf.heatmapColormap)
  keypoints = jhms2ScaledXy(jhms, config.Global.originRes, conf.decode)
  saveJoints(
    outDir / "joints.webp",
    keypoints,
    Image.new("RGB", img.size, (0, 0, 0)),
    conf.jhmsThr,
    conf.poseColormap,
    config.Global.skeletons,
  )
  saveJoints(
    outDir / "joints.img.webp",
    keypoints,
    img,
    conf.jhmsThr,
    conf.poseColormap,
    config.Global.skeletons,
  )

  mask = np.load(maskPath)
  saveMask(outDir / "mask.raw.webp", mask)
  saveMaskWithImg(outDir / "mask.img.webp", mask, img, conf.maskThr)

  # Render ground truth
  with open(mmposePath, "rt") as f:
    rawMmpose = json.load(f)
  gtKeypoints = poseutils.mmpose2np(rawMmpose)
  gtJhms = poseutils.keypoints2Jhms(
    poseutils.mmpose2np(rawMmpose),
    config.Global.outRes,
    config.Global.scaleFactor,
    conf.mmposeThr,
    config.Train.gaussianCov,
  )
  saveJhms(outDir, gtJhms, conf.heatmapColormap, "jhms.gt")
  saveJoints(
    outDir / "joints.gt.webp",
    gtKeypoints,
    Image.new("RGB", img.size, (0, 0, 0)),
    conf.jhmsThr,
    conf.poseColormap,
    config.Global.skeletons,
  )
  saveJoints(
    outDir / "joints.gt.img.webp",
    gtKeypoints,
    img,
    conf.jhmsThr,
    conf.poseColormap,
    config.Global.skeletons,
  )

  with open(detectron2Path, "rb") as f:
    gtMask = pickle.load(f)["zoomedMask"]
  saveMask(outDir / "mask.gt.webp", gtMask, False)
  saveMaskWithImg(outDir / "mask.gt.img.webp", gtMask, img, 0)

def saveJointsRender(
  outDir: Path,
  jhmsPath: np.ndarray,
  conf: config.Render,
):
  jhms: np.ndarray = jhmsPath
  # jhms=np.load("/media/zxx/新加卷/wipe/wipe/out/infer/best/00000000.jhms.npy")
  # saveJhms(outDir, jhms, conf.heatmapColormap)
  keypoints = jhms2ScaledXy(jhms, config.Global.originRes, conf.decode)
  saveJoints(
    "joints.webp",
    keypoints,
    Image.new("RGB", (1920,1080), (0, 0, 0)),
    conf.jhmsThr,
    conf.poseColormap,
    config.Global.skeletons,
  )

def render(conf: config.Render):
  outDir = Path(config.Global.outDir) / conf.outDir

  dataset = MyDateset.Infer(
    Path(conf.datasetDir),
    config.Train.nTimestep,
    config.Train.nSubcarrier,
    config.Train.csiSubDir,
    "mean",
  )
  inDir = Path(config.Global.outDir) / conf.inDir

  jhmsPathList = sorted(inDir.glob("*.jhms.npy"))
  maskPathList = sorted(inDir.glob("*.mask.npy"))
  idxList = [int(x.stem.split(".")[0]) for x in jhmsPathList]
  magPathList = [dataset.getCsiPath(x) for x in idxList]

  imgPathList = [
    x.parent.parent / config.Pre.parsedFramesDir / x.with_suffix(".jpg").name
    for x in magPathList
  ]
  gtMmposePathList = [
    x.parent.parent / config.Annot.mmposeDir / x.with_suffix(".json").name
    for x in magPathList
  ]
  gtDetectron2PathList = [
    x.parent.parent / config.Annot.detectron2Dir / x.with_suffix(".pkl").name
    for x in magPathList
  ]

  total = len(magPathList)
  with tqdm(total=total, desc="Rendering", leave=False) as pbar:
    with ThreadPoolExecutor(max_workers=nSubThreads) as executor:
      argsQueue = Queue(maxsize=nSubThreads * 2)
      futureSet = set()
      for i in range(total):
        imgPath = imgPathList[i]
        jhmsPath = jhmsPathList[i]
        maskPath = maskPathList[i]
        gtMmposePath = gtMmposePathList[i]
        gtDetectron2Path = gtDetectron2PathList[i]

        subdir = f"{idxList[i]}".zfill(8)
        argsQueue.put(
          {
            "outDir": outDir / subdir,
            "imgPath": imgPath,
            "jhmsPath": jhmsPath,
            "maskPath": maskPath,
            "mmposePath": gtMmposePath,
            "detectron2Path": gtDetectron2Path,
            "conf": conf,
          }
        )
        future = executor.submit(utils.funFromQ, saveRender, argsQueue, pbar)
        futureSet.add(future)

        if len(futureSet) >= nSubThreads * 10:
          utils.clearFutureSet(futureSet)
      utils.clearFutureSet(futureSet)

def render_single_file(conf: config.Render, file_path: str):
    outDir = Path(config.Global.outDir) / conf.outDir

    # 读取单个 .npy 文件
    data = np.load(file_path)

    # 在这里进行渲染处理，并将结果保存到输出目录中

    # 示例：保存渲染结果到输出目录
    subdir = Path(file_path).stem
    output_path = outDir / subdir
    output_path.mkdir(parents=True, exist_ok=True)
    np.save(output_path / "rendered_data.npy", data)

    # 返回处理后的结果
    return data

def start():
  for conf in tqdm(confList, desc="Conf"):
    render(conf)
if __name__ == "__main__":
  # for conf in tqdm(confList, desc="Conf"):
  #   render(conf)
  start()