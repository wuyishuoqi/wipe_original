#!/usr/bin/env python3

from libs import utils
import config

from tqdm import tqdm

from pathlib import Path

confList: list[config.Annot] = [
  # config.Annot()
  #config.Annot(
  #  datasetDir=r"/root/docker/data/CSI with frames (2023-11)/parsed/block-small",
  #),
  # config.Annot(
  #   datasetDir=r"/root/docker/data/CSI with frames (2023-11)/parsed/free-big",
  # ),
  config.Annot(
      datasetDir=r"/media/zxx/新加卷/CSI/parsed/free-small",
    ),
  #   config.Annot(
  #     datasetDir=r"/root/docker/data/CSI with frames (2023-11)/parsed/free-small-2.4g",
  # ),
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


def saveMmpose(outDir: Path, inDir: Path):
  from mmpose.apis import MMPoseInferencer

  with utils.HideStdout():
    inferencer = MMPoseInferencer(
      pose2d="mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge-simple_8xb64-210e_coco-256x192.py",
      pose2d_weights="https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge-simple_8xb64-210e_coco-256x192-ffd48c05_20230314.pth",
    )

  for imgPath in tqdm(tuple(inDir.iterdir()), desc="Inference", leave=False):
    with utils.HideStdout():
      resultGenerator = inferencer(str(imgPath), pred_out_dir=outDir)
      next(resultGenerator)


def annotate(conf: config.Annot):
  inDir = Path(conf.datasetDir)

  for subdir in inDir.iterdir():
    if subdir.is_dir() and ";" in subdir.name:
      inDir = subdir / "frames"

      outJsonDir = subdir / config.Annot.mmposeDir
      outJsonDir.mkdir(parents=True, exist_ok=True)

      saveMmpose(outJsonDir, inDir)


if __name__ == "__main__":
  for conf in tqdm(confList, desc="Conf"):
    annotate(conf)
