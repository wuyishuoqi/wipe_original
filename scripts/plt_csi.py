#!/usr/bin/env python3

from sklearn.preprocessing import minmax_scale

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

dataDir = Path(
  "/root/docker/cache/parsed/free-small/[2023-11-27T15;55;55.063456+08;00][30FPS][h265,h265]/mags"
)

for dataPath in dataDir.glob("*.npy"):
  csiNp = np.load(dataPath)

  outDir = outPath = Path("out/csi") / dataPath.stem
  outDir.mkdir(parents=True, exist_ok=True)

  fusion = np.mean(csiNp, axis=0)
  transposed = np.transpose(fusion, (1, 2, 0))
  shape = transposed.shape
  reshaped = transposed.reshape(shape[0] * shape[1], shape[2])
  shape = reshaped.shape
  scaled = minmax_scale(reshaped.reshape((1, -1)), axis=1).reshape(shape)
  plt.imsave(
    outDir / "0.webp",
    scaled,
    cmap="GnBu_r",
    pil_kwargs={"lossless": True},
  )

  for i, csi in enumerate(csiNp):
    transposed = np.transpose(csi, (1, 2, 0))
    shape = transposed.shape
    reshaped = transposed.reshape(shape[0] * shape[1], shape[2])

    shape = reshaped.shape
    scaled = minmax_scale(reshaped.reshape((1, -1)), axis=1).reshape(shape)

    plt.imsave(
      outDir / f"{i+1}.webp",
      scaled,
      cmap="GnBu_r",
      pil_kwargs={"lossless": True},
    )
