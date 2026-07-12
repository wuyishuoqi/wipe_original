import libs.posedataset as posedataset

from dataclasses import dataclass
from multiprocessing import cpu_count


@dataclass(frozen=True)
class Global:
  outDir: str = "out"
  nSubThreads: int = cpu_count() // 2
  nJoints: int = 17
  skeletons = posedataset.COCO.skeletons

  outRes: tuple[int, int] = (56, 100)
  originRes: tuple[int, int] = (1080, 1920)
  scaleFactor: tuple[float, float] = (
    outRes[0] / originRes[0],
    outRes[1] / originRes[1],
  )


@dataclass(frozen=True)
class Pre:
  rawDir: str = r"/home/teacher2/下载/desk-chair-350-raw-preprocess/"
  # rawDir: str = r"/root/docker/data/CSI with frames (2023-11)/FramesRaw/free-small-2.4g"
  rawFrameTimestampFile: str = "rgb.timestamp.txt"

  parsedDir: str = r"/home/teacher2/文档/obstacle-no-zhedang/"
  # parsedDir: str = r"/root/docker/data/CSI with frames (2023-11)/parsed/free-small-2.4g"
  parsedStatsDir: str = "stats"
  parsedAlignedFrameTimestampFile: str = "frame.timestamp.npy"
  parsedSyncedMagDir: str = "mags"
  parsedSyncedPhaseDir: str = "phases"
  parsedFramesDir: str = "frames"

  deltaHistBins: int = 100
  fps: int = 30
  nCsi: int = 9


@dataclass(frozen=True)
class Annot:
  datasetDir: str = Pre.parsedDir

  jhmsDir: str = "jhms"

  mmposeDir: str = "mmpose"
  detectron2Dir: str = "detectron2"

  pafsDir: str = "openpose-paf"
  openposeJsonDir: str = "openpose-json"

#该类里面的数据不可变
@dataclass(frozen=True)
class Train:
  nSubThreads: int = int(cpu_count() * 7 / 8)
  gpus: str = "0"

  outDir: str = r"train/temp"

  trainRatio: float = 0.8
  nEpoch: int = 20
  batchSize: int = 120

  saveEpochStep: int = 1
  valEpochMilestone: tuple = (1,)

  model: str = "Unet3x3"

  nTimestep: int = 9
  nSubcarrier: int = 114
  datasetDir: str = Pre.parsedDir
  csiSubDir: str = Pre.parsedSyncedMagDir

  fusion: str = "mean"
  decode: str = "dark"

  mmposeThr: float = 0.0
  gaussianCov: float = 0.7

  # Loss
  mWeightJhmsParam: tuple[float, float] = (1, 1)  # (0, 0)
  jhmsRegularWeight: float = 1e-7  #  l2 = 1e-5
  jhmsRegularNrom: int = 1

  maskWeight: float = 2e-4
  maskRegularWeight: float = 0
  maskRegularNrom: int = 1

  pckThrList: tuple = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5)

  # For validation
  weightsFile: str = r"wipe\out\ours\0\epoch-19\weights.pt"
  idxFile: str = r"wipe\out\ours\0\00000500.npy"

  def __post_init__(self):
    import model.my
    import inspect

    modelList: set[str] = set()
    for name, _ in inspect.getmembers(model.my, inspect.isclass):
      modelList.add(name)
    assert self.model in modelList, self.model

    assert self.fusion in ("mean", "max", "min", "pca", "ica", "random", "none")
    assert self.decode in ("", "max", "dark")


@dataclass(frozen=True)
class Val(Train):
  outDir: str = "val"


@dataclass(frozen=True)
class Infer(Train):
  outDir: str = "infer"


@dataclass(frozen=True)
class Render:
  datasetDir: str = Pre.parsedDir
  inDir: str = Infer.outDir
  outDir: str = "render"

  mmposeThr: float = 0
  jhmsThr: float = 0
  maskThr: float = 0.5

  decode = Train.decode

  poseColormap: str = "gist_rainbow"
  heatmapColormap: str = "viridis"


@dataclass(frozen=True)
class Fig:
  outDir: str = r"out/stats"

  titleSize = 17
  axisLabelSize = 13
  tickSize = 12
  legendSize = tickSize

  lineColormap: str = "Set1"
  fillColormap: str = "Pastel1"
  errorFillAlpha = 0.3
  lineWidth: float = 2

  boxColormap: str = "Pastel1"


@dataclass(frozen=True)
class Stats:
  inDir: str = r"out/train/times/"
  exceptSubDirList: tuple = ("norm2.backup", "womask", "woreg.backup", "best")
  outDir: str = r"out/stats"

  # pckThrList = Train.pckThrList
  pckThrList = (0.2,)

  order = {
    "norm1": 0,
    "ours": 0,
    "evtformer": 1,
    "woreg": 2,
    "norm2": 3,
    "womw": 4,
    "womask": 5,
  }
