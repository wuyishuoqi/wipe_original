#!/usr/bin/env python3

import train
import parsecli
import config

from tqdm import tqdm

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

trianConfList: list[config.Train] = [
  # config.Train(),
]
valConfList: list[config.Val] = [config.Val()]
inferConfList: list[config.Infer] = [
  # ===== 当前活跃 =====
  # config.Infer(
  #   model="Evtformer",
  #   fusion="none",
  #   weightsFile=r"out/train/model/Evtformer/7/epoch-19/weights.pt",
  #   outDir="infer/evt-v8-desk-chair",
  # ),

  # ===== Wpformer =====
  # config.Infer(
  #   model="Wpformer", fusion="none",
  #   weightsFile=r"out/train/model/Wpformer/0/epoch-19/weights.pt",
  #   outDir="infer/wpformer-desk-chair",
  # ),
  # config.Infer(
  #   model="Wpformer", fusion="none",
  #   weightsFile=r"out/train/model/Wpformer/1/epoch-19/weights.pt",
  #   outDir="infer/wpformer-noostruct",
  # ),
  # config.Infer(
  #   model="Wpformer", fusion="none",
  #   weightsFile=r"out/train/model/Wpformer/2/epoch-19/weights.pt",
  #   outDir="infer/wpformer-obstacle",
  # ),

  # ===== Evtformer =====
  # config.Infer(
  #   model="EvtformerV4", fusion="none",
  #   weightsFile=r"out/train/model/Evtformer/1/epoch-19/weights.pt",
  #   outDir="infer/evt-v4-noostruct",
  # ),
  # config.Infer(
  #   model="EvtformerV7", fusion="none",
  #   weightsFile=r"out/train/model/Evtformer/5/epoch-19/weights.pt",
  #   outDir="infer/evt-v7-desk-chair",
  # ),低验
  config.Infer(
    model="Evtformer", fusion="none",
    weightsFile=r"out/train/model/Evtformer/9/epoch-19/weights.pt",
    outDir="infer/evt-v8-obstacle",
  ),

  # ===== Wisppn =====
  # config.Infer(
  #   model="Wisppn", fusion="none",
  #   weightsFile=r"out/train/model/Wisppn/0/epoch-19/weights.pt",
  #   outDir="infer/wisppn-desk-chair",
  # ),
  # config.Infer(
  #   model="Wisppn", fusion="none",
  #   weightsFile=r"out/train/model/Wisppn/1/epoch-19/weights.pt",
  #   outDir="infer/wisppn-obstacle",
  # ),
]

# for i in range(2, 3):
#   model = "Wpformer"
#   trianConfList.append(
#     config.Train(
#
#       outDir=f"train/model/{model}/{i}",
#       nTimestep=9,
#       fusion="none",
#       model=model,
#       batchSize=24,
#     )
#   )

# for i in range(3, 4):
#   model = "Evtformer"
#   trianConfList.append(
#     config.Train(
#       outDir=f"train/model/{model}/{i}",
#       nTimestep=9,
#       fusion="none",
#       model=model,
#       batchSize=24,
#     )
#   )

# for i in range(2, 3):
#   model = "Wisppn"
#   trianConfList.append(
#     config.Train(
#       outDir=f"train/model/{model}/{i}",
#       nTimestep=9,
#       fusion="none",
#       model=model,
#       batchSize=24,
#     )
#   )

for i in range(21, 22):
  model = "Evtformer"
  trianConfList.append(
    config.Train(
      outDir=f"train/model/{model}/{i}",
      nTimestep=9,
      fusion="none",
      model=model,
      batchSize=24,
    )
  )

# Historical Evtformer ablations. Enable one block and select its matching
# Trainer below when a controlled v8/B1 rerun is needed.
# for i in range(0, 1):
#   model = "EvtformerV8"
#   trianConfList.append(
#     config.Train(
#       outDir=f"train/model/{model}/{i}",
#       nTimestep=9,
#       fusion="none",
#       model=model,
#       batchSize=24,
#     )
#   )

# for i in range(0, 1):
#   model = "EvtformerB1"
#   trianConfList.append(
#     config.Train(
#       outDir=f"train/model/{model}/{i}",
#       nTimestep=9,
#       fusion="none",
#       model=model,
#       batchSize=24,
#     )
#   )


# for i in range(2, 3):
#   model = "Wpnet"
#   trianConfList.append(
#     config.Train(
#       outDir=f"train/model/{model}/{i}",
#       fusion="none",
#       model=model,
#       batchSize=24,
#     )
#   )


# for subcarrier, batch in ((30, 200),):
#   for i in range(10):
#     trianConfList.append(
#       config.Train(
#         outDir=f"train/input/{subcarrier}.phase/{i}",
#         nSubcarrier=subcarrier,
#         batchSize=batch,
#         csiSubDir="phases",
#       )
#     )
# for subcarrier, batch in ((30, 200),):
#   for i in range(2):
#     trianConfList.append(
#       config.Train(
#         outDir=f"train/input/{subcarrier}.phase.2.4g/{i}",
#         datasetDir=r"/root/docker/data/CSI with frames (2023-11)/parsed/free-small-2.4g",
#         nSubcarrier=subcarrier,
#         batchSize=batch,
#         csiSubDir="phases",
#       )
#     )

# for subcarrier, batch in ((1, 210),):
#   for i in range(8):
#     trianConfList.append(
#       config.Train(
#         outDir=f"train/input/{subcarrier}.phase/{i}",
#         nSubcarrier=subcarrier,
#         batchSize=batch,
#         csiSubDir="phases",
#       )
#     )
# for subcarrier, batch in ((1, 210),):
#   for i in range(7, 10):
#     trianConfList.append(
#       config.Train(
#         outDir=f"train/input/{subcarrier}.phase.2.4g/{i}",
#         datasetDir=r"/root/docker/data/CSI with frames (2023-11)/parsed/free-small-2.4g",
#         nSubcarrier=subcarrier,
#         batchSize=batch,
#         csiSubDir="phases",
#       )
#     )


if __name__ == "__main__":
  cliArgs = parsecli.parseCli()

  # Trainer = train.Trainer
  # Trainer = train.TrainerWpformer
  # Trainer = train.TrainerWpnet
  Trainer = train.TrainerEvtformer
  # Trainer = train.TrainerEvtformerV8
  # Trainer = train.TrainerEvtformerB1

  # if cliArgs.val:
  #   for conf in tqdm(valConfList, desc="Conf"):
  #     Trainer(conf).inference(True)
  # elif cliArgs.inference:
  #   for conf in tqdm(inferConfList, desc="Conf"):
  #     Trainer(conf).inference()
  # elif cliArgs.piw:
  #   for conf in tqdm(trianConfList, desc="Conf"):
  #     assert conf.model == "Piw"
  #     Trainer(conf).train()
  # else:
  for conf in tqdm(trianConfList, desc="Conf"):
      Trainer(conf).train()
  # for conf in tqdm(inferConfList, desc="Conf"):
  #     Trainer(conf).inference()
