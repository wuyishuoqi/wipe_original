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
  # Add an active inference configuration here when inference is needed.

  # ===== Wisppn =====
  # config.Infer(
  #   model="Wisppn", fusion="none",
  #   weightsFile=r"out/train/model/Wisppn/0/epoch-19/weights.pt",
  #   outDir="infer/wisppn-desk-chair",
  # ),
  # config.Infer(
  #   model="Wisppn", fusion="none",
  #   weightsFile=r"out/train/model/Wisppn/2/epoch-19/weights.pt",
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

# for i in range(21, 22):
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

# for i in range(1):
#   model = "EvtformerNoDual"
#   trianConfList.append(
#     config.Train(
#       outDir=f"train/model/{model}/{i}",
#       nTimestep=9,
#       fusion="none",
#       model=model,
#       batchSize=24,
#     )
#   )

# for i in range(1, 2):
#   model = "EvtformerNoSonnet"
#   trianConfList.append(
#     config.Train(
#       outDir=f"train/model/{model}/{i}",
#       datasetDir=r"/home/teacher2/文档/noostruct-dataset/",
#       nTimestep=9,
#       fusion="none",
#       model=model,
#       batchSize=24,
#     )
#   )

# for i in range(1, 2):
#   model = "EvtformerTwo"
#   trianConfList.append(
#     config.Train(
#       outDir=f"train/model/{model}/{i}",
#       datasetDir=r"/home/teacher2/文档/noostruct-dataset/",
#       nTimestep=9,
#       fusion="none",
#       model=model,
#       batchSize=24,
#     )
#   )

# for i in range(1, 2):
#   model = "EvtformerThree"
#   trianConfList.append(
#     config.Train(
#       outDir=f"train/model/{model}/{i}",
#       datasetDir=r"/home/teacher2/文档/noostruct-dataset/",
#       nTimestep=9,
#       fusion="none",
#       model=model,
#       batchSize=24,
#     )
#   )

# for i in range(1, 2):
#   model = "EvtformerFour"
#   trianConfList.append(
#     config.Train(
#       outDir=f"train/model/{model}/{i}",
#       datasetDir=r"/home/teacher2/文档/noostruct-dataset/",
#       nTimestep=9,
#       fusion="none",
#       model=model,
#       batchSize=24,
#     )
#   )

# for i in range(1, 2):
#   model = "ThreeTestOne"
#   trianConfList.append(
#     config.Train(
#       outDir=f"train/model/{model}/{i}",
#       datasetDir=r"/home/teacher2/文档/noostruct-dataset/",
#       nTimestep=9,
#       fusion="none",
#       model=model,
#       batchSize=24,
#     )
#   )

# for i in range(1, 2):
#   model = "EvtformerFive"
#   trianConfList.append(
#     config.Train(
#       outDir=f"train/model/{model}/{i}",
#       datasetDir=r"/home/teacher2/文档/noostruct-dataset/",
#       nTimestep=9,
#       fusion="none",
#       model=model,
#       batchSize=24,
#     )
#   )

# for i in range(1, 2):
#   model = "EvtformerSix"
#   trianConfList.append(
#     config.Train(
#       outDir=f"train/model/{model}/{i}",
#       datasetDir=r"/home/teacher2/文档/noostruct-dataset/",
#       nTimestep=9,
#       fusion="none",
#       model=model,
#       batchSize=24,
#     )
#   )

# Hreformer V1 on Noostruct, run 1.
# for i in range(1, 2):
#   model = "Hreformer"
#   trianConfList.append(
#     config.Train(
#       outDir=f"train/model/{model}/{i}",
#       datasetDir=r"/home/teacher2/文档/noostruct-dataset/",
#       nTimestep=9,
#       fusion="none",
#       model=model,
#       batchSize=24,
#     )
#   )

# HreformerV2 on Noostruct, run 1.
# for i in range(1, 2):
#   model = "HreformerV2"
#   trianConfList.append(
#     config.Train(
#       outDir=f"train/model/{model}/{i}",
#       datasetDir=r"/home/teacher2/文档/noostruct-dataset/",
#       nTimestep=9,
#       fusion="none",
#       model=model,
#       batchSize=24,
#     )
#   )

# HreformerV3 on Noostruct, run 1.
# for i in range(1, 2):
#   model = "HreformerV3"
#   trianConfList.append(
#     config.Train(
#       outDir=f"train/model/{model}/{i}",
#       datasetDir=r"/home/teacher2/文档/noostruct-dataset/",
#       nTimestep=9,
#       fusion="none",
#       model=model,
#       batchSize=24,
#     )
#   )

# HreformerV4 on Noostruct, run 1.
# for i in range(1, 2):
#   model = "HreformerV4"
#   trianConfList.append(
#     config.Train(
#       outDir=f"train/model/{model}/{i}",
#       datasetDir=r"/home/teacher2/文档/noostruct-dataset/",
#       nTimestep=9,
#       fusion="none",
#       model=model,
#       batchSize=24,
#     )
#   )

# HreformerV5 on Noostruct, run 1.
# for i in range(1, 2):
#   model = "HreformerV5"
#   trianConfList.append(
#     config.Train(
#       outDir=f"train/model/{model}/{i}",
#       datasetDir=r"/home/teacher2/文档/noostruct-dataset/",
#       nTimestep=9,
#       fusion="none",
#       model=model,
#       batchSize=24,
#     )
#   )

# HreformerV6 on Noostruct, run 1.
# for i in range(1, 2):
#   model = "HreformerV6"
#   trianConfList.append(
#     config.Train(
#       outDir=f"train/model/{model}/{i}",
#       datasetDir=r"/home/teacher2/文档/noostruct-dataset/",
#       nTimestep=9,
#       fusion="none",
#       model=model,
#       batchSize=24,
#     )
#   )

# Wpformer on Desk, run 3.
for i in range(3, 4):
  model = "Wpformer"
  trianConfList.append(
    config.Train(
      outDir=f"train/model/{model}/{i}",
      datasetDir=r"/home/teacher2/文档/desk-chair-350-parse-preprocess/",
      nTimestep=9,
      fusion="none",
      model=model,
      batchSize=24,
    )
  )

# for i in range(1):
#   model = "EvtformerNoEVT"
#   trianConfList.append(
#     config.Train(
#       outDir=f"train/model/{model}/{i}",
#       nTimestep=9,
#       fusion="none",
#       model=model,
#       batchSize=24,
#     )
#   )

# for i in range(1):
#   model = "EvtSsd"
#   trianConfList.append(
#     config.Train(
#       outDir=f"train/model/{model}/{i}",
#       nTimestep=15,
#       fusion="none",
#       model=model,
#       batchSize=24,
#     )
#   )

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
  Trainer = train.TrainerWpformer
  # Trainer = train.TrainerWpnet
  # Trainer = train.TrainerEvtformer
  # Trainer = train.TrainerEvtformerNoDual
  # Trainer = train.TrainerEvtformerNoSonnet
  # Trainer = train.TrainerEvtformerNoEVT
  # Trainer = train.TrainerEvtSsd
  # Trainer = train.TrainerEvtformerTwo
  # Trainer = train.TrainerEvtformerThree
  # Trainer = train.TrainerEvtformerFour
  # Trainer = train.TrainerThreeTestOne
  # Trainer = train.TrainerEvtformerFive
  # Trainer = train.TrainerEvtformerSix
  # Trainer = train.TrainerHreformer
  # Trainer = train.TrainerHreformerV2
  # Trainer = train.TrainerHreformerV3
  # Trainer = train.TrainerHreformerV4
  # Trainer = train.TrainerHreformerV5
  # Trainer = train.TrainerHreformerV6
  # Trainer = train.TrainerHreformerV7
  # Trainer = train.TrainerHreformerV8
  # Trainer = train.TrainerHreformerV9
  # Trainer = train.TrainerHreformerV10
  # Trainer = train.TrainerHreformerV10NoDual
  # Trainer = train.TrainerHreformerV10NoRAF
  # Trainer = train.TrainerHreformerV10NoGraph
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
