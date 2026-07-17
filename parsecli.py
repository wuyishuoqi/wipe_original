import argparse


def parseCli():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "-t", "--train", action="store_const", const=True, default=False,
    help="训练模式"
  )
  parser.add_argument(
    "-p", "--piw", action="store_const", const=True, default=False,
    help="Piw 模型训练"
  )
  parser.add_argument(
    "-v", "--val", action="store_const", const=True, default=False,
    help="验证模式"
  )
  parser.add_argument(
    "-i", "--inference", action="store_const", const=True, default=False,
    help="推理模式"
  )
  parser.add_argument(
    "-m", "--model", type=str, default="Wpformer",
    choices=[
      "Unet3x3", "Piw", "Wpnet", "Wpformer", "Wisppn",
      "Evtformer", "EvtformerB1", "EvtformerNoDual",
      "EvtformerNoEVT", "EvtformerNoSonnet",
      "EvtformerV4", "EvtformerV7", "EvtformerV8",
    ],
    help="选择模型架构 (默认: Wpformer)"
  )
  args = parser.parse_args()
  return args
