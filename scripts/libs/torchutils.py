import torch
import numpy as np


def toNp(inTensor: torch.Tensor) -> np.ndarray:
  return inTensor.detach().cpu().numpy()
