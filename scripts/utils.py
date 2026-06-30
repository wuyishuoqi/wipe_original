from pathlib import Path
from queue import Queue
import os
import sys


def dictAppend(aDict: dict, bDict: dict):
  for k, v in bDict.items():
    if k in aDict:
      aDict[k].append(v)


def checkExt(path: Path, ext: str) -> Path:
  if path.suffix != ext:
    out = path.parent / (path.name + ext)
  else:
    out = path
  return out


def funFromQ(fun, q: Queue, pbar=None):
  fun(**q.get())
  q.task_done()
  if pbar:
    pbar.update(1)


def clearFutureSet(futureSet: set):
  from concurrent.futures import wait

  completedFutureSet = wait(futureSet)
  for doneFuture in completedFutureSet[0]:
    doneResult = doneFuture.result()
    if doneResult:
      print(doneResult)
  for notDoneFuture in completedFutureSet[1]:
    notDoneResult = notDoneFuture.result()
    if notDoneResult:
      print(notDoneResult)
  futureSet.clear()


class HideStdout:
  def __enter__(self):
    self.__originalStdout = sys.stdout
    sys.stdout = open(os.devnull, "w")

  def __exit__(self, exc_type, exc_val, exc_tb):
    sys.stdout.close()
    sys.stdout = self.__originalStdout
