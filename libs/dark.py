import cv2
import numpy as np


def taylor(hm, coord):
  heatmap_height = hm.shape[0]
  heatmap_width = hm.shape[1]
  px = int(coord[0])
  py = int(coord[1])
  if 1 < px < heatmap_width - 2 and 1 < py < heatmap_height - 2:
    dx = 0.5 * (hm[py][px + 1] - hm[py][px - 1])
    dy = 0.5 * (hm[py + 1][px] - hm[py - 1][px])
    dxx = 0.25 * (hm[py][px + 2] - 2 * hm[py][px] + hm[py][px - 2])
    dxy = 0.25 * (
      hm[py + 1][px + 1] - hm[py - 1][px + 1] - hm[py + 1][px - 1] + hm[py - 1][px - 1]
    )
    dyy = 0.25 * (hm[py + 2 * 1][px] - 2 * hm[py][px] + hm[py - 2 * 1][px])
    derivative = np.matrix([[dx], [dy]])
    hessian = np.matrix([[dxx, dxy], [dxy, dyy]])
    if dxx * dyy - dxy**2 != 0:
      hessianinv = hessian.I
      offset = -hessianinv * derivative
      offset = np.squeeze(np.array(offset.T), axis=0)
      coord += offset
  return coord


def gaussian_blur(hm, kernel):
  border = (kernel - 1) // 2
  batch_size = hm.shape[0]
  num_joints = hm.shape[1]
  height = hm.shape[2]
  width = hm.shape[3]
  for i in range(batch_size):
    for j in range(num_joints):
      origin_max = np.max(hm[i, j])
      dr = np.zeros((height + 2 * border, width + 2 * border))
      dr[border:-border, border:-border] = hm[i, j].copy()
      dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
      hm[i, j] = dr[border:-border, border:-border].copy()
      hm[i, j] *= origin_max / np.max(hm[i, j])
  return hm
