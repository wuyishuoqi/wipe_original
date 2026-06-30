openpose2coco = {
  0: 0,
  1: -1,
  2: 6,
  3: 8,
  4: 10,
  5: 5,
  6: 7,
  7: 9,
  8: 12,
  9: 14,
  10: 16,
  11: 11,
  12: 13,
  13: 15,
  14: 2,
  15: 1,
  16: 4,
  17: 3,
}

coco2openpose = {}
for k, v in openpose2coco.items():
  if v >= 0:
    coco2openpose[v] = k

cocoJointsName = {
  0: "Nose",
  1: "L.Eye",
  2: "R.Eye",
  3: "L.Ear",
  4: "R.Ear",
  5: "L.Shoulder",
  6: "R.Shoulder",
  7: "L.Elbow",
  8: "R.Elbow",
  9: "L.Wrist",
  10: "R.Wrist",
  11: "L.Hip",
  12: "R.Hip",
  13: "L.Knee",
  14: "R.Knee",
  15: "L.Heel",
  16: "R.Heel",
}

legend = {
  "norm1": "WiPE (Ours)",
  "woreg": "w/o L1 Regularization",
  "norm2": "w/   L2 Regularization",
  "womask": "- $L_{hsm}$",
  "womw": "w/o Matthew Weight",
}
