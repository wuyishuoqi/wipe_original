class COCO:
  # https://mmpose.readthedocs.io/zh-cn/latest/dataset_zoo/2d_body_keypoint.html#coco
  skeletons: dict[int, tuple] = {
    0: (1, 2),
    1: (2, 3),
    2: (4,),
    3: (5,),
    4: (6,),
    5: (6, 7, 11),
    6: (8, 12),
    7: (9,),
    8: (10,),
    9: (),
    10: (),
    11: (12, 13),
    12: (14,),
    13: (15,),
    14: (16,),
    15: (),
    16: (),
  }


class CrowdPose:
  # https://mmpose.readthedocs.io/zh_CN/latest/dataset_zoo/2d_body_keypoint.html#crowdpose
  skeletons: dict[int, tuple] = {
    0: (2, 6, 13),
    1: (3, 7, 13),
    2: (4,),
    3: (5,),
    4: (),
    5: (),
    6: (7, 8),
    7: (9,),
    8: (10,),
    9: (11,),
    10: (),
    11: (),
    12: (13,),
    13: (),
  }
