"""HreformerV7: V1 with hard RMS-limited joint graph refinement."""

from model.hreformer_v6 import HreformerV6
from model.joint_graph_refinement import (
  RmsNormalizedJointTokenGraphRefinement,
)


class HreformerV7(HreformerV6):
  """Constrain graph refinement to at most 0.5% of each joint token RMS."""

  def __init__(self, nInChannel: int, nOutChannel: int):
    super().__init__(nInChannel, nOutChannel)
    self.joint_graph = RmsNormalizedJointTokenGraphRefinement(
      channels=128,
      n_joints=17,
      residual_gate_init=-2.0,
      residual_gate_max=0.005,
    )
