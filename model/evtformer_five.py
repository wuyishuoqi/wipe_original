"""EvtformerFive: EvtformerThree trained with a PCK-aligned objective."""

from model.evtformer_three import EvtformerThree


class EvtformerFive(EvtformerThree):
  """Keep the validated Three architecture unchanged for loss ablation."""

  pass
