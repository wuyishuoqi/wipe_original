"""EvtformerSix: EvtformerThree with delayed, gentle PCK alignment."""

from model.evtformer_three import EvtformerThree


class EvtformerSix(EvtformerThree):
  """Keep the validated Three architecture unchanged for loss tuning."""

  pass
