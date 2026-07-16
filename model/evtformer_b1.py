"""Historical Evtformer B1: stabilized DualToken and one bounded EVT block."""

from model.evtformer_historical import HistoricalEvtformer


class EvtformerB1(HistoricalEvtformer):
  def __init__(self, nInChannel: int, nOutChannel: int):
    super().__init__(
      nInChannel,
      nOutChannel,
      num_evt_layers=1,
      stabilize_evt=True,
      stabilize_dual_token=True,
    )
