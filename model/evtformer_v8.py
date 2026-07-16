"""Historical Evtformer v8: original DualToken residual and two EVT blocks."""

from model.evtformer_historical import HistoricalEvtformer


class EvtformerV8(HistoricalEvtformer):
  def __init__(self, nInChannel: int, nOutChannel: int):
    super().__init__(
      nInChannel,
      nOutChannel,
      num_evt_layers=2,
      stabilize_evt=False,
      stabilize_dual_token=False,
    )
