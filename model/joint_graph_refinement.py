"""Directed COCO joint-token graph refinement."""

import torch
import torch.nn as nn


class JointTokenGraphRefinement(nn.Module):
  """Propagate parent/child context and return a bounded feature residual."""

  # Directed parent-to-child edges in the standard 17-joint COCO order.
  _EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
  )

  def __init__(
    self,
    channels: int = 128,
    n_joints: int = 17,
    residual_gate_init: float = -3.0,
    residual_gate_max: float = 0.05,
  ):
    super().__init__()
    if n_joints != 17:
      raise ValueError("the COCO graph requires 17 joint tokens")

    self.channels = channels
    self.n_joints = n_joints
    self.residual_gate_max = residual_gate_max

    inward = torch.zeros(n_joints, n_joints)
    outward = torch.zeros(n_joints, n_joints)
    for parent, child in self._EDGES:
      inward[child, parent] = 1.0
      outward[parent, child] = 1.0
    self.register_buffer("inward_adjacency", self._row_normalize(inward))
    self.register_buffer("outward_adjacency", self._row_normalize(outward))

    self.input_norm = nn.LayerNorm(channels)
    self.self_projection = nn.Linear(channels, channels, bias=False)
    self.parent_projection = nn.Linear(channels, channels, bias=False)
    self.child_projection = nn.Linear(channels, channels, bias=False)
    self.activation = nn.GELU()
    self.output_projection = nn.Linear(channels, channels, bias=False)
    self.residual_gate_logit = nn.Parameter(torch.tensor(residual_gate_init))

    self._reset_parameters()

  @staticmethod
  def _row_normalize(adjacency: torch.Tensor) -> torch.Tensor:
    degree = adjacency.sum(dim=1, keepdim=True).clamp_min(1.0)
    return adjacency / degree

  def _reset_parameters(self):
    nn.init.xavier_uniform_(self.self_projection.weight)
    nn.init.xavier_uniform_(self.parent_projection.weight)
    nn.init.xavier_uniform_(self.child_projection.weight)
    nn.init.zeros_(self.output_projection.weight)

  def effective_gate(self) -> torch.Tensor:
    return self.residual_gate_max * torch.sigmoid(
      self.residual_gate_logit
    )

  def _validate_tokens(self, tokens: torch.Tensor):
    if tokens.ndim != 3:
      raise ValueError(f"expected joint tokens [B, 17, C], got {tokens.shape}")
    if tokens.shape[1:] != (self.n_joints, self.channels):
      raise ValueError(
        f"expected joint tokens [B, {self.n_joints}, {self.channels}], "
        f"got {tokens.shape}"
      )

  def raw_residual(self, tokens: torch.Tensor) -> torch.Tensor:
    """Return the unscaled directed-graph residual."""
    self._validate_tokens(tokens)
    tokens = self.input_norm(tokens)
    parent_context = torch.einsum(
      "ij,bjc->bic", self.inward_adjacency, tokens
    )
    child_context = torch.einsum(
      "ij,bjc->bic", self.outward_adjacency, tokens
    )
    hidden = (
      self.self_projection(tokens)
      + self.parent_projection(parent_context)
      + self.child_projection(child_context)
    )
    hidden = self.activation(hidden)
    return self.output_projection(hidden)

  def forward(self, tokens: torch.Tensor) -> torch.Tensor:
    """Return graph residuals for tokens shaped ``[B, 17, C]``."""
    return self.effective_gate() * self.raw_residual(tokens)


class RmsNormalizedJointTokenGraphRefinement(JointTokenGraphRefinement):
  """Hard-limit every joint's graph residual to a token RMS ratio."""

  def __init__(
    self,
    channels: int = 128,
    n_joints: int = 17,
    residual_gate_init: float = -2.0,
    residual_gate_max: float = 0.005,
    eps: float = 1e-6,
  ):
    super().__init__(
      channels=channels,
      n_joints=n_joints,
      residual_gate_init=residual_gate_init,
      residual_gate_max=residual_gate_max,
    )
    self.eps = eps

  def forward(self, tokens: torch.Tensor) -> torch.Tensor:
    raw_residual = self.raw_residual(tokens)
    residual_rms = raw_residual.square().mean(
      dim=2, keepdim=True
    ).add(self.eps).sqrt()
    token_rms = tokens.detach().square().mean(
      dim=2, keepdim=True
    ).add(self.eps).sqrt()
    normalized_residual = raw_residual / residual_rms
    return self.effective_gate() * token_rms * normalized_residual
