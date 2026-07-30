"""EvtformerTwo: detail-preserving Sonnet and adaptive multi-scale EVT.

This variant targets strict keypoint localization. It keeps the current
DualToken/encoder/decoder route while making Sonnet and EVT refinements
bounded, bypassable, and joint-adaptive.
"""

import math

import torch
import torch.nn as nn

from model.evtformer import Evtformer, _init_weights


def _logit(probability: float) -> float:
  return math.log(probability / (1.0 - probability))


class DetailPreservingSonnet(nn.Module):
  """Apply Sonnet as a calibrated, bounded residual per antenna."""

  def __init__(
    self,
    sonnet: nn.Module,
    antenna_width: int = 4,
    gate_init: float = 0.10,
    gate_max: float = 0.50,
  ):
    super().__init__()
    self.sonnet = sonnet
    self.antenna_width = antenna_width
    self.gate_max = gate_max
    probability = gate_init / gate_max
    self.gate_logits = nn.Parameter(torch.full((3,), _logit(probability)))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    transformed = self.sonnet(x)
    batch, channels, height, width = x.shape
    if width != self.antenna_width * 3:
      raise ValueError(f"Expected three antenna blocks, got feature width {width}")

    raw = x.reshape(batch, channels, height, 3, self.antenna_width)
    delta = (transformed - x).reshape_as(raw)

    # Make the gate represent an actual fraction of the raw feature scale.
    reduce_dims = (1, 2, 4)
    raw_rms = raw.square().mean(reduce_dims, keepdim=True).add(1e-6).sqrt()
    delta_rms = delta.square().mean(reduce_dims, keepdim=True).add(1e-6).sqrt()
    delta = delta * (raw_rms / delta_rms).detach()

    gate = self.gate_max * torch.sigmoid(self.gate_logits)
    out = raw + gate.view(1, 1, 1, 3, 1) * delta
    return out.reshape_as(x)


class MultiScaleEVTAttention(nn.Module):
  """Spatial attention with positive, independently learned head priors."""

  def __init__(
    self,
    channels: int = 128,
    num_heads: int = 4,
    gamma_init: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20),
    gamma_max: float = 0.60,
    dropout: float = 0.05,
  ):
    super().__init__()
    if channels % num_heads:
      raise ValueError("channels must be divisible by num_heads")
    if len(gamma_init) != num_heads:
      raise ValueError("gamma_init must provide one value per attention head")

    self.channels = channels
    self.num_heads = num_heads
    self.head_dim = channels // num_heads
    self.gamma_max = gamma_max
    probabilities = torch.tensor(gamma_init) / gamma_max
    self.gamma_logits = nn.Parameter(torch.logit(probabilities))
    self.gamma_logits.register_hook(lambda grad: grad.clamp(-0.05, 0.05))

    self.qkv = nn.Linear(channels, channels * 3, bias=False)
    self.proj = nn.Linear(channels, channels)
    self.norm1 = nn.LayerNorm(channels)
    self.norm2 = nn.LayerNorm(channels)
    self.mlp = nn.Sequential(
      nn.Linear(channels, channels * 4),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(channels * 4, channels),
      nn.Dropout(dropout),
    )
    self._decay_cache = {}

  def _spatial_distance(self, height, width, device, dtype):
    key = (height, width, device, dtype)
    if key not in self._decay_cache:
      y = torch.arange(height, device=device, dtype=dtype)
      x = torch.arange(width, device=device, dtype=dtype)
      yy, xx = torch.meshgrid(y, x, indexing="ij")
      coordinates = torch.stack((yy.flatten(), xx.flatten()), dim=-1)
      self._decay_cache[key] = torch.cdist(coordinates, coordinates)
    return self._decay_cache[key]

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = x.shape
    positions = height * width
    flat = x.flatten(2).transpose(1, 2)
    normalized = self.norm1(flat)

    qkv = self.qkv(normalized).reshape(
      batch, positions, 3, self.num_heads, self.head_dim
    ).permute(2, 0, 3, 1, 4)
    query, key, value = qkv.unbind(0)
    attention = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)

    distance = self._spatial_distance(height, width, x.device, x.dtype)
    gamma = self.gamma_max * torch.sigmoid(self.gamma_logits)
    attention = attention - gamma.view(1, self.num_heads, 1, 1) * distance
    attention = attention.softmax(dim=-1)

    refined = attention @ value
    refined = refined.transpose(1, 2).reshape(batch, positions, channels)
    refined = flat + self.proj(refined)
    refined = refined + self.mlp(self.norm2(refined))
    return refined.transpose(1, 2).reshape(batch, channels, height, width)


class JointAdaptiveEVT(nn.Module):
  """Strong primary EVT and weak secondary EVT with per-joint gates."""

  def __init__(
    self,
    channels: int = 128,
    num_heads: int = 4,
    n_joints: int = 17,
    primary_gate_init: float = 0.80,
    secondary_gate_init: float = 0.02,
    secondary_gate_max: float = 0.15,
  ):
    super().__init__()
    self.primary = MultiScaleEVTAttention(channels, num_heads)
    self.secondary = MultiScaleEVTAttention(channels, num_heads)
    self.primary_gate_logits = nn.Parameter(
      torch.full((n_joints,), _logit(primary_gate_init))
    )
    self.secondary_gate_max = secondary_gate_max
    probability = secondary_gate_init / secondary_gate_max
    self.secondary_gate_logits = nn.Parameter(
      torch.full((n_joints,), _logit(probability))
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if x.shape[2] != self.primary_gate_logits.numel():
      raise ValueError(f"Expected 17 joint rows, got {x.shape[2]}")

    primary = self.primary(x)
    primary_gate = torch.sigmoid(self.primary_gate_logits).view(1, 1, -1, 1)
    x = x + primary_gate * (primary - x)

    secondary = self.secondary(x)
    secondary_gate = self.secondary_gate_max * torch.sigmoid(
      self.secondary_gate_logits
    ).view(1, 1, -1, 1)
    return x + secondary_gate * (secondary - x)


class EvtformerTwo(Evtformer):
  """Second-generation weak-gated Evtformer for strict PCK thresholds."""

  def __init__(self, nInChannel: int, nOutChannel: int):
    super().__init__(nInChannel, nOutChannel)
    self.sonnet = DetailPreservingSonnet(self.sonnet)
    self.evt = JointAdaptiveEVT(channels=128, num_heads=4, n_joints=17)
    self.evt.apply(_init_weights)
