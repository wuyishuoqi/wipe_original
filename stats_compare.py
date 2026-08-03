#!/usr/bin/env python3
"""Fair comparison of HreformerV10, WPFormer, and WiSPPN.

This script reads the cumulative validation records saved at epoch 19. It
normalizes every model to the common 17 COCO joints, removes the extra
first-batch validation point, weights validation batches by sample count, and
selects exactly one checkpoint per model and dataset.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/wipe-matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
TRAIN_ROOT = ROOT / "out" / "train" / "model"
OUT_ROOT = ROOT / "out" / "stats" / "model_comparison"

THRESHOLDS = (0.05, 0.10, 0.20, 0.30)
SELECTION_WEIGHTS = {0.05: 0.25, 0.10: 0.15, 0.20: 0.50, 0.30: 0.10}

JOINT_NAMES = (
  "Nose",
  "L.Eye",
  "R.Eye",
  "L.Ear",
  "R.Ear",
  "L.Shoulder",
  "R.Shoulder",
  "L.Elbow",
  "R.Elbow",
  "L.Wrist",
  "R.Wrist",
  "L.Hip",
  "R.Hip",
  "L.Knee",
  "R.Knee",
  "L.Ankle",
  "R.Ankle",
)

MODEL_COLORS = {
  "Hreformer (Ours)": "#c62828",
  "WPFormer": "#1565c0",
  "WiSPPN": "#2e7d32",
}


@dataclass(frozen=True)
class ExperimentSpec:
  dataset: str
  model: str
  run_dir: Path


@dataclass
class ExperimentResult:
  spec: ExperimentSpec
  source_epoch_dir: Path
  val_indices: np.ndarray
  batch_size: int
  # Threshold -> [Epoch, Mean + 17 joints]
  metrics: dict[float, np.ndarray]
  selection_score: np.ndarray
  selected_epoch: int


EXPERIMENTS = {
  "Desk": (
    ExperimentSpec("Desk", "Hreformer (Ours)", TRAIN_ROOT / "HreformerV10" / "0"),
    ExperimentSpec("Desk", "WPFormer", TRAIN_ROOT / "Wpformer" / "0"),
    ExperimentSpec("Desk", "WiSPPN", TRAIN_ROOT / "Wisppn" / "0"),
  ),
  "Noostruct": (
    ExperimentSpec(
      "Noostruct", "Hreformer (Ours)", TRAIN_ROOT / "HreformerV10" / "1"
    ),
    ExperimentSpec("Noostruct", "WPFormer", TRAIN_ROOT / "Wpformer" / "1"),
    ExperimentSpec("Noostruct", "WiSPPN", TRAIN_ROOT / "Wisppn" / "1"),
  ),
  "Obstacle": (
    ExperimentSpec(
      "Obstacle", "Hreformer (Ours)", TRAIN_ROOT / "HreformerV10" / "2"
    ),
    ExperimentSpec("Obstacle", "WPFormer", TRAIN_ROOT / "Wpformer" / "2"),
    ExperimentSpec("Obstacle", "WiSPPN", TRAIN_ROOT / "Wisppn" / "2"),
  ),
}


def _epoch_number(path: Path) -> int:
  match = re.fullmatch(r"epoch-(\d+)", path.name)
  if match is None:
    raise ValueError(f"invalid epoch directory: {path}")
  return int(match.group(1))


def _latest_complete_epoch(run_dir: Path) -> Path:
  candidates = [
    path
    for path in run_dir.glob("epoch-*")
    if (path / "val.pck.pkl").is_file()
  ]
  if not candidates:
    raise FileNotFoundError(f"no complete epoch found in {run_dir}")
  return max(candidates, key=_epoch_number)


def _batch_weights(n_samples: int, n_batches: int) -> tuple[int, np.ndarray]:
  batch_size = math.ceil(n_samples / n_batches)
  if math.ceil(n_samples / batch_size) != n_batches:
    raise ValueError(
      f"cannot infer batch size from {n_samples} samples and {n_batches} batches"
    )
  weights = np.full(n_batches, batch_size, dtype=np.float64)
  weights[-1] = n_samples - batch_size * (n_batches - 1)
  if weights[-1] <= 0:
    raise ValueError("invalid final validation batch size")
  return batch_size, weights


def _load_experiment(spec: ExperimentSpec) -> ExperimentResult:
  val_index_path = spec.run_dir / "valset.idx.npy"
  epoch_index_path = spec.run_dir / "epoch.idx.npy"
  if not val_index_path.is_file() or not epoch_index_path.is_file():
    raise FileNotFoundError(f"missing split or epoch index in {spec.run_dir}")

  val_indices = np.load(val_index_path)
  n_epochs = len(np.load(epoch_index_path))
  source_epoch_dir = _latest_complete_epoch(spec.run_dir)
  with open(source_epoch_dir / "val.pck.pkl", "rb") as file:
    raw_pck = pickle.load(file)

  metrics: dict[float, np.ndarray] = {}
  batch_size = 0
  expected_batches = None
  for threshold in THRESHOLDS:
    if threshold not in raw_pck:
      raise KeyError(f"PCK@{threshold} missing from {source_epoch_dir}")
    values = np.asarray(raw_pck[threshold], dtype=np.float64)
    if values.ndim != 3 or values.shape[2] < 18:
      raise ValueError(
        f"unexpected PCK shape for {spec.model} on {spec.dataset}: {values.shape}"
      )
    if values.shape[0] != n_epochs + 1:
      raise ValueError(
        f"expected {n_epochs + 1} validation points, got {values.shape[0]} "
        f"for {spec.run_dir}"
      )

    # Index 0 is validation after the first training batch, not a full epoch.
    values = values[1:]
    n_batches = values.shape[1]
    if expected_batches is not None and n_batches != expected_batches:
      raise ValueError(f"batch count changes across thresholds in {spec.run_dir}")
    expected_batches = n_batches
    batch_size, weights = _batch_weights(len(val_indices), n_batches)

    # Ignore the stored mean and WiSPPN's appended neck. Recompute a common
    # mean from the first 17 COCO joints for every model.
    common_joint_batches = values[:, :, 1:18]
    joint_metrics = np.average(
      common_joint_batches, axis=1, weights=weights
    )
    mean_metric = joint_metrics.mean(axis=1, keepdims=True)
    metric = np.concatenate((mean_metric, joint_metrics), axis=1)
    if not np.isfinite(metric).all():
      raise ValueError(f"non-finite PCK values in {source_epoch_dir}")
    metrics[threshold] = metric

  selection_score = sum(
    SELECTION_WEIGHTS[threshold] * metrics[threshold][:, 0]
    for threshold in THRESHOLDS
  )
  selected_epoch = int(np.argmax(selection_score))
  return ExperimentResult(
    spec=spec,
    source_epoch_dir=source_epoch_dir,
    val_indices=val_indices,
    batch_size=batch_size,
    metrics=metrics,
    selection_score=selection_score,
    selected_epoch=selected_epoch,
  )


def _validate_dataset(results: list[ExperimentResult]):
  reference = results[0].val_indices
  for result in results[1:]:
    if not np.array_equal(reference, result.val_indices):
      raise ValueError(
        f"validation split mismatch on {result.spec.dataset}: "
        f"{results[0].spec.model} vs {result.spec.model}"
      )


def _save_master_csv(all_results: dict[str, list[ExperimentResult]]):
  rows = []
  for dataset, results in all_results.items():
    for result in results:
      epoch = result.selected_epoch
      row = {
        "Dataset": dataset,
        "Model": result.spec.model,
        "Selected Epoch": epoch,
        "Selection Score": result.selection_score[epoch],
        "Validation Samples": len(result.val_indices),
        "Inferred Batch Size": result.batch_size,
        "Source": str(result.source_epoch_dir.relative_to(ROOT)),
      }
      for threshold in THRESHOLDS:
        row[f"PCK@{threshold:.2f}"] = result.metrics[threshold][epoch, 0]
      rows.append(row)
  frame = pd.DataFrame(rows)
  frame.to_csv(OUT_ROOT / "master_pck.csv", index=False, float_format="%.9f")
  return frame


def _save_epoch_csv(all_results: dict[str, list[ExperimentResult]]):
  rows = []
  for dataset, results in all_results.items():
    for result in results:
      for epoch in range(len(result.selection_score)):
        row = {
          "Dataset": dataset,
          "Model": result.spec.model,
          "Epoch": epoch,
          "Selection Score": result.selection_score[epoch],
        }
        for threshold in THRESHOLDS:
          row[f"PCK@{threshold:.2f}"] = result.metrics[threshold][epoch, 0]
        rows.append(row)
  pd.DataFrame(rows).to_csv(
    OUT_ROOT / "epoch_metrics.csv", index=False, float_format="%.9f"
  )


def _save_joint_csv(all_results: dict[str, list[ExperimentResult]]):
  rows = []
  for dataset, results in all_results.items():
    for result in results:
      epoch = result.selected_epoch
      for joint_index, joint_name in enumerate(JOINT_NAMES, start=1):
        row = {
          "Dataset": dataset,
          "Model": result.spec.model,
          "Selected Epoch": epoch,
          "Joint Index": joint_index - 1,
          "Joint": joint_name,
        }
        for threshold in THRESHOLDS:
          row[f"PCK@{threshold:.2f}"] = result.metrics[threshold][
            epoch, joint_index
          ]
        rows.append(row)
  pd.DataFrame(rows).to_csv(
    OUT_ROOT / "joint_pck.csv", index=False, float_format="%.9f"
  )


def _save_manifest(all_results: dict[str, list[ExperimentResult]]):
  datasets = {}
  for dataset, results in all_results.items():
    split_hash = hashlib.sha256(results[0].val_indices.tobytes()).hexdigest()
    datasets[dataset] = {
      "validation_samples": len(results[0].val_indices),
      "validation_index_sha256": split_hash,
      "experiments": [
        {
          "model": result.spec.model,
          "run_dir": str(result.spec.run_dir.relative_to(ROOT)),
          "source_epoch_dir": str(result.source_epoch_dir.relative_to(ROOT)),
          "inferred_batch_size": result.batch_size,
          "selected_epoch": result.selected_epoch,
        }
        for result in results
      ],
    }
  manifest = {
    "generated_at": datetime.now().astimezone().isoformat(),
    "metric_scope": "validation",
    "common_joint_protocol": "17 COCO joints; WiSPPN neck excluded",
    "thresholds": THRESHOLDS,
    "selection_weights": {
      f"PCK@{threshold:.2f}": weight
      for threshold, weight in SELECTION_WEIGHTS.items()
    },
    "datasets": datasets,
  }
  with open(OUT_ROOT / "experiment_manifest.json", "w", encoding="utf-8") as file:
    json.dump(manifest, file, ensure_ascii=False, indent=2)


def _format_percent_axis(axis):
  axis.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
  axis.grid(axis="y", alpha=0.3)


def _plot_thresholds(out_dir: Path, dataset: str, results: list[ExperimentResult]):
  figure, axis = plt.subplots(figsize=(8.2, 4.8))
  x = np.arange(len(THRESHOLDS))
  width = 0.24
  offsets = np.arange(len(results)) - (len(results) - 1) / 2
  for offset, result in zip(offsets, results):
    epoch = result.selected_epoch
    values = [result.metrics[t][epoch, 0] for t in THRESHOLDS]
    axis.bar(
      x + offset * width,
      values,
      width,
      label=result.spec.model,
      color=MODEL_COLORS[result.spec.model],
    )
  axis.set_xticks(x, [f"PCK@{threshold:.2f}" for threshold in THRESHOLDS])
  axis.set_ylim(0.0, 1.02)
  axis.set_ylabel("PCK")
  axis.set_title(f"{dataset}: selected-checkpoint PCK")
  axis.legend(loc="lower right")
  _format_percent_axis(axis)
  figure.tight_layout()
  figure.savefig(out_dir / "pck_thresholds.pdf", bbox_inches="tight")
  plt.close(figure)


def _plot_convergence(
  out_dir: Path,
  dataset: str,
  results: list[ExperimentResult],
  threshold: float,
  late_only: bool,
):
  figure, axis = plt.subplots(figsize=(8.2, 4.8))
  for result in results:
    values = result.metrics[threshold][:, 0]
    epochs = np.arange(len(values))
    axis.plot(
      epochs,
      values,
      linewidth=2,
      label=result.spec.model,
      color=MODEL_COLORS[result.spec.model],
    )
    epoch = result.selected_epoch
    axis.scatter(
      [epoch],
      [values[epoch]],
      color=MODEL_COLORS[result.spec.model],
      s=32,
      zorder=3,
    )
  axis.set_xlabel("Epoch")
  axis.set_ylabel(f"PCK@{threshold:.2f}")
  axis.set_title(f"{dataset}: PCK@{threshold:.2f} convergence")
  axis.set_xticks(range(12, 20) if late_only else range(0, 20, 2))
  if late_only:
    axis.set_xlim(12, 19)
    visible = np.concatenate([result.metrics[threshold][12:, 0] for result in results])
    margin = max(0.002, (visible.max() - visible.min()) * 0.12)
    axis.set_ylim(max(0.0, visible.min() - margin), min(1.0, visible.max() + margin))
  else:
    axis.set_xlim(0, 19)
    axis.set_ylim(0.0, 1.02)
  axis.legend(loc="lower right")
  _format_percent_axis(axis)
  figure.tight_layout()
  threshold_name = f"pck{int(threshold * 100):03d}"
  suffix = f"convergence_{threshold_name}{'_late' if late_only else ''}.pdf"
  figure.savefig(out_dir / suffix, bbox_inches="tight")
  plt.close(figure)


def _plot_joints(
  out_dir: Path,
  dataset: str,
  results: list[ExperimentResult],
  threshold: float,
):
  figure, axis = plt.subplots(figsize=(13.5, 5.2))
  x = np.arange(len(JOINT_NAMES))
  width = 0.25
  offsets = np.arange(len(results)) - (len(results) - 1) / 2
  all_values = []
  for offset, result in zip(offsets, results):
    values = result.metrics[threshold][result.selected_epoch, 1:]
    all_values.append(values)
    axis.bar(
      x + offset * width,
      values,
      width,
      label=result.spec.model,
      color=MODEL_COLORS[result.spec.model],
    )
  values = np.concatenate(all_values)
  margin = max(0.01, (values.max() - values.min()) * 0.10)
  axis.set_ylim(max(0.0, values.min() - margin), min(1.0, values.max() + margin))
  axis.set_xticks(x, JOINT_NAMES, rotation=45, ha="right")
  axis.set_ylabel(f"PCK@{threshold:.2f}")
  axis.set_title(f"{dataset}: per-joint PCK@{threshold:.2f}")
  axis.legend(loc="lower left")
  _format_percent_axis(axis)
  figure.tight_layout()
  figure.savefig(out_dir / f"joints_pck{int(threshold * 100):03d}.pdf", bbox_inches="tight")
  plt.close(figure)


def _save_plots(all_results: dict[str, list[ExperimentResult]]):
  for dataset, results in all_results.items():
    out_dir = OUT_ROOT / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_thresholds(out_dir, dataset, results)
    for threshold in (0.05, 0.10, 0.20):
      _plot_convergence(out_dir, dataset, results, threshold, late_only=False)
      _plot_convergence(out_dir, dataset, results, threshold, late_only=True)
    _plot_joints(out_dir, dataset, results, 0.05)
    _plot_joints(out_dir, dataset, results, 0.20)


def main():
  OUT_ROOT.mkdir(parents=True, exist_ok=True)
  all_results: dict[str, list[ExperimentResult]] = {}
  for dataset, specs in EXPERIMENTS.items():
    results = [_load_experiment(spec) for spec in specs]
    _validate_dataset(results)
    all_results[dataset] = results

  _save_master_csv(all_results)
  _save_epoch_csv(all_results)
  _save_joint_csv(all_results)
  _save_manifest(all_results)
  _save_plots(all_results)


if __name__ == "__main__":
  main()
