#!/usr/bin/env python3
"""Exported metric methods reused from this Ultralytics project.

This module keeps the same metric definitions used in trainer logs:
1) Params(M)
2) GFLOPs(total[default])
3) Avg FPS(all epochs)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from ultralytics import YOLO
from ultralytics.nn.mm.complexity import (
    build_default_complexity_summary,
    compute_default_multimodal_complexity_report,
)
from ultralytics.utils.torch_utils import de_parallel, get_flops


def _normalize_imgsz(imgsz: int | list[int] | tuple[int, ...]) -> int:
    """Return a single int image size from trainer-like imgsz configs."""
    if isinstance(imgsz, (list, tuple)):
        if not imgsz:
            return 640
        return int(imgsz[0])
    return int(imgsz)


def params_m(base_model) -> float:
    """Same Params(M) definition as trainer summary."""
    try:
        return sum(p.numel() for p in base_model.parameters()) / 1e6
    except Exception:
        return 0.0


def gflops_total_default(base_model, imgsz: int | list[int] | tuple[int, ...] = 640) -> float:
    """Same GFLOPs(total[default]) priority path as trainer summary."""
    size = _normalize_imgsz(imgsz)

    # Priority: default multimodal complexity report (same as training logs).
    try:
        report = compute_default_multimodal_complexity_report(base_model, imgsz=size)
        summary = build_default_complexity_summary(base_model, report)
        return float(summary.get("gflops_total", 0.0))
    except Exception:
        pass

    # Fallback: generic thop path.
    try:
        return float(get_flops(base_model, size))
    except Exception:
        return 0.0


def epoch_fps_values(epoch_times: Iterable[float], dataset_size: int) -> list[float]:
    """Mirror trainer epoch FPS: len(dataset) / epoch_time (seconds)."""
    out = []
    for t in epoch_times:
        try:
            t = float(t)
        except Exception:
            continue
        if t <= 0:
            continue
        fps = float(dataset_size) / t
        if math.isfinite(fps) and fps > 0:
            out.append(fps)
    return out


def avg_epoch_fps(epoch_times: Iterable[float], dataset_size: int) -> float:
    """Average of all valid epoch FPS values."""
    vals = epoch_fps_values(epoch_times, dataset_size)
    return float(np.mean(vals)) if vals else 0.0


def read_epoch_times_txt(path: str | Path) -> list[float]:
    """Read epoch time (seconds) text: one value per line."""
    p = Path(path)
    if not p.exists():
        return []
    vals = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            vals.append(float(line))
        except ValueError:
            continue
    return vals


def summarize_from_weights(
    weights: str | Path,
    imgsz: int | list[int] | tuple[int, ...] = 640,
    epoch_times: Iterable[float] | None = None,
    dataset_size: int | None = None,
) -> dict:
    """Build summary compatible with trainer final print."""
    model = YOLO(str(weights))
    base_model = de_parallel(model.model)

    summary = {
        "Params(M)": round(params_m(base_model), 3),
        "GFLOPs(total[default])": round(gflops_total_default(base_model, imgsz), 3),
    }

    if epoch_times is not None and dataset_size is not None:
        summary["Avg FPS(all epochs)"] = round(avg_epoch_fps(epoch_times, int(dataset_size)), 3)

    return summary


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export Params/GFLOPs/FPS with this project's metric definitions.")
    p.add_argument("--weights", type=str, required=True, help="Path to best.pt/last.pt")
    p.add_argument("--imgsz", type=int, default=640, help="Image size used for GFLOPs")
    p.add_argument("--dataset-size", type=int, default=None, help="len(train_dataset)")
    p.add_argument(
        "--epoch-times-txt",
        type=str,
        default=None,
        help="Text file with epoch_time(sec), one value per line. Optional for Avg FPS.",
    )
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    epoch_times = None
    if args.epoch_times_txt:
        epoch_times = read_epoch_times_txt(args.epoch_times_txt)

    summary = summarize_from_weights(
        weights=args.weights,
        imgsz=args.imgsz,
        epoch_times=epoch_times,
        dataset_size=args.dataset_size,
    )

    # Keep output style aligned with trainer summary field names.
    keys = ["Params(M)", "GFLOPs(total[default])", "Avg FPS(all epochs)"]
    line = ", ".join(f"{k}={summary[k]:.3f}" for k in keys if k in summary)
    print(line)


if __name__ == "__main__":
    main()

