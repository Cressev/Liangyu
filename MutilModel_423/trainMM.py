"""Baseline YOLOMM training entrypoint.

Defaults reproduce the previous YOLO11-small RGB run, while command-line
arguments make RGB/sonar baselines reproducible without editing this file.
For parallel single-GPU jobs, set CUDA_VISIBLE_DEVICES in the launch command
and pass --device 0 inside the isolated process.
"""

import argparse
import os

from ultralytics import YOLO

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(ROOT, "dataset", "underwater", "data_rgb.yaml")


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO baselines on underwater datasets.")
    parser.add_argument("--model", default="yolo11.yaml", help="Model YAML or weights path.")
    parser.add_argument("--data", default=DEFAULT_DATA, help="Dataset YAML path.")
    parser.add_argument("--scale", default="s", help="Model scale for YAML models: n/s/m/l/x.")
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="run")
    parser.add_argument("--name", default="yolo11s-rgb")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cache", default=False)
    parser.add_argument("--exist-ok", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    model = YOLO(args.model)
    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        amp=args.amp,
        device=args.device,
        workers=args.workers,
        verbose=args.verbose,
        cache=args.cache,
        exist_ok=args.exist_ok,
        patience=args.patience,
        fraction=args.fraction,
        val=args.val,
        project=args.project,
        name=args.name,
    )
    if args.scale and str(args.model).endswith((".yaml", ".yml")):
        train_kwargs["scale"] = args.scale  # 选择模型 YAML `scales`（n/s/m/l/x）

    model.train(**train_kwargs)
