"""Parameterized RT-DETR baseline training entrypoint."""

import argparse
import os

from ultralytics import RTDETR


ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(ROOT, "dataset", "underwater", "data_rgb.yaml")


def parse_args():
    parser = argparse.ArgumentParser(description="Train RT-DETR baselines on underwater datasets.")
    parser.add_argument("--model", default="rtdetr-l.yaml")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="run")
    parser.add_argument("--name", default="rtdetr-l-rgb")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cache", default=False)
    parser.add_argument("--exist-ok", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = RTDETR(args.model)
    model.train(
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
        resume=args.resume,
        project=args.project,
        name=args.name,
    )
