#!/usr/bin/env python3
import csv
import glob
import json
import re
from pathlib import Path

ROOT = Path("/dpc/yuanxiangqing/projects/detection")
PROJECT = ROOT / "MutilModel_423"
LOG_DIR = ROOT / "training_logs"
RUN_ROOT = PROJECT / "run_yolo_bs64"
SIZE_MAP_DIR = ROOT / "reports" / "size_maps"

YOLO_ROWS = [
    ("YOLOv5s", "RGB", "yolov5s-rgb-bs64"),
    ("YOLOv5s", "Sonar", "yolov5s-sonar-bs64"),
    ("YOLOv8s", "RGB", "yolov8s-rgb-bs64"),
    ("YOLOv8s", "Sonar", "yolov8s-sonar-bs64"),
    ("YOLOv9s", "RGB", "yolov9s-rgb-bs64"),
    ("YOLOv9s", "Sonar", "yolov9s-sonar-bs64"),
    ("YOLOv10s", "RGB", "yolov10s-rgb-bs64"),
    ("YOLOv10s", "Sonar", "yolov10s-sonar-bs64"),
    ("YOLOv11s", "RGB", "yolov11s-rgb-bs64"),
    ("YOLOv11s", "Sonar", "yolov11s-sonar-bs64"),
    ("YOLOv12s", "RGB", "yolov12s-rgb-bs64"),
    ("YOLOv12s", "Sonar", "yolov12s-sonar-bs64"),
    ("YOLOv26s", "RGB", "yolov26s-rgb-bs64"),
    ("YOLOv26s", "Sonar", "yolov26s-sonar-bs64"),
]

NON_YOLO = ["RT-DETR", "SSD", "fast RCNN"]


def last_csv_row(path):
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def pick(row, *names):
    if not row:
        return ""
    for name in names:
        if name in row and row[name] != "":
            return row[name]
    return ""


def latest_train_log(group):
    logs = [
        p
        for p in glob.glob(str(LOG_DIR / f"{group}_*.log"))
        if "_smoke_" not in Path(p).name and "_size_maps" not in Path(p).name
    ]
    return Path(sorted(logs)[-1]) if logs else None


def parse_efficiency(group):
    log = latest_train_log(group)
    if not log or not log.exists():
        return "", "", ""
    text = log.read_text(errors="ignore")
    matches = re.findall(
        r"Params:\s*([0-9.]+)M \(([0-9,]+)\) \| GFLOPs\(total\[default\]\):\s*([0-9.]+)(?: \| Avg FPS\(all epochs\):\s*([0-9.]+))?",
        text,
    )
    if not matches:
        return "", "", ""
    params_m, _, gflops, fps = matches[-1]
    return params_m, gflops, fps or ""


def fmt_metric(value):
    if value in ("", None):
        return ""
    try:
        return f"{float(value):.5f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(value)


def load_size_maps(group):
    path = SIZE_MAP_DIR / f"{group}.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: fmt_metric(v) for k, v in data.items() if k != "source"}


def status_note(group, row, has_size_maps):
    out = RUN_ROOT / group
    notes = []
    if not row:
        notes.append("no results.csv")
    else:
        epoch = pick(row, "epoch")
        if epoch and epoch != "350":
            notes.append(f"last_epoch={epoch}")
    if has_size_maps:
        notes.append("AP metrics from uploaded zip COCOevalBBoxMM on best.pt")
    else:
        notes.append("mapS/mapM/mapL missing")
    if not (out / "weights" / "best.pt").exists():
        notes.append("best.pt missing")
    return "; ".join(notes)


def main():
    header = ["Model", "Modality", "map50", "map75", "map50-75", "mapS", "mapM", "mapL", "Params", "GFLOPS", "FPS", "备注"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join([":---"] * len(header)) + "|"]
    last_model = None
    for model, modality, group in YOLO_ROWS:
        row = last_csv_row(RUN_ROOT / group / "results.csv")
        params, gflops, fps = parse_efficiency(group)
        size_maps = load_size_maps(group)
        map_s, map_m, map_l = size_maps.get("mapS", ""), size_maps.get("mapM", ""), size_maps.get("mapL", "")
        if not map_s:
            map_s = pick(row, "metrics/mAP50-95(S)", "metrics/mAP(S)", "mapS")
        if not map_m:
            map_m = pick(row, "metrics/mAP50-95(M)", "metrics/mAP(M)", "mapM")
        if not map_l:
            map_l = pick(row, "metrics/mAP50-95(L)", "metrics/mAP(L)", "mapL")
        map50 = size_maps.get("AP50") or fmt_metric(pick(row, "metrics/mAP50(B)", "mAP50", "map50"))
        map75 = size_maps.get("AP75") or fmt_metric(pick(row, "metrics/mAP75(B)", "mAP75", "map75"))
        map5095 = size_maps.get("AP") or fmt_metric(pick(row, "metrics/mAP50-95(B)", "mAP50-95", "map50-95", "map50_95"))
        vals = [
            model if model != last_model else "",
            modality,
            map50,
            map75,
            map5095,
            map_s,
            map_m,
            map_l,
            params,
            gflops,
            fps,
            status_note(group, row, bool(map_s and map_m and map_l)),
        ]
        lines.append("| " + " | ".join(vals) + " |")
        last_model = model
    for model in NON_YOLO:
        lines.append(f"| {model} | RGB | | | | | | | | | | not run: current task restricted to YOLO series |")
        lines.append("|  | Sonar | | | | | | | | | | not run: current task restricted to YOLO series |")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
