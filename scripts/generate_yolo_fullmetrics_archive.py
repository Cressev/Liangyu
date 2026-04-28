from __future__ import annotations

import csv
import datetime
import re
from pathlib import Path


ROOT = Path("/dpc/yuanxiangqing/projects/detection")
CODE = ROOT / "MutilModel_423"
RUN = CODE / "run_yolo_bs64_fullmetrics_20260427_r2"
LOG = ROOT / "training_logs" / "yolo_fullmetrics_20260427_r2"
NOW = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")

MODEL_ORDER = ["yolov5s", "yolov8s", "yolov9s", "yolov10s", "yolov11s", "yolov12s", "yolov26s"]


def display_model(name: str) -> str:
    return name.replace("yolov", "YOLOv")


def split_name(run_name: str) -> tuple[str, str]:
    base = run_name.replace("-bs64-fullmetrics", "")
    if base.endswith("-rgb"):
        return base[:-4], "RGB"
    if base.endswith("-sonar"):
        return base[:-6], "Sonar"
    return base, ""


def fmt(value: object, nd: int = 5) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.{nd}f}"
    except Exception:
        return str(value)


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as f:
        rows: list[dict[str, str]] = []
        for row in csv.DictReader(f):
            rows.append({(k.strip() if k else k): (v.strip() if isinstance(v, str) else v) for k, v in row.items()})
        return rows


def fval(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or "nan")
    except Exception:
        return float("nan")


def best_row(rows: list[dict[str, str]]) -> dict[str, str]:
    if not rows:
        return {}
    key = "metrics/mAP50-95(B)"
    valid = [r for r in rows if r.get(key)]
    return max(valid or rows, key=lambda r: fval(r, key))


def latest_log(name: str, completed_only: bool = False) -> tuple[Path | None, str]:
    logs = sorted(
        set(LOG.glob(f"{name}*.log")) | set(LOG.glob(f"*{name}*.log")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if completed_only:
        for p in logs:
            text = p.read_text(errors="ignore")
            if "Results saved to" in text:
                return p, text
    if logs:
        p = logs[0]
        return p, p.read_text(errors="ignore")
    return None, ""


def metrics_from_log(text: str) -> tuple[str, str, str, str]:
    clean = re.sub(r"\x1b\[[0-9;]*m", "", text)
    last_match = None
    for match in re.finditer(
        r"Params:\s*([0-9.]+)M.*?GFLOPs\(total\[default\]\):\s*([0-9.]+)"
        r"(?:\s*\|\s*Avg FPS\(all epochs\):\s*([0-9.]+))?",
        clean,
    ):
        last_match = match
    params = gflops = fps = ""
    if last_match:
        params = last_match.group(1)
        gflops = last_match.group(2)
        fps = last_match.group(3) or ""
    best_epoch_log = ""
    match = re.search(r"Best results observed at epoch\s+([0-9]+)", clean)
    if match:
        best_epoch_log = match.group(1)
    return params, gflops, fps, best_epoch_log


def status_for(name: str) -> tuple[str, Path | None, str]:
    log_path, text = latest_log(name, completed_only=True)
    if log_path and "Results saved to" in text:
        return "completed", log_path, text
    log_path, text = latest_log(name, completed_only=False)
    if text and re.search(r"Traceback|CUDA out of memory|No space left", text):
        return "error", log_path, text
    return "paused", log_path, text


def sort_key(path: Path) -> tuple[int, str]:
    model, modality = split_name(path.name)
    try:
        idx = MODEL_ORDER.index(model)
    except ValueError:
        idx = 99
    return idx, modality


def collect() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    completed: list[dict[str, object]] = []
    paused: list[dict[str, object]] = []
    for run_dir in sorted([p for p in RUN.iterdir() if p.is_dir()], key=sort_key):
        name = run_dir.name
        model, modality = split_name(name)
        rows = read_rows(run_dir / "results.csv")
        br = best_row(rows)
        last = rows[-1] if rows else {}
        state, log_path, log_text = status_for(name)
        params, gflops, fps, best_epoch_log = metrics_from_log(log_text)
        item: dict[str, object] = {
            "name": name,
            "model": display_model(model),
            "modality": modality,
            "run_dir": str(run_dir),
            "log": str(log_path) if log_path else "",
            "state": state,
            "best_row": br,
            "last_row": last,
            "best_epoch_csv": br.get("epoch", ""),
            "last_epoch": last.get("epoch", ""),
            "elapsed": last.get("time", ""),
            "params": params,
            "gflops": gflops,
            "fps": fps,
            "best_epoch_log": best_epoch_log,
            "last_pt": str(run_dir / "weights" / "last.pt"),
            "best_pt": str(run_dir / "weights" / "best.pt"),
            "last_exists": (run_dir / "weights" / "last.pt").exists(),
            "best_exists": (run_dir / "weights" / "best.pt").exists(),
        }
        (completed if state == "completed" else paused).append(item)
    return completed, paused


def table_for_completed(completed: list[dict[str, object]]) -> str:
    headers = [
        "Model",
        "Modality",
        "best_epoch",
        "last_epoch",
        "map50",
        "map75",
        "map50-95",
        "mapS",
        "mapM",
        "mapL",
        "Params(M)",
        "GFLOPS",
        "FPS",
        "log",
        "run_dir",
        "note",
    ]
    lines = [
        "# YOLO Full-Metrics Rerun Completed Table",
        "",
        f"- Generated: {NOW}",
        "- Batch: `20260427_r2`",
        "- Scope: YOLO full-metrics rerun tasks that completed normally.",
        "- Metric source: best row in each run `results.csv`, selected by maximum `metrics/mAP50-95(B)`.",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join([":---"] * len(headers)) + "|",
    ]
    for item in completed:
        row = item["best_row"]
        assert isinstance(row, dict)
        note = (
            f"completed; earlystop_best_epoch={item['best_epoch_log'] or 'n/a'}; "
            "batch=64; no_amp; size metrics emitted during training"
        )
        values = [
            item["model"],
            item["modality"],
            item["best_epoch_csv"],
            item["last_epoch"],
            fmt(row.get("metrics/mAP50(B)")),
            fmt(row.get("metrics/mAP75(B)")),
            fmt(row.get("metrics/mAP50-95(B)")),
            fmt(row.get("metrics/mAP50-95(S)")),
            fmt(row.get("metrics/mAP50-95(M)")),
            fmt(row.get("metrics/mAP50-95(L)")),
            item["params"],
            item["gflops"],
            item["fps"],
            item["log"],
            item["run_dir"],
            note,
        ]
        lines.append("| " + " | ".join(str(v) for v in values) + " |")
    return "\n".join(lines) + "\n"


def archive_text(completed: list[dict[str, object]], paused: list[dict[str, object]]) -> str:
    lines = [
        "# YOLO Full-Metrics Rerun Archive",
        "",
        f"- Generated: {NOW}",
        "- Batch: `20260427_r2`",
        f"- Formal run root: `{RUN}`",
        f"- Smoke run root: `{CODE / 'run_smoke_yolo_fullmetrics_20260427_r2'}`",
        f"- Formal logs: `{LOG}`",
        "- This archive reflects the final state after paused YOLOv12/YOLOv26 jobs were resumed.",
        "",
        "## Completed Tasks",
        "",
        "| Model | Modality | best_epoch | last_epoch | map50 | map75 | map50-95 | mapS | mapM | mapL | Params(M) | GFLOPS | FPS | best.pt | last.pt |",
        "|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|",
    ]
    for item in completed:
        row = item["best_row"]
        assert isinstance(row, dict)
        values = [
            item["model"],
            item["modality"],
            item["best_epoch_csv"],
            item["last_epoch"],
            fmt(row.get("metrics/mAP50(B)")),
            fmt(row.get("metrics/mAP75(B)")),
            fmt(row.get("metrics/mAP50-95(B)")),
            fmt(row.get("metrics/mAP50-95(S)")),
            fmt(row.get("metrics/mAP50-95(M)")),
            fmt(row.get("metrics/mAP50-95(L)")),
            item["params"],
            item["gflops"],
            item["fps"],
            item["best_pt"],
            item["last_pt"],
        ]
        lines.append("| " + " | ".join(str(v) for v in values) + " |")
    lines.extend(
        [
            "",
            "## Paused / Unfinished Tasks",
            "",
            "| Model | Modality | current_epoch | elapsed_s | best.pt exists | last.pt exists | log | resume command |",
            "|:---|:---|:---|:---|:---|:---|:---|:---|",
        ]
    )
    for item in paused:
        cmd = (
            f"cd {CODE} && CUDA_VISIBLE_DEVICES=<GPU> "
            "/dpc/yuanxiangqing/envs/conda/envs/Liangyu/bin/python trainMM.py "
            f"--model {item['last_pt']} --resume --device 0"
        )
        values = [
            item["model"],
            item["modality"],
            item["last_epoch"],
            item["elapsed"],
            "yes" if item["best_exists"] else "no",
            "yes" if item["last_exists"] else "no",
            item["log"],
            f"`{cmd}`",
        ]
        lines.append("| " + " | ".join(str(v) for v in values) + " |")
    lines.extend(
        [
            "",
            "## Resume Notes",
            "",
            "- `trainMM.py` has explicit `--resume/--no-resume` support.",
            "- Resume from `weights/last.pt`; do not pass `--scale` when using `--resume`, because checkpoint architecture is restored from the checkpoint.",
            "- Keep `batch=64`, `imgsz=640`, `amp=False/no_amp`, and the same dataset YAML to stay comparable with this batch.",
            "- At final generation time, no unfinished YOLO r2 runs should remain. If this section is non-empty, resume those rows first.",
            "- The completed table includes all runs whose logs contain `Results saved to`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    completed, paused = collect()
    table = table_for_completed(completed)
    archive = archive_text(completed, paused)
    for path in [
        ROOT / "实验表格_yolo_fullmetrics_20260427_r2_已完成.md",
        ROOT / "实验表格_yolo_fullmetrics_20260427_r2_已完成.txt",
    ]:
        path.write_text(table)
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "yolo_fullmetrics_20260427_r2_experiment_archive.md").write_text(archive)
    (ROOT / "实验存档_yolo_fullmetrics_20260427_r2.md").write_text(archive)
    print("completed_count", len(completed))
    print("paused_count", len(paused))
    print(ROOT / "实验表格_yolo_fullmetrics_20260427_r2_已完成.md")
    print(ROOT / "实验表格_yolo_fullmetrics_20260427_r2_已完成.txt")
    print(reports / "yolo_fullmetrics_20260427_r2_experiment_archive.md")
    print(ROOT / "实验存档_yolo_fullmetrics_20260427_r2.md")


if __name__ == "__main__":
    main()
