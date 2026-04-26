"""
Batch preflight test — representative YAML configs from every major category.

Usage:
  # Full test matrix (default)
  python preflight_batch_test.py

  # Specific YAML configs
  python preflight_batch_test.py path/a.yaml path/b.yaml

  # With options
  python preflight_batch_test.py --device 0 --timeout 120 path/a.yaml path/b.yaml
  python preflight_batch_test.py --task segment path/a.yaml
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PYTHON = "/home/zhizi/anaconda3/envs/yolon/bin/python"
SCRIPT = Path(__file__).parent / "preflightMM.py"
BASE = "ultralytics/cfg/models"

# (label, yaml_path, task)
TEST_MATRIX = [
    # ── A: YOLO11 基础单模态 ──
    ("A1", "YOLO11 detect (单模态)",        f"{BASE}/11/yolo11.yaml",                None),

    # ── B: 多模态融合策略 ──
    ("B1", "MM early fusion",               f"{BASE}/mm/yolo11n-mm-early.yaml",      None),
    ("B2", "MM mid fusion",                 f"{BASE}/mm/yolo11n-mm-mid.yaml",        None),
    ("B3", "MM late fusion",                f"{BASE}/mm/yolo11n-mm-late-ref.yaml",   None),

    # ── C: 多模态多任务 ──
    ("C1", "MM segment",                    f"{BASE}/mm/yolo11n-mm-mid-seg.yaml",    "segment"),
    ("C2", "MM pose",                       f"{BASE}/mm/yolo11n-mm-mid-pose.yaml",   "pose"),
    ("C3", "MM obb",                        f"{BASE}/mm/yolo11n-mm-obb.yaml",        "obb"),
    ("C4", "MM classify",                   f"{BASE}/mm/yolo11n-mm-mid-cls.yaml",    "classify"),

    # ── D: 多模态架构变体 ──
    ("D1", "MM asymmetric",                 f"{BASE}/mm/yolo11n-mm-asymmetric.yaml",  None),
    ("D2", "MM residual",                   f"{BASE}/mm/yolo11n-mm-residual.yaml",    None),
    ("D3", "MM mrod (YOLO-based)",          f"{BASE}/mm/mrod/yolo11n-mm-mrod.yaml",   None),

    # ── E: 融合模块变体 ──
    ("E1", "MM CTF fusion",                 f"{BASE}/mm/change/yolo11n-mm-mid-ctf.yaml",  None),
    ("E2", "MM SEFN fusion",                f"{BASE}/mm/change/yolo11n-mm-mid-sefn.yaml", None),
    ("E3", "MM IIA fusion",                 f"{BASE}/mm/change/yolo11n-mm-mid-iia.yaml",  None),

    # ── F: 提取模块变体 ──
    ("F1", "MM C2PSA-DYT",                  f"{BASE}/mm/extraction/yolo11n-mm-c2psa-dyt.yaml", None),
    ("F2", "MM C2PSA-CGLU",                 f"{BASE}/mm/extraction/yolo11n-mm-c2psa-cglu.yaml",None),

    # ── G: Neck 变体 ──
    ("G1", "MM Neck MSIA",                  f"{BASE}/mm/Neck/yolo11n-mm-mid-msia.yaml", None),
    ("G2", "MM Neck SOEP",                  f"{BASE}/mm/Neck/yolo11n-mm-mid-soep.yaml", None),

    # ── H: Head 变体 ──
    ("H1", "MM Head LSCD",                  f"{BASE}/mm/Head/yolo11n-mm-mid-lscd.yaml", None),

    # ── I: YOLOv8 多模态 ──
    ("I1", "v8 MM mid",                     f"{BASE}/mm/v8/yolov8-mm-mid.yaml",       None),

    # ── J: YOLO26 系列 ──
    ("J1", "YOLO26 detect",                 f"{BASE}/26/yolo26n.yaml",                None),
    ("J2", "YOLO26 MM mid",                 f"{BASE}/mm/26/yolo26n-mm-mid.yaml",      None),

    # ── K: RTDETRMM ──
    ("K1", "RTDETRMM r18 mid",              f"{BASE}/rtmm/r18/rtdetr-r18-mm-mid.yaml", None),
    ("K2", "RTDETRMM r18 SEFN",             f"{BASE}/rtmm/r18/rtdetr-r18-mm-mid-sefn.yaml", None),

    # ── L: RT-DETR 基础单模态 ──
    ("L1", "RT-DETR r34 (单模态)",          f"{BASE}/rt-detr/rtdetr-r34.yaml",        None),
]


def run_one(tag, label, yaml_path, task, device="cpu", timeout=300, scale=""):
    """Run preflight for a single config, return result dict."""
    cmd = [PYTHON, str(SCRIPT), yaml_path, "--device", device, "--json", "--quiet"]
    if task:
        cmd += ["--task", task]
    if scale:
        cmd += ["--scale", scale]

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=str(Path(__file__).resolve().parent),
        )
    except subprocess.TimeoutExpired:
        return {"tag": tag, "label": label, "ok": False, "fail_stage": "TIMEOUT",
                "fail_msg": f"exceeded {timeout}s", "elapsed": time.time() - t0}

    elapsed = time.time() - t0

    # Parse JSON from stdout — find first '{' and collect everything after it
    report = None
    lines = proc.stdout.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("{"):
            start = i
            break

    if start is not None:
        json_text = "\n".join(lines[start:])
        try:
            report = json.loads(json_text)
        except json.JSONDecodeError:
            pass

    if report is None:
        # Fallback: try last line that looks like JSON
        last_err = ""
        for line in reversed((proc.stderr or proc.stdout).strip().splitlines()):
            line = line.strip()
            if line:
                last_err = line
                break
        return {"tag": tag, "label": label, "ok": False, "fail_stage": "PARSE",
                "fail_msg": last_err[-80:], "elapsed": elapsed}

    ok = report.get("ok", False)
    model_class = ""
    params = ""
    fail_stage = report.get("failed_stage", "")
    fail_msg = ""

    for stage in report.get("stages", []):
        if stage.get("name") == "Model build" and stage.get("passed"):
            model_class = stage.get("data", {}).get("model_class", "")
            p = stage.get("data", {}).get("parameters", "")
            params = f"{p:,}" if isinstance(p, int) else str(p)
        if not stage.get("passed"):
            fail_msg = stage.get("error_message", "")[:60]
            break

    return {
        "tag": tag, "label": label, "ok": ok,
        "model_class": model_class, "params": params,
        "fail_stage": fail_stage or "", "fail_msg": fail_msg,
        "elapsed": round(elapsed, 1),
    }


def build_custom_matrix(yaml_paths, task_override=None):
    """Build test matrix from user-provided YAML paths."""
    matrix = []
    for i, yp in enumerate(yaml_paths, 1):
        label = Path(yp).stem
        matrix.append((f"#{i}", label, yp, task_override))
    return matrix


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch preflight test for YOLOMM/RTDETRMM YAML configs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s                                          # full test matrix
  %(prog)s a.yaml b.yaml c.yaml                     # specific configs
  %(prog)s --device 0 a.yaml b.yaml                 # GPU mode
  %(prog)s --task segment a.yaml                     # task override
  %(prog)s --timeout 120 a.yaml b.yaml              # custom timeout
""",
    )
    parser.add_argument("yamls", nargs="*", help="YAML config paths (default: full test matrix)")
    parser.add_argument("--device", default="cpu", help="Device for preflight (default: cpu)")
    parser.add_argument("--task", default=None,
                        help="Task override for all YAMLs (detect/segment/pose/obb/classify)")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per config in seconds (default: 300)")
    parser.add_argument("--scale", default="", help="Model scale key, e.g. n/s/m/l/x (default: YAML first key)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Build test matrix: custom YAMLs or full hardcoded matrix
    if args.yamls:
        matrix = build_custom_matrix(args.yamls, args.task)
    else:
        matrix = TEST_MATRIX

    total = len(matrix)
    results = []

    print(f"\n{'='*82}")
    print(f"  Preflight Batch Test  —  {total} configs  —  {args.device.upper()} mode")
    print(f"{'='*82}\n")

    for i, (tag, label, yaml_path, task) in enumerate(matrix, 1):
        print(f"  [{i:2d}/{total}] {tag:3s} {label:<36s} ", end="", flush=True)
        r = run_one(tag, label, yaml_path, task, device=args.device, timeout=args.timeout, scale=args.scale)
        results.append(r)
        if r["ok"]:
            print(f"  OK   {r['elapsed']:>6.1f}s  {r['model_class']}")
        else:
            print(f"  FAIL {r['elapsed']:>6.1f}s  [{r['fail_stage']}] {r['fail_msg']}")

    # ── Summary ──
    passed = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]

    print(f"\n{'='*82}")
    print(f"  RESULT: {len(passed)}/{total} PASSED, {len(failed)} FAILED")
    print(f"{'='*82}\n")

    if passed:
        print(f"  PASSED ({len(passed)}):")
        for r in passed:
            print(f"    {r['tag']:3s}  {r['label']:<36s}  {r['model_class']:>25s}  params={r['params']:>12s}  {r['elapsed']}s")

    if failed:
        print(f"\n  FAILED ({len(failed)}):")
        for r in failed:
            print(f"    {r['tag']:3s}  {r['label']:<36s}  [{r['fail_stage']}] {r['fail_msg']}")

    print()
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
