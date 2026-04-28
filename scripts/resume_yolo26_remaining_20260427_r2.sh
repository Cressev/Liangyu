#!/usr/bin/env bash
set -euo pipefail

ROOT=/dpc/yuanxiangqing/projects/detection/MutilModel_423
PY=/dpc/yuanxiangqing/envs/conda/envs/Liangyu/bin/python
LOGDIR=/dpc/yuanxiangqing/projects/detection/training_logs/yolo_fullmetrics_20260427_r2
STAMP=$(date +%Y%m%d_%H%M%S)
SCHED_LOG="$LOGDIR/resume_yolo26_scheduler_${STAMP}.log"

mkdir -p "$LOGDIR"
cd "$ROOT"

run_resume() {
  local physical_gpu="$1"
  local exp_name="$2"
  local ckpt="$3"
  local log="$LOGDIR/resume_${STAMP}_${exp_name}.log"

  printf '[%s] START gpu=%s exp=%s ckpt=%s\n' "$(date '+%F %T')" "$physical_gpu" "$exp_name" "$ckpt" >>"$log"
  CUDA_VISIBLE_DEVICES="$physical_gpu" "$PY" trainMM.py --model "$ckpt" --resume --device 0 >>"$log" 2>&1
  local rc=$?
  printf '[%s] END rc=%s gpu=%s exp=%s\n' "$(date '+%F %T')" "$rc" "$physical_gpu" "$exp_name" >>"$log"
  return "$rc"
}

{
  printf '[%s] scheduler start, pid=%s\n' "$(date '+%F %T')" "$$"
  printf 'using physical GPUs 6 and 7 for remaining YOLOv26 jobs\n'
} >>"$SCHED_LOG"

run_resume 6 yolov26s-rgb-bs64-fullmetrics run_yolo_bs64_fullmetrics_20260427_r2/yolov26s-rgb-bs64-fullmetrics/weights/last.pt &
pid_a=$!
run_resume 7 yolov26s-sonar-bs64-fullmetrics run_yolo_bs64_fullmetrics_20260427_r2/yolov26s-sonar-bs64-fullmetrics/weights/last.pt &
pid_b=$!

{
  printf 'worker_a_pid=%s\n' "$pid_a"
  printf 'worker_b_pid=%s\n' "$pid_b"
  wait "$pid_a"
  rc_a=$?
  wait "$pid_b"
  rc_b=$?
  printf '[%s] scheduler end rc_a=%s rc_b=%s\n' "$(date '+%F %T')" "$rc_a" "$rc_b"
  if [ "$rc_a" -ne 0 ] || [ "$rc_b" -ne 0 ]; then
    exit 1
  fi
} >>"$SCHED_LOG" 2>&1
