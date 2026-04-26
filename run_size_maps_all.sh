#!/usr/bin/env bash
set -u

ROOT=/dpc/yuanxiangqing/projects/detection
PROJECT=$ROOT/MutilModel_423
ENV_PY=/dpc/yuanxiangqing/envs/conda/envs/Liangyu/bin/python
LOG_DIR=$ROOT/training_logs
OUT_DIR=$ROOT/reports/size_maps
EVAL=$ROOT/eval_size_maps.py
mkdir -p "$OUT_DIR" "$LOG_DIR"

run_one() {
  local gpu="$1" group="$2" modality="$3" data="$4"
  local weights="$PROJECT/run_yolo_bs64/$group/weights/best.pt"
  local out="$OUT_DIR/${group}.json"
  local log="$LOG_DIR/${group}_size_maps.log"
  echo "$(date '+%F %T') start $group gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$PROJECT" "$ENV_PY" "$EVAL" \
    --weights "$weights" \
    --data "$PROJECT/$data" \
    --modality "$modality" \
    --device 0 \
    --output "$out" > "$log" 2>&1
  echo "$(date '+%F %T') done $group rc=$?"
}

run_batch() {
  "$@" &
}

RGB=dataset/underwater/data_rgb.yaml
SONAR=dataset/underwater/data_sonar.yaml

run_one 0 yolov5s-rgb-bs64 RGB "$RGB" &
run_one 1 yolov5s-sonar-bs64 Sonar "$SONAR" &
run_one 2 yolov8s-rgb-bs64 RGB "$RGB" &
run_one 3 yolov8s-sonar-bs64 Sonar "$SONAR" &
run_one 4 yolov9s-rgb-bs64 RGB "$RGB" &
run_one 5 yolov9s-sonar-bs64 Sonar "$SONAR" &
run_one 6 yolov10s-rgb-bs64 RGB "$RGB" &
run_one 7 yolov10s-sonar-bs64 Sonar "$SONAR" &
wait

run_one 0 yolov11s-rgb-bs64 RGB "$RGB" &
run_one 1 yolov11s-sonar-bs64 Sonar "$SONAR" &
run_one 2 yolov12s-rgb-bs64 RGB "$RGB" &
run_one 3 yolov12s-sonar-bs64 Sonar "$SONAR" &
run_one 4 yolov26s-rgb-bs64 RGB "$RGB" &
run_one 5 yolov26s-sonar-bs64 Sonar "$SONAR" &
wait

echo "$(date '+%F %T') all size maps done"
