#!/usr/bin/env bash
set -u

ROOT=/dpc/yuanxiangqing/projects/detection
PROJECT=$ROOT/MutilModel_423
ENV_PY=/dpc/yuanxiangqing/envs/conda/envs/Liangyu/bin/python
LOG_DIR=$ROOT/training_logs
RUN_PROJECT=run_yolo_bs64
SMOKE_PROJECT=run_smoke_yolo_bs64
TS=$(date +%Y%m%d_%H%M%S)
README=$LOG_DIR/README.md
STATE=$LOG_DIR/yolo_bs64_launch_${TS}.tsv

mkdir -p "$LOG_DIR"

cat > "$README" <<EOF
# Training Log Index

Older log mappings were cleared as requested. YOLO-series bs64 experiments are listed below.

| Group | Model | Modality | Batch | Epochs | GPU | PID | Smoke log | Train log | Output directory | Status | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |
EOF

printf "group\tmodel\tmodality\tgpu\tpid\tstatus\tsmoke_log\ttrain_log\toutput\tnote\n" > "$STATE"

append_row() {
  local group="$1" model="$2" modality="$3" gpu="$4" pid="$5" smoke_log="$6" train_log="$7" output="$8" status="$9" note="${10}"
  echo "| $group | $model | $modality | 64 | 350 | $gpu | $pid | $smoke_log | $train_log | $output | $status | $note |" >> "$README"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$group" "$model" "$modality" "$gpu" "$pid" "$status" "$smoke_log" "$train_log" "$output" "$note" >> "$STATE"
}

launch_one() {
  local idx="$1" model="$2" modality="$3" yaml="$4" scale="$5" data="$6"
  local group model_slug modality_slug smoke_log train_log output
  model_slug=$(echo "$model" | tr '[:upper:]' '[:lower:]' | tr -d ' -')
  modality_slug=$(echo "$modality" | tr '[:upper:]' '[:lower:]')
  group="${model_slug}-${modality_slug}-bs64"
  smoke_log="$LOG_DIR/${group}_smoke_${TS}.log"
  train_log="$LOG_DIR/${group}_${TS}.log"
  output="$PROJECT/$RUN_PROJECT/$group"

  echo "$(date '+%F %T') smoke start $group gpu=$idx"
  if [ -n "$scale" ]; then
    scale_arg=(--scale "$scale")
  else
    scale_arg=(--scale "")
  fi

  (
    cd "$PROJECT" &&
    CUDA_VISIBLE_DEVICES="$idx" PYTHONPATH="$PROJECT" "$ENV_PY" trainMM.py \
      --model "$yaml" \
      --data "$data" \
      --epochs 1 \
      --batch 64 \
      --imgsz 640 \
      --workers 4 \
      --project "$SMOKE_PROJECT" \
      --name "smoke-$group" \
      --device 0 \
      --exist-ok \
      "${scale_arg[@]}"
  ) > "$smoke_log" 2>&1
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "$(date '+%F %T') smoke failed $group rc=$rc"
    append_row "$group" "$model" "$modality" "$idx" "" "$smoke_log" "" "$output" "smoke_failed" "smoke rc=$rc"
    return 0
  fi

  echo "$(date '+%F %T') train launch $group gpu=$idx"
  (
    cd "$PROJECT" &&
    CUDA_VISIBLE_DEVICES="$idx" PYTHONPATH="$PROJECT" "$ENV_PY" trainMM.py \
      --model "$yaml" \
      --data "$data" \
      --epochs 350 \
      --batch 64 \
      --imgsz 640 \
      --workers 8 \
      --project "$RUN_PROJECT" \
      --name "$group" \
      --device 0 \
      --exist-ok \
      "${scale_arg[@]}"
  ) > "$train_log" 2>&1 &
  local pid=$!
  echo "$pid" > "${train_log%.log}.pid"
  append_row "$group" "$model" "$modality" "$idx" "$pid" "$smoke_log" "$train_log" "$output" "running" ""
}

RGB_DATA=dataset/underwater/data_rgb.yaml
SONAR_DATA=dataset/underwater/data_sonar.yaml

launch_one 0 YOLOv5s RGB ultralytics/cfg/models/v5/yolov5.yaml s "$RGB_DATA"
launch_one 1 YOLOv5s Sonar ultralytics/cfg/models/v5/yolov5.yaml s "$SONAR_DATA"
launch_one 2 YOLOv8s RGB ultralytics/cfg/models/v8/yolov8.yaml s "$RGB_DATA"
launch_one 3 YOLOv8s Sonar ultralytics/cfg/models/v8/yolov8.yaml s "$SONAR_DATA"
launch_one 4 YOLOv9s RGB ultralytics/cfg/models/v9/yolov9s.yaml "" "$RGB_DATA"
launch_one 5 YOLOv9s Sonar ultralytics/cfg/models/v9/yolov9s.yaml "" "$SONAR_DATA"
launch_one 6 YOLOv10s RGB ultralytics/cfg/models/v10/yolov10s.yaml "" "$RGB_DATA"
launch_one 7 YOLOv10s Sonar ultralytics/cfg/models/v10/yolov10s.yaml "" "$SONAR_DATA"

cat >> "$README" <<EOF

## Pending YOLO Experiments

| Group | Model | Modality | Status |
| --- | --- | --- | --- |
| yolov11s-rgb-bs64 | YOLOv11s | RGB | pending |
| yolov11s-sonar-bs64 | YOLOv11s | Sonar | pending |
| yolov12s-rgb-bs64 | YOLOv12s | RGB | pending |
| yolov12s-sonar-bs64 | YOLOv12s | Sonar | pending |
| yolov26s-rgb-bs64 | YOLOv26s | RGB | pending |
| yolov26s-sonar-bs64 | YOLOv26s | Sonar | pending |

## Notes

- Non-YOLO rows are not launched in this batch.
- The replacement ultralytics.zip smoke test did not add mapS/mapM/mapL columns to results.csv.
- Launch state: $STATE
EOF

echo "STATE=$STATE"
echo "README=$README"
