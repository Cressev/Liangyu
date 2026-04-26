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
LOG=$LOG_DIR/yolo_remaining6_launcher_${TS}.log

declare -a ACTIVE_PIDS=(545597 547003 548502 549937 551451 552957 554424 555916)
declare -a ACTIVE_GPUS=(0 1 2 3 4 5 6 7)

declare -a MODELS=(YOLOv11s YOLOv11s YOLOv12s YOLOv12s YOLOv26s YOLOv26s)
declare -a MODS=(RGB Sonar RGB Sonar RGB Sonar)
declare -a YAMLS=(ultralytics/cfg/models/11/yolo11.yaml ultralytics/cfg/models/11/yolo11.yaml ultralytics/cfg/models/12/yolo12.yaml ultralytics/cfg/models/12/yolo12.yaml ultralytics/cfg/models/26/yolo26n.yaml ultralytics/cfg/models/26/yolo26n.yaml)
declare -a SCALES=(s s s s s s)
declare -a DATAS=(dataset/underwater/data_rgb.yaml dataset/underwater/data_sonar.yaml dataset/underwater/data_rgb.yaml dataset/underwater/data_sonar.yaml dataset/underwater/data_rgb.yaml dataset/underwater/data_sonar.yaml)

log() {
  echo "$(date '+%F %T') $*" | tee -a "$LOG"
}

group_name() {
  local model="$1" mod="$2"
  local model_slug mod_slug
  model_slug=$(echo "$model" | tr '[:upper:]' '[:lower:]' | tr -d ' -')
  mod_slug=$(echo "$mod" | tr '[:upper:]' '[:lower:]')
  echo "${model_slug}-${mod_slug}-bs64"
}

append_readme_row() {
  local group="$1" model="$2" mod="$3" gpu="$4" pid="$5" smoke_log="$6" train_log="$7" output="$8" status="$9" note="${10}"
  sed -i "/| ${group} | ${model} | ${mod} | pending |/d" "$README"
  sed -i "/| ${group} | ${model} | ${mod} | 64 | 350 |/d" "$README"
  sed -i "/^## Pending YOLO Experiments/i | ${group} | ${model} | ${mod} | 64 | 350 | ${gpu} | ${pid} | ${smoke_log} | ${train_log} | ${output} | ${status} | ${note} |" "$README"
}

launch_one() {
  local slot_gpu="$1" model="$2" mod="$3" yaml="$4" scale="$5" data="$6"
  local group smoke_log train_log output scale_arg pid rc
  group=$(group_name "$model" "$mod")
  smoke_log="$LOG_DIR/${group}_smoke_${TS}.log"
  train_log="$LOG_DIR/${group}_${TS}.log"
  output="$PROJECT/$RUN_PROJECT/$group"
  log "smoke start $group gpu=$slot_gpu"

  (
    cd "$PROJECT" &&
    CUDA_VISIBLE_DEVICES="$slot_gpu" PYTHONPATH="$PROJECT" "$ENV_PY" trainMM.py \
      --model "$yaml" --data "$data" --epochs 1 --batch 64 --imgsz 640 --workers 4 \
      --project "$SMOKE_PROJECT" --name "smoke-$group" --device 0 --exist-ok --scale "$scale"
  ) > "$smoke_log" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    log "smoke failed $group rc=$rc"
    append_readme_row "$group" "$model" "$mod" "$slot_gpu" "" "$smoke_log" "" "$output" "smoke_failed" "smoke rc=$rc"
    return 1
  fi

  log "train launch $group gpu=$slot_gpu"
  (
    cd "$PROJECT" &&
    CUDA_VISIBLE_DEVICES="$slot_gpu" PYTHONPATH="$PROJECT" "$ENV_PY" trainMM.py \
      --model "$yaml" --data "$data" --epochs 350 --batch 64 --imgsz 640 --workers 8 \
      --project "$RUN_PROJECT" --name "$group" --device 0 --exist-ok --scale "$scale"
  ) > "$train_log" 2>&1 &
  pid=$!
  echo "$pid" > "${train_log%.log}.pid"
  append_readme_row "$group" "$model" "$mod" "$slot_gpu" "$pid" "$smoke_log" "$train_log" "$output" "running" ""
  log "launched $group pid=$pid gpu=$slot_gpu"
  ACTIVE_PIDS+=("$pid")
  ACTIVE_GPUS+=("$slot_gpu")
}

log "remaining launcher started"
idx=0
while [ "$idx" -lt "${#MODELS[@]}" ]; do
  freed_gpu=""
  freed_index=""
  for i in "${!ACTIVE_PIDS[@]}"; do
    pid="${ACTIVE_PIDS[$i]}"
    gpu="${ACTIVE_GPUS[$i]}"
    if ! ps -p "$pid" >/dev/null 2>&1; then
      freed_gpu="$gpu"
      freed_index="$i"
      break
    fi
  done
  if [ -z "$freed_gpu" ]; then
    sleep 300
    continue
  fi
  unset 'ACTIVE_PIDS[freed_index]'
  unset 'ACTIVE_GPUS[freed_index]'
  ACTIVE_PIDS=("${ACTIVE_PIDS[@]}")
  ACTIVE_GPUS=("${ACTIVE_GPUS[@]}")

  launch_one "$freed_gpu" "${MODELS[$idx]}" "${MODS[$idx]}" "${YAMLS[$idx]}" "${SCALES[$idx]}" "${DATAS[$idx]}" || true
  idx=$((idx + 1))
done
log "all remaining experiments launched"
