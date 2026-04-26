#!/usr/bin/env bash
set -u
PROJECT=/dpc/yuanxiangqing/projects/detection/MutilModel_423
OUT=/dpc/yuanxiangqing/projects/detection/training_logs/yolo11s_bs144_runtime_monitor_20260426_074605.log
RGB_PID=505211
SONAR_PID=505212
RGB_RESULTS=$PROJECT/run_baselines/yolo11s-rgb-bs144/results.csv
SONAR_RESULTS=$PROJECT/run_baselines/yolo11s-sonar-bs144/results.csv
interval=300
summarize() {
  local name=$1 pid=$2 results=$3 gpu=$4
  local alive=no
  ps -p "$pid" >/dev/null 2>&1 && alive=yes
  local row=""
  if [ -f "$results" ]; then
    row=$(awk -F, "NR>1{e=\$1;t=\$2} END{if(e>0) printf(\"epoch=%s elapsed_sec=%.1f avg_sec_per_epoch=%.2f eta_hours=%.2f\", e,t,t/e,(350-e)*t/e/3600); else printf(\"epoch=0 elapsed_sec=0 avg_sec_per_epoch=0 eta_hours=unknown\")}" "$results")
  else
    row="epoch=0 elapsed_sec=0 avg_sec_per_epoch=0 eta_hours=unknown"
  fi
  local gpu_row
  gpu_row=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | awk -F, "{gsub(/ /,\"\"); printf(\"gpu_mem_mib=%s gpu_util_pct=%s\", \$1, \$2)}")
  printf "%s group=%s pid=%s alive=%s gpu=%s %s %s\n" "$(date "+%F %T")" "$name" "$pid" "$alive" "$gpu" "$row" "$gpu_row" >> "$OUT"
}
echo "# Runtime monitor started at $(date "+%F %T")" >> "$OUT"
while true; do
  summarize rgb "$RGB_PID" "$RGB_RESULTS" 0
  summarize sonar "$SONAR_PID" "$SONAR_RESULTS" 1
  if ! ps -p "$RGB_PID" >/dev/null 2>&1 && ! ps -p "$SONAR_PID" >/dev/null 2>&1; then
    echo "# Runtime monitor finished at $(date "+%F %T")" >> "$OUT"
    break
  fi
  sleep "$interval"
done
