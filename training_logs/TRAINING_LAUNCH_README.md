# Training Launch Commands

This file records the commands used to start smoke tests and formal training jobs for the detection baselines.

Run these commands on the training machine:

```bash
ssh root@172.26.0.107
```

## Paths

```bash
ROOT=/dpc/yuanxiangqing/projects/detection
CODE=$ROOT/MutilModel_423
PY=/dpc/yuanxiangqing/envs/conda/envs/Liangyu/bin/python
LOG=$ROOT/training_logs
SMOKE_LOG=$ROOT/training_logs/smoke
RGB_DATA=$CODE/dataset/underwater/data_rgb.yaml
SONAR_DATA=$CODE/dataset/underwater/data_sonar.yaml
RGB_LEGACY=$ROOT/legacy_annotations/rgb
SONAR_LEGACY=$ROOT/legacy_annotations/sonar
```

Keep smoke artifacts separate from formal artifacts:

```bash
mkdir -p "$SMOKE_LOG" \
  "$CODE/run_smoke" \
  "$CODE/run_yolo_bs64" \
  "$CODE/run_non_yolo_bs64" \
  "$CODE/run_non_yolo_legacy"
```

## YOLO Smoke

Example smoke test for one YOLO baseline:

```bash
cd "$CODE"
CUDA_VISIBLE_DEVICES=0 "$PY" trainMM.py \
  --model ultralytics/cfg/models/11/yolo11.yaml \
  --scale s \
  --data "$RGB_DATA" \
  --epochs 1 \
  --batch 64 \
  --device 0 \
  --workers 8 \
  --project "$CODE/run_smoke" \
  --name detect-size-maps-yolo11s-rgb \
  --exist-ok \
  --no-amp \
  > "$SMOKE_LOG/detect-size-maps-yolo11s-rgb_$(date +%Y%m%d_%H%M%S).log" 2>&1
```

YOLO model path mapping:

| Model | `--model` | `--scale` |
| --- | --- | --- |
| YOLOv5s | `ultralytics/cfg/models/v5/yolov5.yaml` | `s` |
| YOLOv8s | `ultralytics/cfg/models/v8/yolov8.yaml` | `s` |
| YOLOv9s | `ultralytics/cfg/models/v9/yolov9s.yaml` | `s` |
| YOLOv10s | `ultralytics/cfg/models/v10/yolov10s.yaml` | `s` |
| YOLOv11s | `ultralytics/cfg/models/11/yolo11.yaml` | `s` |
| YOLOv12s | `ultralytics/cfg/models/12/yolo12.yaml` | `s` |
| YOLOv26s | `ultralytics/cfg/models/26/yolo26n.yaml` | `s` |

## YOLO Formal

Launch one formal YOLO run:

```bash
cd "$CODE"
group=yolov11s-rgb-bs64
ts=$(date +%Y%m%d_%H%M%S)
CUDA_VISIBLE_DEVICES=0 nohup "$PY" trainMM.py \
  --model ultralytics/cfg/models/11/yolo11.yaml \
  --scale s \
  --data "$RGB_DATA" \
  --epochs 350 \
  --batch 64 \
  --device 0 \
  --workers 8 \
  --project "$CODE/run_yolo_bs64" \
  --name "$group" \
  --exist-ok \
  --no-amp \
  > "$LOG/${group}_${ts}.log" 2>&1 &
echo $! > "$LOG/${group}_${ts}.pid"
```

Launch RGB and Sonar for all YOLO baseline models, up to 8 jobs in parallel:

```bash
cd "$CODE"
models=(
  "yolov5s ultralytics/cfg/models/v5/yolov5.yaml s"
  "yolov8s ultralytics/cfg/models/v8/yolov8.yaml s"
  "yolov9s ultralytics/cfg/models/v9/yolov9s.yaml s"
  "yolov10s ultralytics/cfg/models/v10/yolov10s.yaml s"
  "yolov11s ultralytics/cfg/models/11/yolo11.yaml s"
  "yolov12s ultralytics/cfg/models/12/yolo12.yaml s"
  "yolov26s ultralytics/cfg/models/26/yolo26n.yaml s"
)
modalities=(
  "rgb $RGB_DATA"
  "sonar $SONAR_DATA"
)

gpu=0
for model_spec in "${models[@]}"; do
  read -r model_name model_yaml scale <<< "$model_spec"
  for mod_spec in "${modalities[@]}"; do
    read -r mod data_yaml <<< "$mod_spec"
    group=${model_name}-${mod}-bs64
    ts=$(date +%Y%m%d_%H%M%S)
    CUDA_VISIBLE_DEVICES=$gpu nohup "$PY" trainMM.py \
      --model "$model_yaml" \
      --scale "$scale" \
      --data "$data_yaml" \
      --epochs 350 \
      --batch 64 \
      --device 0 \
      --workers 8 \
      --project "$CODE/run_yolo_bs64" \
      --name "$group" \
      --exist-ok \
      --no-amp \
      > "$LOG/${group}_${ts}.log" 2>&1 &
    echo $! > "$LOG/${group}_${ts}.pid"
    echo "$group gpu=$gpu pid=$!"
    gpu=$(( (gpu + 1) % 8 ))
  done
done
```

## RT-DETR Smoke

```bash
cd "$CODE"
CUDA_VISIBLE_DEVICES=0 "$PY" trainRT.py \
  --model ultralytics/cfg/models/rt-detr/rtdetr-l.yaml \
  --data "$RGB_DATA" \
  --epochs 1 \
  --batch 64 \
  --device 0 \
  --workers 8 \
  --project "$CODE/run_smoke" \
  --name rtdetr-l-rgb-bs64-smoke \
  --exist-ok \
  --no-amp \
  > "$SMOKE_LOG/rtdetr-l-rgb-bs64-smoke_$(date +%Y%m%d_%H%M%S).log" 2>&1
```

## RT-DETR Formal

```bash
cd "$CODE"
for item in "rgb 0 $RGB_DATA" "sonar 1 $SONAR_DATA"; do
  read -r mod gpu data_yaml <<< "$item"
  group=rtdetr-l-${mod}-bs64
  ts=$(date +%Y%m%d_%H%M%S)
  CUDA_VISIBLE_DEVICES=$gpu nohup "$PY" trainRT.py \
    --model ultralytics/cfg/models/rt-detr/rtdetr-l.yaml \
    --data "$data_yaml" \
    --epochs 350 \
    --batch 64 \
    --device 0 \
    --workers 8 \
    --project "$CODE/run_non_yolo_bs64" \
    --name "$group" \
    --exist-ok \
    --no-amp \
    > "$LOG/${group}_${ts}.log" 2>&1 &
  echo $! > "$LOG/${group}_${ts}.pid"
  echo "$group gpu=$gpu pid=$!"
done
```

## SSD Smoke

SSD uses converted VOC-style annotation text files under `legacy_annotations`.

```bash
cd "$CODE/ssd-pytorch-master"
CUDA_VISIBLE_DEVICES=2 \
SSD_CLASSES_PATH="$RGB_LEGACY/classes.txt" \
SSD_TRAIN_TXT="$RGB_LEGACY/train.txt" \
SSD_VAL_TXT="$RGB_LEGACY/val.txt" \
SSD_SAVE_DIR="$CODE/run_smoke/ssd-rgb-bs64-smoke" \
SSD_MODEL_PATH="" \
SSD_EPOCHS=1 \
SSD_BATCH=64 \
SSD_FREEZE_TRAIN=False \
SSD_EVAL_PERIOD=1 \
SSD_SAVE_PERIOD=1 \
SSD_WORKERS=4 \
"$PY" train.py \
  > "$SMOKE_LOG/ssd-rgb-bs64-smoke_$(date +%Y%m%d_%H%M%S).log" 2>&1
```

## SSD Formal

```bash
cd "$CODE/ssd-pytorch-master"
for item in "rgb 2 $RGB_LEGACY" "sonar 3 $SONAR_LEGACY"; do
  read -r mod gpu ann_root <<< "$item"
  group=ssd-${mod}-bs64
  ts=$(date +%Y%m%d_%H%M%S)
  save="$CODE/run_non_yolo_legacy/$group"
  mkdir -p "$save"
  CUDA_VISIBLE_DEVICES=$gpu \
  SSD_CLASSES_PATH="$ann_root/classes.txt" \
  SSD_TRAIN_TXT="$ann_root/train.txt" \
  SSD_VAL_TXT="$ann_root/val.txt" \
  SSD_SAVE_DIR="$save" \
  SSD_MODEL_PATH="" \
  SSD_EPOCHS=350 \
  SSD_BATCH=64 \
  SSD_FREEZE_TRAIN=False \
  SSD_EVAL_PERIOD=10 \
  SSD_SAVE_PERIOD=10 \
  SSD_WORKERS=4 \
  nohup "$PY" train.py > "$LOG/${group}_${ts}.log" 2>&1 &
  echo $! > "$LOG/${group}_${ts}.pid"
  echo "$group gpu=$gpu pid=$!"
done
```

Resume SSD from an existing checkpoint:

```bash
cd "$CODE/ssd-pytorch-master"
group=ssd-sonar-bs64-resume-e9
ts=$(date +%Y%m%d_%H%M%S)
save="$CODE/run_non_yolo_legacy/ssd-sonar-bs64"
CUDA_VISIBLE_DEVICES=3 \
SSD_CLASSES_PATH="$SONAR_LEGACY/classes.txt" \
SSD_TRAIN_TXT="$SONAR_LEGACY/train.txt" \
SSD_VAL_TXT="$SONAR_LEGACY/val.txt" \
SSD_SAVE_DIR="$save" \
SSD_MODEL_PATH="$save/last_epoch_weights.pth" \
SSD_INIT_EPOCH=9 \
SSD_EPOCHS=350 \
SSD_BATCH=64 \
SSD_FREEZE_TRAIN=False \
SSD_EVAL_PERIOD=10 \
SSD_SAVE_PERIOD=10 \
SSD_WORKERS=4 \
nohup "$PY" train.py > "$LOG/${group}_${ts}.log" 2>&1 &
echo $! > "$LOG/${group}_${ts}.pid"
```

## Faster R-CNN Smoke

Do not use external `CUDA_VISIBLE_DEVICES` for Faster R-CNN. The script sets it internally from `FRCNN_GPUS`.

```bash
cd "$CODE/faster-rcnn-pytorch-master"
FRCNN_GPUS=4 \
FRCNN_CLASSES_PATH="$RGB_LEGACY/classes.txt" \
FRCNN_TRAIN_TXT="$RGB_LEGACY/train.txt" \
FRCNN_VAL_TXT="$RGB_LEGACY/val.txt" \
FRCNN_SAVE_DIR="$CODE/run_smoke/frcnn-rgb-bs2-smoke" \
FRCNN_EPOCHS=1 \
FRCNN_BATCH=2 \
FRCNN_FREEZE_TRAIN=False \
FRCNN_EVAL_PERIOD=1 \
FRCNN_SAVE_PERIOD=1 \
FRCNN_WORKERS=4 \
"$PY" train.py \
  > "$SMOKE_LOG/frcnn-rgb-bs2-smoke_$(date +%Y%m%d_%H%M%S).log" 2>&1
```

## Faster R-CNN Formal

Faster R-CNN OOMed at larger batches during smoke testing; `FRCNN_BATCH=2` is the stable formal setting used here.

```bash
cd "$CODE/faster-rcnn-pytorch-master"
for item in "rgb 4 $RGB_LEGACY" "sonar 5 $SONAR_LEGACY"; do
  read -r mod gpu ann_root <<< "$item"
  group=frcnn-${mod}-bs2
  ts=$(date +%Y%m%d_%H%M%S)
  save="$CODE/run_non_yolo_legacy/$group"
  mkdir -p "$save"
  FRCNN_GPUS=$gpu \
  FRCNN_CLASSES_PATH="$ann_root/classes.txt" \
  FRCNN_TRAIN_TXT="$ann_root/train.txt" \
  FRCNN_VAL_TXT="$ann_root/val.txt" \
  FRCNN_SAVE_DIR="$save" \
  FRCNN_EPOCHS=350 \
  FRCNN_BATCH=2 \
  FRCNN_FREEZE_TRAIN=False \
  FRCNN_EVAL_PERIOD=10 \
  FRCNN_SAVE_PERIOD=10 \
  FRCNN_WORKERS=4 \
  nohup "$PY" train.py > "$LOG/${group}_${ts}.log" 2>&1 &
  echo $! > "$LOG/${group}_${ts}.pid"
  echo "$group gpu=$gpu pid=$!"
done
```

## Monitoring

```bash
nvidia-smi
ps -ef | grep -E "trainMM.py|trainRT.py|ssd-pytorch-master/train.py|faster-rcnn-pytorch-master/train.py" | grep -v grep
tail -f "$LOG/<log-file>.log"
```

## Metric Notes

- YOLO and RT-DETR use the patched Ultralytics standard detection validator, which emits `metrics/mAP50-95(S)`, `metrics/mAP50-95(M)`, and `metrics/mAP50-95(L)` in `results.csv`.
- SSD and Faster R-CNN use their own COCO eval path; logs print COCO `area=small`, `area=medium`, and `area=large` AP lines.
- SSD/Faster R-CNN eval callbacks were patched so temporary map files are created under each run's own `loss_*` directory, avoiding collisions between parallel RGB/Sonar jobs.
