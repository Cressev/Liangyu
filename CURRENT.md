# Current Project State

Last updated: 2026-04-28 Asia/Shanghai

## Active Work
- No active YOLO `20260427_r2` training jobs remain.
- Final status: all 14 YOLO full-metrics rerun tasks completed normally.
- YOLOv12 RGB/Sonar ended by EarlyStopping after resume.
- YOLOv26 RGB reached epoch 350; YOLOv26 Sonar ended normally by EarlyStopping.

## Final Outputs
- Remote completed table:
  - `/dpc/yuanxiangqing/projects/detection/实验表格_yolo_fullmetrics_20260427_r2_已完成.md`
  - `/dpc/yuanxiangqing/projects/detection/实验表格_yolo_fullmetrics_20260427_r2_已完成.txt`
- Local completed table:
  - `/Users/liam/Code/codex/zly_detection/实验表格_yolo_fullmetrics_20260427_r2_已完成.md`
  - `/Users/liam/Code/codex/zly_detection/实验表格_yolo_fullmetrics_20260427_r2_已完成.txt`
- Archive:
  - `/dpc/yuanxiangqing/projects/detection/实验存档_yolo_fullmetrics_20260427_r2.md`
  - `/Users/liam/Code/codex/zly_detection/实验存档_yolo_fullmetrics_20260427_r2.md`
  - `/dpc/yuanxiangqing/projects/detection/reports/yolo_fullmetrics_20260427_r2_experiment_archive.md`

## Notes
- Table generation script: `/Users/liam/Code/codex/zly_detection/scripts/generate_yolo_fullmetrics_archive.py` and remote copy `/tmp/generate_yolo_fullmetrics_archive.py`.
- Resume compatibility fixes remain in `trainMM.py` and `ultralytics/engine/model.py` so future checkpoint resumes do not fail on checkpoint `model_scale` metadata.
