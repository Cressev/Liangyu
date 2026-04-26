# Current State

- Remote project root: /dpc/yuanxiangqing/projects/detection.
- Active conda environment: /dpc/yuanxiangqing/envs/conda/envs/Liangyu.
- Project code: /dpc/yuanxiangqing/projects/detection/MutilModel_423.
- ultralytics package was replaced from /Users/liam/Downloads/ultralytics.zip; previous remote package backup is ultralytics.backup_before_zip_20260426_084448.
- Completed YOLO bs64 experiments: YOLOv5s, YOLOv8s, YOLOv9s, YOLOv10s, YOLOv11s, YOLOv12s, YOLOv26s on RGB and Sonar.
- Log mapping: /dpc/yuanxiangqing/projects/detection/training_logs/README.md.
- YOLO output root: /dpc/yuanxiangqing/projects/detection/MutilModel_423/run_yolo_bs64.
- Smoke output root: /dpc/yuanxiangqing/projects/detection/MutilModel_423/run_smoke.
- Smoke log root: /dpc/yuanxiangqing/projects/detection/training_logs/smoke.
- Non-YOLO formal output roots: /dpc/yuanxiangqing/projects/detection/MutilModel_423/run_non_yolo_bs64 and /dpc/yuanxiangqing/projects/detection/MutilModel_423/run_non_yolo_legacy.
- Local experiment table updated: /Users/liam/Code/codex/zly_detection/实验表格.txt.
- Size-metric JSON root: /dpc/yuanxiangqing/projects/detection/reports/size_maps.
- Standard ultralytics DetectionValidator was patched so normal baseline validation now emits AP-S/AP-M/AP-L columns in results.csv and console output.
- RT-DETR-L formal runs are active: RGB PID 721425 on GPU0, Sonar PID 725173 on GPU1.
- SSD formal runs are active: RGB PID 728127 on GPU2, Sonar resumed PID 749613 on GPU3.
- Faster R-CNN formal runs are active after restart: RGB PID 749615 on GPU4, Sonar PID 749751 on GPU5.
- Faster R-CNN bs64 OOMed during smoke; bs2 smoke passed and is being used for formal runs.
- SSD/Faster R-CNN eval callbacks were patched so `.temp_map_out` is unique per run log directory; this fixed the concurrent RGB/Sonar temporary evaluation directory collision that stopped the original SSD Sonar process at epoch 9.
- Validation check: resumed SSD Sonar passed epoch 10 COCO eval with `Get map done` and continued to epoch 11, confirming the temp-dir patch works.

## Immediate next steps

- Monitor the six active non-YOLO jobs and update training_logs/README.md plus /Users/liam/Code/codex/zly_detection/实验表格.txt after metrics are available.
- Keep smoke logs/runs separate from formal logs/runs.
