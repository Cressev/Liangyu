# Log

## 2026-04-26
- Initialized project memory structure under `/dpc/yuanxiangqing/projects/detection` using the project-memory-persistence rules.
- Recorded local and training-machine environment findings in `findings/`.
- Started upload of `MutilModel_423.tar.gz` using preserved prefix plus 4-way parallel `scp` chunks after slower rsync/single-stream SSH attempts.
- Accepted conda channel ToS and created `/dpc/yuanxiangqing/envs/conda/envs/Liangyu` with Python 3.10.3.
- Copied `requirements_liangyu.txt` to the project root and began installing PyTorch CUDA 11.8 dependencies.
- Completed environment installation in `/dpc/yuanxiangqing/envs/conda/envs/Liangyu`: Python 3.10.3, PyTorch 2.2.0+cu118, torchvision 0.17.0+cu118, torchaudio 2.2.0+cu118, and `requirements_liangyu.txt` via Aliyun PyPI mirror.
- Installed `libgl1`, `mmengine==0.10.7`, and `mmcv==2.1.0`; verified core imports. `mmcv.ops.nms` still fails because the available `mmcv==2.1.0` wheel is built for torch2.1 while the requested environment uses torch2.2.0+cu118.
- Tested `mmcv==2.2.0` for torch2.2/cu118: `mmcv.ops.nms` works, while `mmdet==3.2.0` blocks import via a version cap. Backed up and patched `mmdet/__init__.py` to allow `mmcv<2.3.0`; validation now passes for `mmcv`, `mmdet`, and `mmcv.ops.nms`.
- Completed upload assembly for `MutilModel_423.tar.gz`; verified exact size 5,874,223,756 bytes and `gzip -t` passed.
- Extracted archive into `/dpc/yuanxiangqing/projects/detection/MutilModel_423` with 37,374 files. Removed temporary `MutilModel_423.parts/` and `MutilModel_423.tar.gz.prefix`; retained final archive.
- Inspected `trainMM.py` for YOLO11 small baselines. Fixed hard-coded dataset paths and parameterized the script for RGB/sonar runs. RGB smoke test passed. Sonar smoke test initially found zero labels because `images_ir/*` did not match the standard label layout; corrected `data_sonar.yaml` to use `sonar/train/images` and `sonar/val/images`, then sonar smoke test passed.
- Launched YOLO11 small RGB and sonar full baselines for 350 epochs. The first launch put both jobs on physical GPU0 despite `--device 0/1`, so the sonar job PID `488568` was stopped.
- Patched `trainMM.py` to stop setting `CUDA_VISIBLE_DEVICES` internally after imports. For parallel single-GPU launches, use external GPU isolation such as `CUDA_VISIBLE_DEVICES=1` with `--device 0`.
- Relaunched sonar baseline on physical GPU1 with PID `492632`. Current active jobs: RGB PID `488567` on GPU0 with log `/dpc/yuanxiangqing/projects/detection/training_logs/yolo11s_rgb_20260426_072810.log`; sonar PID `492632` on GPU1 with log `/dpc/yuanxiangqing/projects/detection/training_logs/yolo11s_sonar_gpu1_20260426_073246.log`.
- Switched baseline batch size from 36 to 32 on request, then tuned for higher H100 memory utilization. Batch-size probes showed approximate training memory peaks: bs96 45.7GB, bs112 53.2GB, bs128 58.8GB, bs144 68.5GB. Selected bs144 for formal runs because it exceeds 80% of 80GB H100 memory without OOM.
- Stopped preliminary bs36 and bs32 jobs. Launched formal YOLO11 small bs144 baselines: RGB PID `505211`, GPU0, log `/dpc/yuanxiangqing/projects/detection/training_logs/yolo11s_rgb_bs144_20260426_074605.log`; sonar PID `505212`, GPU1, log `/dpc/yuanxiangqing/projects/detection/training_logs/yolo11s_sonar_bs144_20260426_074605.log`.
- Created `/dpc/yuanxiangqing/projects/detection/training_logs/README.md` with a table mapping formal experiment groups to log files, output directories, PIDs, and GPU bindings.
- Replaced project ultralytics package with /Users/liam/Downloads/ultralytics.zip (remote backup: ultralytics.backup_before_zip_20260426_084448). Verified epoch=1 YOLO11s RGB bs64 smoke passed. Replacement package still writes mAP50/mAP75/mAP50-95 and logs Params/GFLOPs/FPS, but does not add mapS/mapM/mapL columns to results.csv.
- Stopped previous bs144 jobs. Launched first YOLO bs64 wave after per-experiment epoch=1 smoke tests: YOLOv5s/YOLOv8s/YOLOv9s/YOLOv10s on RGB and Sonar, GPUs 0-7. Logs and PIDs are recorded in /dpc/yuanxiangqing/projects/detection/training_logs/README.md. Pending: YOLOv11s/YOLOv12s/YOLOv26s RGB and Sonar.
- Completed all YOLO bs64 baseline experiments (YOLOv5s/v8s/v9s/v10s/v11s/v12s/v26s on RGB and Sonar). Updated local experiment table at /Users/liam/Code/codex/zly_detection/实验表格.txt. All remote GPUs are idle after completion. Replacement ultralytics.zip did not output mapS/mapM/mapL, so those table cells remain blank with notes.

- Verified uploaded `/Users/liam/Downloads/ultralytics.zip` contains size-specific COCO metrics as `APsmall/APmedium/APlarge` in `ultralytics/utils/coco_metrics.py` and `ultralytics/utils/coco_eval_bbox_mm.py`. Recomputed all 14 YOLO bs64 size metrics using `COCOevalBBoxMM` from the uploaded package, saved JSON outputs under `/dpc/yuanxiangqing/projects/detection/reports/size_maps/`, and regenerated `/Users/liam/Code/codex/zly_detection/实验表格.txt` with mapS/mapM/mapL populated from those JSON files.
- Updated the experiment table to use a single AP metric source for all AP columns: `AP50`, `AP75`, `AP`, `APsmall`, `APmedium`, and `APlarge` from the uploaded zip packages on each experiments `best.pt`.
- Updated the experiment table to use a single AP metric source for all AP columns: AP50, AP75, AP, APsmall, APmedium, and APlarge from the uploaded zip package's COCOevalBBoxMM on each experiment's best.pt.
- Patched standard `/dpc/yuanxiangqing/projects/detection/MutilModel_423/ultralytics/models/yolo/detect/val.py` so regular detection validation records GT areas, matches predictions to GT areas, and emits `metrics/mAP50-95(S)`, `metrics/mAP50-95(M)`, and `metrics/mAP50-95(L)` in `results.csv` and console output. Smoke test `detect-size-maps-yolo11s-rgb` passed with output under `/dpc/yuanxiangqing/projects/detection/MutilModel_423/run_smoke/detect-size-maps-yolo11s-rgb` and log `/dpc/yuanxiangqing/projects/detection/training_logs/smoke/detect-size-maps-yolo11s-rgb_20260426.log`.
- Parameterized `trainRT.py`; RT-DETR-L RGB and Sonar smoke tests passed at bs64 with size metrics. Launched formal RT-DETR-L RGB PID `721425` on GPU0 and Sonar PID `725173` on GPU1. Logs are `/dpc/yuanxiangqing/projects/detection/training_logs/rtdetr-l-rgb-bs64_20260426_134624.log` and `/dpc/yuanxiangqing/projects/detection/training_logs/rtdetr-l-sonar-bs64_20260426_135001.log`; outputs are under `/dpc/yuanxiangqing/projects/detection/MutilModel_423/run_non_yolo_bs64`.
- Converted YOLO labels to legacy VOC-style annotation files under `/dpc/yuanxiangqing/projects/detection/legacy_annotations/{rgb,sonar}` for SSD and Faster R-CNN, each with 4000 train lines, 1000 val lines, and 16 classes.
- Parameterized `ssd-pytorch-master/train.py`; SSD RGB smoke passed at bs64 and COCO eval printed AP small/medium/large. Launched formal SSD RGB PID `728127` on GPU2 and SSD Sonar PID `728131` on GPU3. Logs are `/dpc/yuanxiangqing/projects/detection/training_logs/ssd-rgb-bs64_20260426_135236.log` and `/dpc/yuanxiangqing/projects/detection/training_logs/ssd-sonar-bs64_20260426_135236.log`; outputs are under `/dpc/yuanxiangqing/projects/detection/MutilModel_423/run_non_yolo_legacy`.
- Parameterized `faster-rcnn-pytorch-master/train.py`. A first bs64-style smoke collided/OOMed due to the script overwriting `CUDA_VISIBLE_DEVICES`; corrected launch control via `FRCNN_GPUS`. Faster R-CNN RGB bs2 smoke passed and printed AP small/medium/large. Launched formal Faster R-CNN RGB PID `740252` on GPU4 and Sonar PID `740320` on GPU5, with logs `/dpc/yuanxiangqing/projects/detection/training_logs/frcnn-rgb-bs2_20260426_140017.log` and `/dpc/yuanxiangqing/projects/detection/training_logs/frcnn-sonar-bs2_20260426_140018.log`; outputs are under `/dpc/yuanxiangqing/projects/detection/MutilModel_423/run_non_yolo_legacy`.
- Found SSD Sonar PID `728131` had exited at epoch 9 because SSD RGB and SSD Sonar were sharing `.temp_map_out` in the same working directory during COCO eval cleanup (`OSError: Directory not empty: detection-results`). Patched both `ssd-pytorch-master/utils/callbacks.py` and `faster-rcnn-pytorch-master/utils/callbacks.py` so default eval temp directories live under each run's `loss_*` log directory and cleanup uses `ignore_errors=True`. Also added `SSD_INIT_EPOCH` and `FRCNN_INIT_EPOCH` environment controls.
- Resumed SSD Sonar from `/dpc/yuanxiangqing/projects/detection/MutilModel_423/run_non_yolo_legacy/ssd-sonar-bs64/last_epoch_weights.pth` with `SSD_INIT_EPOCH=9`; new PID `749613`, GPU3, log `/dpc/yuanxiangqing/projects/detection/training_logs/ssd-sonar-bs64-resume-e9_20260426_140606.log`.
- Stopped original Faster R-CNN PIDs `740252` and `740320` before their first scheduled epoch-10 eval because they had the same shared-temp-dir risk. Restarted Faster R-CNN RGB PID `749615` on GPU4 with log `/dpc/yuanxiangqing/projects/detection/training_logs/frcnn-rgb-bs2-restart_20260426_140606.log` and Sonar PID `749751` on GPU5 with log `/dpc/yuanxiangqing/projects/detection/training_logs/frcnn-sonar-bs2-restart_20260426_140607.log`.
- Confirmed the resumed SSD Sonar run passed the previously failing epoch-10 COCO eval: log shows `Get map done`, `Epoch:10/350`, best weight save, and continuation to epoch 11. Active formal jobs after the fix: RT-DETR RGB/Sonar PIDs `721425`/`725173`, SSD RGB/Sonar PIDs `728127`/`749613`, Faster R-CNN RGB/Sonar PIDs `749615`/`749751`.
- Added `/dpc/yuanxiangqing/projects/detection/training_logs/TRAINING_LAUNCH_README.md`, documenting the exact smoke, formal, resume, and monitoring commands for YOLO, RT-DETR, SSD, and Faster R-CNN. Updated the training log index to point to it.

## 2026-04-26 15:26 GitHub upload snapshot
- Pushed the source/docs/log index snapshot to GitHub repo  on branch , commit .
- Started background GitHub Release asset upload for the current remote project snapshot, including weights, logs, run outputs, extracted data, and project memory.
- Release upload PID: ; log: .
- Excluded duplicate source archives from the release tar split:  and .

## 2026-04-26 15:32 GitHub upload memory correction
- Correction for the previous GitHub upload note: source/docs/log indexes were pushed to GitHub repo Cressev/Liangyu on branch main, commit 9e802f5.
- Large project snapshot upload is running as GitHub Release asset upload PID 864878.
- Upload log: /dpc/yuanxiangqing/projects/detection/training_logs/github_release_upload_20260426_152626.log.
- The release snapshot excludes duplicate archives /dpc/yuanxiangqing/projects/detection/MutilModel_423.tar.gz and MutilModel_423/数据集.zip, while keeping extracted project files, weights, logs, run outputs, datasets, and memory files.

## 2026-04-27 Training and GitHub status update
- GitHub code branch main is pushed through commit 93e7857.
- GitHub Release project-snapshot-20260426_152626 completed split asset upload at 2026-04-26 20:00:20 CST. Assets include SHA256SUMS.txt and seven tar parts aa-ag.
- RT-DETR RGB and Sonar runs finished early at epochs 321 and 322; final metrics were copied into /Users/liam/Code/codex/zly_detection/实验表格.txt.
- SSD RGB and Sonar completed 350/350; VOC mAP50 and model complexity were copied into the experiment table. Legacy SSD eval did not output map75/map50-95/S/M/L/FPS.
- Faster R-CNN RGB and Sonar are still running on PIDs 749615 and 749751, around epochs 205/350 and 207/350 at the time of this update.

## 2026-04-27 SSD full metric correction
- Corrected an experiment-table gap: the previous SSD smoke check verified that training/eval ran, but did not verify that all requested table metrics were emitted. This was a process mistake.
- Added post-eval script `/dpc/yuanxiangqing/projects/detection/MutilModel_423/eval_legacy_detector_coco_metrics.py` to recompute COCO-style `map50`, `map75`, `map50_95`, `mapS`, `mapM`, `mapL` from legacy SSD/FRCNN detection outputs.
- Validated the script first on SSD RGB with `--limit 20`, then ran full SSD RGB/Sonar validation on 1000 images each using best_epoch_weights.pth.
- SSD RGB full metrics: map50=0.56934, map75=0.21113, map50_95=0.26360, mapS=0.01989, mapM=0.31935, mapL=0.46855, inference FPS=47.83.
- SSD Sonar full metrics: map50=0.30768, map75=0.12693, map50_95=0.15035, mapS=0.00074, mapM=0.15461, mapL=0.40296, inference FPS=50.06.
- Updated experiment tables at `/Users/liam/Code/codex/zly_detection/实验表格.md`, `/Users/liam/Code/codex/zly_detection/实验表格.txt`, and remote copies `/dpc/yuanxiangqing/projects/detection/实验表格.md` and `/dpc/yuanxiangqing/projects/detection/实验表格.txt`.
- Eval logs: `/dpc/yuanxiangqing/projects/detection/training_logs/full_metric_eval/ssd-rgb-full-metrics_20260427.log` and `ssd-sonar-full-metrics_20260427.log`.

## 2026-04-27 FRCNN table completion
- FRCNN formal runs did not finish normally: both RGB and Sonar interrupted around epoch 219/350 during eval because /dpc returned `No space left on device` while writing `.temp_map_out` files.
- Completed the experiment table by running full COCO-style post-eval from best_epoch_weights.pth into /tmp outputs.
- FRCNN RGB metrics: map50=0.62547, map75=0.19370, map50_95=0.27184, mapS=0.04551, mapM=0.29810, mapL=0.46332, inference FPS=30.53.
- FRCNN Sonar metrics: map50=0.58072, map75=0.16552, map50_95=0.24054, mapS=0.03042, mapM=0.26743, mapL=0.38335, inference FPS=31.28.
- Updated experiment tables locally and remotely: /Users/liam/Code/codex/zly_detection/实验表格.txt, /Users/liam/Code/codex/zly_detection/实验表格.md, /dpc/yuanxiangqing/projects/detection/实验表格.txt, /dpc/yuanxiangqing/projects/detection/实验表格.md.

## 2026-04-27 Storage recheck
- Rechecked `/dpc` storage after the user questioned the 200T-capacity behavior.
- `/dpc` is Lustre shared scratch, about 259T total, 251T used, 6.2T available, 97-98% full; inode use is only about 20%.
- `/dpc/yuanxiangqing` visible usage is hundreds of GB rather than hundreds of TB: `projects` about 219G, `model` about 94G, `envs/conda` about 59G, `envs/uv` about 11G, `.cache` about 8.9G.
- The detection project is about 18G, with `MutilModel_423/run_non_yolo_legacy` about 12G and `training_logs` about 308M. This project is not the source of the 251T global filesystem usage.
- A write test under `/dpc/yuanxiangqing/projects/detection` succeeded during the check. Main conclusion: previous `No space left on device` behavior is consistent with a nearly-full shared Lustre filesystem or quota/backend pressure, not the detection experiment directory consuming 200T.
- Detailed storage notes were appended to `findings/training_machine_environment.md`.

## 2026-04-27 YOLO full-metrics rerun started
- Started a new YOLO rerun batch after patching training-time validation to emit size-specific metrics in normal `results.csv` output.
- New smoke/formal separation: smoke logs under `training_logs/smoke_yolo_fullmetrics_20260427_r2`, formal logs and launch table under `training_logs/yolo_fullmetrics_20260427_r2`, smoke runs under `MutilModel_423/run_smoke_yolo_fullmetrics_20260427_r2`, and formal runs under `MutilModel_423/run_yolo_bs64_fullmetrics_20260427_r2`.
- Initial scheduler successfully launched YOLOv5s RGB/Sonar, YOLOv8s RGB/Sonar, and YOLOv10s RGB/Sonar after `epochs=1` smoke tests verified `metrics/mAP50-95(S/M/L)` columns.
- Found and corrected two launch issues: GPU assignment needed to avoid occupied GPUs after smoke failures, and YOLOv9 cannot receive `scale=s` because `yolov9s.yaml` has no `scales:` block.
- Patched `MutilModel_423/trainMM.py` so `scale` is passed only when the model YAML defines `scales:`.
- Restarted continuation scheduling with `continue_yolo_fullmetrics_20260427_r2_part2.sh`. YOLOv9 RGB smoke passed after the patch and formal training launched. The part2 scheduler remains active to queue YOLOv9 Sonar, YOLOv11 Sonar, YOLOv12 RGB/Sonar, and YOLOv26 RGB/Sonar as GPUs free up.
- Added explicit `--resume/--no-resume` support to `MutilModel_423/trainMM.py`. To resume a YOLO run, launch with `--model <run_dir>/weights/last.pt --resume --device 0` under the same project environment; the Ultralytics backend restores checkpoint args and continues from the saved epoch.

## 2026-04-27 YOLO rerun paused for GPU release
- Stopped all active training processes from YOLO full-metrics rerun batch `20260427_r2` at user request so the GPUs can be used by others. `nvidia-smi` showed all 8 GPUs clear after stopping.
- Completed normally before pause: YOLOv5s RGB/Sonar, YOLOv8s RGB/Sonar, YOLOv9s RGB/Sonar, YOLOv10s RGB/Sonar, YOLOv11s RGB/Sonar.
- Paused before normal completion: YOLOv12s RGB, YOLOv12s Sonar, YOLOv26s RGB, YOLOv26s Sonar. Their `last.pt` checkpoints are available under `/dpc/yuanxiangqing/projects/detection/MutilModel_423/run_yolo_bs64_fullmetrics_20260427_r2/<run>/weights/last.pt`.
- Created a new completed-only experiment table distinct from previous tables: `/dpc/yuanxiangqing/projects/detection/实验表格_yolo_fullmetrics_20260427_r2_已完成.md` and `.txt`; synced local copies to `/Users/liam/Code/codex/zly_detection/`.
- Created resume/archive file: `/dpc/yuanxiangqing/projects/detection/实验存档_yolo_fullmetrics_20260427_r2.md` and `/dpc/yuanxiangqing/projects/detection/reports/yolo_fullmetrics_20260427_r2_experiment_archive.md`; synced local copy to `/Users/liam/Code/codex/zly_detection/实验存档_yolo_fullmetrics_20260427_r2.md`.

## 2026-04-27 YOLO r2 resume restarted
- GPU state at start: GPU 0-5 had residual memory with no listed process; GPU 6-7 were free.
- Resumed paused YOLOv12 RGB/Sonar on physical GPU 6/7 using last.pt checkpoints.
- Fixed resume compatibility: trainMM.py drops checkpoint model_scale override; ultralytics/engine/model.py clears model_scale after trainer restore during resume.
- Scheduler path: /dpc/yuanxiangqing/projects/detection/resume_yolo_paused_20260427_r2.sh
- Logs: /dpc/yuanxiangqing/projects/detection/training_logs/yolo_fullmetrics_20260427_r2/resume_20260427_152217_*.log

## 2026-04-27 YOLOv26 remaining jobs resumed
- User asked whether training ended. Check showed YOLOv12 RGB/Sonar ended normally by EarlyStopping, but YOLOv26 RGB/Sonar had not run because the previous queue function exited after the first job per GPU.
- Created and launched `/dpc/yuanxiangqing/projects/detection/resume_yolo26_remaining_20260427_r2.sh` to resume only remaining YOLOv26 jobs on physical GPUs 6 and 7.
- Scheduler PID: 1799511.
- Logs: `/dpc/yuanxiangqing/projects/detection/training_logs/yolo_fullmetrics_20260427_r2/resume_20260427_154839_yolov26s-*.log`.

## 2026-04-28 YOLO r2 final completion confirmed
- No active training or scheduler processes remain for `20260427_r2`.
- GPU 6/7 are free; GPU 0-5 still show residual memory with no listed process.
- YOLOv26 RGB finished normally at epoch 350, best epoch 334.
- YOLOv26 Sonar finished normally by EarlyStopping, results file last epoch 235; log reports successful rc=0.
- Regenerated final completed table and archive: completed_count=14, paused_count=0.
