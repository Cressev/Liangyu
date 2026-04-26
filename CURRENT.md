# Current State

Last updated: 2026-04-26 15:32 CST

## Active Work
- Formal baseline training jobs are still running on the training machine under /dpc/yuanxiangqing/projects/detection/MutilModel_423.
- Code/docs/log indexes have been pushed to GitHub repo Cressev/Liangyu, branch main, commit 9e802f5.
- Large project snapshot upload is running as a GitHub Release asset upload in the background.

## GitHub Upload
- Release upload PID: 864878.
- Upload log: /dpc/yuanxiangqing/projects/detection/training_logs/github_release_upload_20260426_152626.log.
- Snapshot source: /dpc/yuanxiangqing/projects/detection.
- Snapshot excludes duplicate archives MutilModel_423.tar.gz and MutilModel_423/数据集.zip, but includes extracted project files, weights, logs, run outputs, datasets, and memory files.

## Next Steps
- Monitor github_release_upload_20260426_152626.log until the release URL and all split asset uploads finish.
- Keep monitoring the active baseline training jobs and fill the experiment table as runs complete.
- Revoke or rotate the GitHub token after upload is complete.
