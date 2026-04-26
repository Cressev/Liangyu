# Detection Project Agent Rules

## Canonical paths
- Remote project root: `/dpc/yuanxiangqing/projects/detection`.
- Conda environments root: `/dpc/yuanxiangqing/envs/conda/envs`.
- Active environment for this project: `/dpc/yuanxiangqing/envs/conda/envs/Liangyu`.
- Source package being uploaded: local `/Users/liam/Downloads/MutilModel_423`; remote archive `/dpc/yuanxiangqing/projects/detection/MutilModel_423.tar.gz`.

## Operating rules
- Read `CURRENT.md`, `AGENTS.md`, `PROGRESS.md`, relevant `README.md` files, and recent `log.md` entries before substantial work.
- Keep durable environment/workflow discoveries in `findings/` and update `findings/README.md` when adding files.
- Keep chronological actions, decisions, validations, and artifact paths in `log.md`.
- Do not remove partial upload artifacts until the assembled archive passes size and gzip validation.
- Prefer resumable or chunked transfers for multi-GB assets over single fragile streams.

## User preferences
- Use pip mirror `-i https://mirrors.aliyun.com/pypi/simple` for PyPI packages.
- Keep long uploads running in the background while investigating faster options or configuring the environment.
