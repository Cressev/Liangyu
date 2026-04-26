# Training Machine Environment Finding

- SSH target: `root@172.26.0.107`; local SSH alias observed: `zhipu-dev`.
- Hostname: `develop-20260423172203-u7ry4`.
- OS/kernel: Linux 5.15.0-113-generic x86_64.
- Internal IP observed on host: `10.184.112.176`; local machine had no direct route to it.
- GPUs: 8 x NVIDIA H100 80GB HBM3, driver 535.230.02.
- Memory: about 2.0 TiB total, about 1.9 TiB available at discovery time.
- Project root: `/dpc/yuanxiangqing/projects/detection`.
- `/dpc` is a shared scratch filesystem mounted from `10.218.1.211-218:/scratch`; capacity about 259T with about 2.6T free at discovery time.
- Remote local writes to `/dpc` tested fast (~981 MB/s for a 256MiB dd), so upload bottleneck is the network/SSH path from local Mac to training machine.
- Conda binary available at `/dpc/yuanxiangqing/envs/conda/miniconda3/condabin/conda`; conda version 26.1.1.
- Project environment target: `/dpc/yuanxiangqing/envs/conda/envs/Liangyu`.

## Environment configuration notes
- Installed `libgl1` via apt to satisfy `cv2`/`albumentations` runtime dependency on `libGL.so.1`.
- `mmdet==3.2.0` requires `mmcv>=2.0.0rc4,<2.2.0` and `mmengine>=0.7.1,<1.0.0` for full MMDetection support.
- Official OpenMMLab wheel index has `mmcv-2.2.0` for `cu118/torch2.2.0`, but that violates the `mmdet==3.2.0` `<2.2.0` constraint.
- Installed `mmengine==0.10.7` and tried official `mmcv==2.1.0` wheel from `cu118/torch2.1.0`; ordinary imports work, but `mmcv.ops.nms` fails under `torch==2.2.0+cu118` with an undefined-symbol ABI mismatch.
- System `nvcc` is CUDA 12.6 while PyTorch is built for CUDA 11.8, so source-building `mmcv==2.1.0` against this PyTorch may hit CUDA version mismatch unless a CUDA 11.8 toolkit is provided.
- Practical resolution chosen on 2026-04-26: installed official `mmcv==2.2.0` wheel for `cu118/torch2.2.0`, then backed up and patched `/dpc/yuanxiangqing/envs/conda/envs/Liangyu/lib/python3.10/site-packages/mmdet/__init__.py` to allow `mmcv<2.3.0` instead of `<2.2.0`. Validation passed for `mmcv`, `mmdet`, and `mmcv.ops.nms`. Backup file: `__init__.py.orig_mmcv_cap_2_2_0`.
