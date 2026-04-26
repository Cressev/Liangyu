# Progress

- Confirmed local source directory `/Users/liam/Downloads/MutilModel_423` is about 5.9G with 37k+ files.
- Direct `rsync` and single-stream SSH upload were slow; switched to tar/gzip archive and then to preserved-prefix plus parallel chunk upload.
- Created local archive `/Users/liam/Downloads/MutilModel_423.tar.gz` with size 5,874,223,756 bytes.
- Accepted conda channel ToS on training machine and created `/dpc/yuanxiangqing/envs/conda/envs/Liangyu` with Python 3.10.3.
- Copied `requirements_liangyu.txt` into the project root.
