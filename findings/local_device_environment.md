# Local Device Environment Finding

- Local host: macOS 26.4.1, Darwin 25.4.0, arm64.
- CPU string observed: Apple M5.
- Memory observed via `hw.memsize`: 17,179,869,184 bytes (~16 GiB).
- Local workspace: `/Users/liam/Code/codex`.
- Source asset directory: `/Users/liam/Downloads/MutilModel_423`, about 5.9G and 37,374 files.
- Local archives created:
  - `/Users/liam/Downloads/MutilModel_423.tar` (~5.8G)
  - `/Users/liam/Downloads/MutilModel_423.tar.gz`, 5,874,223,756 bytes (~5.5G)
- Local IPs observed: `en0` 192.168.31.115, `utun4` 28.0.0.1, `utun5` 192.168.113.53.
- Training machine could not HTTP-pull from these local IPs during testing; direct temporary TCP ports were also not reachable.
