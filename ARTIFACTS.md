# Large Artifacts

The git repository contains source code, configuration, project memory, launch documentation, experiment indexes, and text logs that are small enough for normal Git history.

Large datasets, model weights, run directories, and the full project snapshot are uploaded as GitHub Release assets. Reassemble split snapshot parts with:

```bash
cat detection_project_snapshot_*.tar.part_* > detection_project_snapshot.tar
tar -xf detection_project_snapshot.tar
```

The full snapshot excludes duplicate source archives such as the original uploaded `MutilModel_423.tar.gz` and duplicate dataset zip when the extracted project files are already present.

