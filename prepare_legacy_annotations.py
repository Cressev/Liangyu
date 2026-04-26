#!/usr/bin/env python3
import argparse
from pathlib import Path

import yaml
from PIL import Image


def image_files(root):
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        files.extend(root.rglob(ext))
    return sorted(files)


def convert_split(root, image_rel, out_path):
    image_dir = root / image_rel
    lines = []
    for img_path in image_files(image_dir):
        label_path = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
        if not label_path.exists():
            continue
        with Image.open(img_path) as im:
            width, height = im.size
        boxes = []
        for raw in label_path.read_text().splitlines():
            parts = raw.strip().split()
            if len(parts) < 5:
                continue
            cls, xc, yc, bw, bh = map(float, parts[:5])
            x1 = max(0, min(width - 1, int(round((xc - bw / 2) * width))))
            y1 = max(0, min(height - 1, int(round((yc - bh / 2) * height))))
            x2 = max(0, min(width - 1, int(round((xc + bw / 2) * width))))
            y2 = max(0, min(height - 1, int(round((yc + bh / 2) * height))))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append(f"{x1},{y1},{x2},{y2},{int(cls)}")
        if boxes:
            lines.append(str(img_path.resolve()) + " " + " ".join(boxes))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return len(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    data = yaml.safe_load(Path(args.data).read_text())
    root = Path(data["path"])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = data["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    (out / "classes.txt").write_text("\n".join(names) + "\n")
    train_count = convert_split(root, data["train"].replace("./", ""), out / "train.txt")
    val_count = convert_split(root, data["val"].replace("./", ""), out / "val.txt")
    print(f"wrote {train_count} train and {val_count} val lines to {out}")


if __name__ == "__main__":
    main()
