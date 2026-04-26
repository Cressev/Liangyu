#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import cv2
import yaml


PROJECT = Path("/dpc/yuanxiangqing/projects/detection/MutilModel_423")
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.coco_eval_bbox_mm import COCOevalBBoxMM  # noqa: E402


def load_dataset(data_yaml):
    data = yaml.safe_load(Path(data_yaml).read_text())
    root = Path(data["path"])
    val_images = root / data["val"].replace("./", "")
    names = data["names"]
    images = sorted([p for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp") for p in val_images.rglob(ext)])
    return images, len(names)


def labels_for_image(image_path):
    label_path = Path(str(image_path).replace("/images/", "/labels/")).with_suffix(".txt")
    if not label_path.exists():
        return []
    rows = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            rows.append([float(x) for x in parts[:5]])
    return rows


def yolo_xywhn_to_coco_xywh(label, width, height):
    cls, x, y, w, h = label
    bw, bh = w * width, h * height
    cx, cy = x * width, y * height
    return int(cls), [float(cx - bw / 2), float(cy - bh / 2), float(bw), float(bh)]


def xyxy_to_coco_xywh(box):
    x1, y1, x2, y2 = [float(v) for v in box]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def evaluate(args):
    images, nc = load_dataset(args.data)
    model = YOLO(args.weights)
    gts, dts, img_ids = [], [], []

    results = model.predict(
        source=[str(p) for p in images],
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        verbose=False,
        stream=True,
    )

    gt_id = 1
    for image_id, (image_path, result) in enumerate(zip(images, results)):
        img_ids.append(image_id)
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        height, width = img.shape[:2]

        for label in labels_for_image(image_path):
            cls, bbox = yolo_xywhn_to_coco_xywh(label, width, height)
            area = max(0.0, bbox[2]) * max(0.0, bbox[3])
            gts.append(
                {
                    "id": gt_id,
                    "image_id": image_id,
                    "category_id": cls,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "ignore": 0,
                }
            )
            gt_id += 1

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        pred_boxes = boxes.xyxy.detach().cpu().numpy()
        pred_conf = boxes.conf.detach().cpu().numpy()
        pred_cls = boxes.cls.detach().cpu().numpy().astype(int)
        for box, score, cls in zip(pred_boxes, pred_conf, pred_cls):
            bbox = xyxy_to_coco_xywh(box)
            dts.append(
                {
                    "image_id": image_id,
                    "category_id": int(cls),
                    "bbox": bbox,
                    "area": max(0.0, bbox[2]) * max(0.0, bbox[3]),
                    "score": float(score),
                }
            )

    cat_ids = sorted(set(range(nc)) | {g["category_id"] for g in gts} | {d["category_id"] for d in dts})
    evaluator = COCOevalBBoxMM()
    evaluator.set_data(gts=gts, dts=dts, imgIds=img_ids, catIds=cat_ids)
    evaluator.evaluate()
    evaluator.accumulate()
    stats = evaluator.summarize()
    stats = {k: (0.0 if v == -1 else float(v)) for k, v in stats.items()}
    out = {
        "mapS": stats.get("APsmall", 0.0),
        "mapM": stats.get("APmedium", 0.0),
        "mapL": stats.get("APlarge", 0.0),
        "AP": stats.get("AP", 0.0),
        "AP50": stats.get("AP50", 0.0),
        "AP75": stats.get("AP75", 0.0),
        "source": "ultralytics.utils.coco_eval_bbox_mm.COCOevalBBoxMM from uploaded ultralytics.zip",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(out, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--modality", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
