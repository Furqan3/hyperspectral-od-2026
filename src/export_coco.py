"""Export fold splits as COCO json (for DETR-family repos such as D-FINE / DEIM).

Boxes come from the VOC XMLs, clipped to the cube and with zero-size boxes dropped, the same
as the YOLO labels. Image file names point at the prepared 16-band TIFFs.

  python src/export_coco.py --fold 0     # -> data/prepared/coco/fold0_{train,val}.json
  python src/export_coco.py --fold all   # -> data/prepared/coco/all_train.json
"""
import argparse
import json
import os

import cv2

from common import CLASSES, PREP, TRAIN_XML, parse_voc


def to_coco(paths):
    images, anns = [], []
    for p in paths:
        iid = int(os.path.basename(p)[:-5])
        ok, frames = cv2.imreadmulti(p, flags=cv2.IMREAD_UNCHANGED)
        H, W = frames[0].shape[:2]
        images.append({"id": iid, "file_name": os.path.basename(p), "width": W, "height": H})
        for c, x1, y1, x2, y2 in parse_voc(os.path.join(TRAIN_XML, f"{iid}.xml"))[2]:
            x1, x2 = max(0.0, min(x1, W)), max(0.0, min(x2, W))
            y1, y2 = max(0.0, min(y1, H)), max(0.0, min(y2, H))
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            anns.append({"id": len(anns) + 1, "image_id": iid, "category_id": c, "bbox": [x1, y1, x2 - x1, y2 - y1],
                         "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
    return {"images": images, "annotations": anns,
            "categories": [{"id": i, "name": n} for i, n in enumerate(CLASSES)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="0")
    a = ap.parse_args()
    out = os.path.join(PREP, "coco")
    os.makedirs(out, exist_ok=True)
    splits = ["train"] if a.fold == "all" else ["train", "val"]
    base = "all" if a.fold == "all" else f"fold{a.fold}"
    for s in splits:
        paths = [l.strip() for l in open(os.path.join(PREP, f"{base}_{s}.txt")) if l.strip()]
        d = to_coco(paths)
        path = os.path.join(out, f"{base}_{s}.json")
        json.dump(d, open(path, "w"))
        print(f"{path}: {len(d['images'])} images, {len(d['annotations'])} boxes")


if __name__ == "__main__":
    main()
