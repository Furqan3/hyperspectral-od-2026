"""Local COCO-style evaluation matching the competition metric.

mAP@[0.5:0.95] (primary) and mAP@0.5 (secondary) from a submission-format
DataFrame/CSV against the VOC ground truth of the given image ids.
"""
import contextlib
import io
import os

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from common import CLASSES, TRAIN_XML, parse_voc


def build_coco_gt(image_ids):
    images, anns = [], []
    for iid in image_ids:
        w, h, boxes = parse_voc(os.path.join(TRAIN_XML, f"{iid}.xml"))
        images.append({"id": int(iid), "width": w, "height": h})
        for c, x1, y1, x2, y2 in boxes:
            anns.append({"id": len(anns) + 1, "image_id": int(iid), "category_id": c,
                         "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
    gt = COCO()
    gt.dataset = {"images": images, "annotations": anns,
                  "categories": [{"id": i, "name": n} for i, n in enumerate(CLASSES)]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
    return gt


def evaluate(pred, image_ids, per_class=False):
    """pred: DataFrame or CSV path with image_id,class_id,confidence,x1,y1,x2,y2."""
    if isinstance(pred, str):
        pred = pd.read_csv(pred)
    image_ids = [int(i) for i in image_ids]
    gt = build_coco_gt(image_ids)
    pred = pred[pred.image_id.isin(image_ids)]
    dets = [{"image_id": int(r.image_id), "category_id": int(r.class_id), "score": float(r.confidence),
             "bbox": [r.x1, r.y1, r.x2 - r.x1, r.y2 - r.y1]} for r in pred.itertuples()]
    if not dets:
        out = dict.fromkeys(["mAP50_95", "mAP50", "mAP75", "mAP_small", "mAP_medium"], 0.0)
        if per_class:
            out["per_class_AP50_95"] = dict.fromkeys(CLASSES, 0.0)
        return out
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(dets)
        ev = COCOeval(gt, dt, "bbox")
        ev.params.maxDets = [1, 10, 300]  # no 100-per-image cap; Kaggle scores every row
        ev.evaluate(); ev.accumulate()
    # computed from the precision array directly: COCOeval.summarize() hardcodes maxDets=100 for mAP50_95
    prec = ev.eval["precision"]  # [T(iou), R(recall), K(class), A(area: all/small/medium/large), M(maxDets)]

    def mean_ap(p):
        return float(p[p > -1].mean()) if (p > -1).any() else float("nan")

    out = {"mAP50_95": mean_ap(prec[:, :, :, 0, -1]), "mAP50": mean_ap(prec[0, :, :, 0, -1]),
           "mAP75": mean_ap(prec[5, :, :, 0, -1]),
           "mAP_small": mean_ap(prec[:, :, :, 1, -1]), "mAP_medium": mean_ap(prec[:, :, :, 2, -1])}
    if per_class:
        ap = {}
        for k, name in enumerate(CLASSES):
            p = prec[:, :, k, 0, -1]
            ap[name] = float(p[p > -1].mean()) if (p > -1).any() else float("nan")
        out["per_class_AP50_95"] = ap
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_csv")
    ap.add_argument("--ids", required=True, help="text file with one image id per line")
    a = ap.parse_args()
    ids = [l.strip() for l in open(a.ids) if l.strip()]
    print(evaluate(a.pred_csv, ids, per_class=True))
