"""Run a trained model on prepared TIFFs and write a Kaggle submission CSV.

Examples:
  # out-of-fold validation (scores with the local evaluator)
  python src/predict.py runs/fold0/weights/best.pt --split train --ids data/prepared/fold0_val.txt --eval
  # Phase 1 submission (test only)
  python src/predict.py runs/all/weights/best.pt --split test --out submissions/sub_test.csv
  # Phase 2 submission (test + ranking in ONE csv)
  python src/predict.py runs/all/weights/best.pt --split test ranking --out submissions/sub_final.csv
"""
import argparse
import glob
import os

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import RTDETR, YOLO

from common import CLASSES, PREP

COLS = ["id", "image_id", "class_id", "confidence", "x1", "y1", "x2", "y2"]


def read_tiff(path):
    ok, frames = cv2.imreadmulti(path, flags=cv2.IMREAD_UNCHANGED)
    assert ok, path
    return np.stack(frames, axis=2)


def predict(weights, paths, imgsz=1024, conf=0.001, iou=0.6, max_det=300, batch=16, device=None, round_px=True):
    """round_px: snap box coordinates to whole pixels. The ground truth is integer VOC coordinates,
    and snapping raised OOF mAP@[.5:.95] by +0.004..+0.005 on three models (fold 0)."""
    # RT-DETR checkpoints need the RTDETR wrapper (its own predictor, no NMS)
    arch = type(torch.load(weights, map_location="cpu", weights_only=False)["model"]).__name__
    model = (RTDETR if "RTDETR" in arch else YOLO)(weights)
    rows = []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        # Arrays, not paths: ultralytics' batched file loader drops to 3 channels (8.4.147), the numpy loader keeps 16.
        ims = [read_tiff(p) for p in chunk]
        results = model.predict(ims, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, device=device, verbose=False)
        for p, r in zip(chunk, results):
            iid = int(os.path.splitext(os.path.basename(p))[0])
            h, w = r.orig_shape
            b = r.boxes
            for (x1, y1, x2, y2), c, s in zip(b.xyxy.tolist(), b.cls.tolist(), b.conf.tolist()):
                nd = None if round_px else 2  # round(x, None) -> int, written as 486 rather than 486.0
                x1, y1 = round(max(0.0, x1), nd), round(max(0.0, y1), nd)
                x2, y2 = round(min(w, x2), nd), round(min(h, y2), nd)
                if x2 <= x1 or y2 <= y1:  # fully outside the image after clipping
                    continue
                rows.append((iid, int(c), round(float(s), 5), x1, y1, x2, y2))
    df = pd.DataFrame(rows, columns=COLS[1:])
    df.insert(0, "id", range(len(df)))
    return df


def check_submission(df, expected_ids):
    assert list(df.columns) == COLS, df.columns
    assert df["id"].is_unique and (df["id"].values == range(len(df))).all()
    assert df.class_id.between(0, len(CLASSES) - 1).all()
    assert df.confidence.between(0, 1).all()
    assert ((df.x2 > df.x1) & (df.y2 > df.y1)).all()
    missing = set(map(int, expected_ids)) - set(df.image_id)
    print(f"rows={len(df)} images={df.image_id.nunique()}/{len(expected_ids)} "
          f"(no detections: {len(missing)}) mean dets/img={len(df) / max(1, df.image_id.nunique()):.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--split", nargs="+", default=["test"], help="train | test | ranking (several = one CSV)")
    ap.add_argument("--ids", help="file with image paths/ids to restrict to (e.g. fold0_val.txt)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--eval", action="store_true", help="score against VOC ground truth (train split only)")
    ap.add_argument("--metrics-out", default=None, help="write --eval metrics to this JSON file")
    ap.add_argument("--no-round", action="store_true", help="keep sub-pixel box coordinates")
    a = ap.parse_args()

    paths = []
    for s in a.split:
        paths += sorted(glob.glob(os.path.join(PREP, "images", s, "*.tiff")), key=lambda p: int(os.path.basename(p)[:-5]))
    if a.ids:
        keep = {os.path.splitext(os.path.basename(l.strip()))[0] for l in open(a.ids) if l.strip()}
        paths = [p for p in paths if os.path.basename(p)[:-5] in keep]
    ids = [int(os.path.basename(p)[:-5]) for p in paths]

    df = predict(a.weights, paths, imgsz=a.imgsz, batch=a.batch, device=a.device, round_px=not a.no_round)
    check_submission(df, ids)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        df.to_csv(a.out, index=False)
        print("wrote", a.out)
    if a.eval:
        import json
        from evaluate import evaluate
        m = evaluate(df, ids, per_class=True)
        print(json.dumps(m, indent=1))
        if a.metrics_out:
            json.dump(m, open(a.metrics_out, "w"), indent=1)


if __name__ == "__main__":
    main()
