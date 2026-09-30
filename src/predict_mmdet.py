"""Inference for mmdetection checkpoints (DINO / Co-DETR) on the prepared 16-band TIFFs.

Runs inside the isolated mmdet venv (third_party/mmdet-venv) and writes a submission-format CSV.
Scoring is done afterwards in the main env with src/evaluate.py, so this file deliberately uses only
the stdlib plus mmdet/torch (no pandas/pycocotools here).

  third_party/mmdet-venv/bin/python src/predict_mmdet.py <config> <checkpoint> \
      --split train --ids data/prepared/fold0_val.txt --out runs/<name>/oof_pred.csv
"""
import argparse
import csv
import glob
import os

import torch
from mmdet.apis import inference_detector, init_detector

PREP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "prepared")
COLS = ["id", "image_id", "class_id", "confidence", "x1", "y1", "x2", "y2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--split", nargs="+", default=["test"])
    ap.add_argument("--ids", help="file with image paths/ids to restrict to (e.g. fold0_val.txt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--no-round", action="store_true", help="keep sub-pixel coordinates")
    a = ap.parse_args()

    paths = []
    for s in a.split:
        paths += sorted(glob.glob(os.path.join(PREP, "images", s, "*.tiff")),
                        key=lambda p: int(os.path.basename(p)[:-5]))
    if a.ids:
        keep = {os.path.splitext(os.path.basename(l.strip()))[0] for l in open(a.ids) if l.strip()}
        paths = [p for p in paths if os.path.basename(p)[:-5] in keep]
    assert paths, "no images selected"

    model = init_detector(a.config, a.checkpoint, device=a.device)
    rows, n = [], 0
    for i in range(0, len(paths), a.batch):
        chunk = paths[i:i + a.batch]
        results = inference_detector(model, chunk)
        if not isinstance(results, (list, tuple)):
            results = [results]
        for p, res in zip(chunk, results):
            iid = int(os.path.basename(p)[:-5])
            inst = res.pred_instances
            keep = inst.scores > a.conf
            boxes = inst.bboxes[keep].cpu()
            scores = inst.scores[keep].cpu()
            labels = inst.labels[keep].cpu()
            if len(scores) > a.max_det:                       # keep the top-scoring detections
                top = torch.topk(scores, a.max_det).indices
                boxes, scores, labels = boxes[top], scores[top], labels[top]
            h, w = res.ori_shape
            for (x1, y1, x2, y2), s, c in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
                nd = None if not a.no_round else 2            # GT is integer VOC coords
                x1, y1 = round(max(0.0, x1), nd), round(max(0.0, y1), nd)
                x2, y2 = round(min(float(w), x2), nd), round(min(float(h), y2), nd)
                if x2 <= x1 or y2 <= y1:
                    continue
                rows.append([n, iid, int(c), round(float(s), 5), x1, y1, x2, y2])
                n += 1

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(COLS)
        wr.writerows(rows)
    imgs = len({r[1] for r in rows})
    print(f"wrote {a.out}: {len(rows)} rows, {imgs}/{len(paths)} images with detections")


if __name__ == "__main__":
    main()
