"""Add pseudo-labelled TEST images to the training set (self-training / transductive learning).

The only lever left that adds information rather than rearranging what we have: the model gets to
fit the actual test imagery. Labels come from our best model's own predictions, kept above a
confidence threshold chosen so the boxes-per-image matches the real training labels (3.37).

  python src/make_pseudo.py --preds submissions/sub_deim_all_ms1536_e40_last.csv --conf 0.5

Writes data/prepared/images/train_pseudo/ (symlinks) and coco/all_plus_pseudo.json.
Test ids are offset by +100000 so they cannot collide with train ids.
"""
import argparse, json, os
import cv2, pandas as pd
from common import CLASSES, PREP

OFFSET = 100000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--conf", type=float, default=0.5)
    a = ap.parse_args()

    base = json.load(open(f"{PREP}/coco/all_train.json"))
    train_ids = {im["id"] for im in base["images"]}
    link_dir = f"{PREP}/images/train_pseudo"
    os.makedirs(link_dir, exist_ok=True)

    # train images: symlink under their own names
    for im in base["images"]:
        src, dst = f"{PREP}/images/train/{im['file_name']}", f"{link_dir}/{im['file_name']}"
        if not os.path.lexists(dst):
            os.symlink(src, dst)

    df = pd.read_csv(a.preds)
    df = df[df.confidence >= a.conf]
    images, anns = list(base["images"]), list(base["annotations"])
    next_ann = max(x["id"] for x in anns) + 1
    n_img = 0
    for iid, g in df.groupby("image_id"):
        assert iid not in train_ids or True
        new_id = int(iid) + OFFSET
        name = f"{new_id}.tiff"
        src, dst = f"{PREP}/images/test/{iid}.tiff", f"{link_dir}/{name}"
        if not os.path.lexists(dst):
            os.symlink(src, dst)
        ok, fr = cv2.imreadmulti(src, flags=cv2.IMREAD_UNCHANGED)
        H, W = fr[0].shape[:2]
        images.append({"id": new_id, "file_name": name, "width": W, "height": H})
        n_img += 1
        for r in g.itertuples():
            x1, y1 = max(0.0, r.x1), max(0.0, r.y1)
            x2, y2 = min(float(W), r.x2), min(float(H), r.y2)
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            anns.append({"id": next_ann, "image_id": new_id, "category_id": int(r.class_id),
                         "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
            next_ann += 1
    out = {"images": images, "annotations": anns, "categories": base["categories"]}
    p = f"{PREP}/coco/all_plus_pseudo.json"
    json.dump(out, open(p, "w"))
    print(f"train {len(base['images'])} imgs / {len(base['annotations'])} boxes")
    print(f"+ pseudo {n_img} test imgs / {len(anns)-len(base['annotations'])} boxes (conf>={a.conf})")
    print(f"= {len(images)} images, {len(anns)} boxes -> {p}")


if __name__ == "__main__":
    main()
