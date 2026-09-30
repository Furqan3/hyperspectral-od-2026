"""Refine predicted boxes with SAM: prompt with our box, take the mask's tight extent.

Targets the measured bottleneck: 12% of objects are found and classified correctly but boxed
loosely (IoU 0.3-0.75), while only 0.1% get the wrong class. SAM sees a pseudo-RGB built from
3 of the 16 bands, so this is a domain stretch - the point of the experiment is to find out
whether its masks are tighter than our boxes on 15-25 px objects in dark scenes.

  python src/sam_refine.py preds.csv out.csv --split train --limit 200
"""
import argparse, os
import cv2, numpy as np, pandas as pd, torch
from segment_anything import sam_model_registry, SamPredictor
from common import PREP


def pseudo_rgb(cube):
    """16-band uint8 cube -> 3-channel uint8, per-channel stretched (what SAM expects to see)."""
    v = cube[..., [0, 7, 15]].astype(np.float32)
    for c in range(3):
        lo, hi = np.percentile(v[..., c], 1), np.percentile(v[..., c], 99)
        v[..., c] = np.clip((v[..., c] - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    return v.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds"); ap.add_argument("out")
    ap.add_argument("--split", default="train")
    ap.add_argument("--conf", type=float, default=0.25, help="only refine boxes above this")
    ap.add_argument("--limit", type=int, default=0, help="first N images (0 = all)")
    ap.add_argument("--scale", type=int, default=4, help="upscale before SAM: objects are 15-25 px")
    a = ap.parse_args()

    df = pd.read_csv(a.preds)
    ids = sorted(df.image_id.unique())
    if a.limit:
        ids = ids[:a.limit]
    sam = sam_model_registry["vit_b"](checkpoint="weights/sam_vit_b_01ec64.pth").to("cuda").eval()
    pred = SamPredictor(sam)

    out, n_ref, n_kept = [], 0, 0
    for iid in ids:
        ok, fr = cv2.imreadmulti(f"{PREP}/images/{a.split}/{iid}.tiff", flags=cv2.IMREAD_UNCHANGED)
        img = pseudo_rgb(np.stack(fr, 2))
        s = a.scale
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        pred.set_image(img)
        d = df[df.image_id == iid]
        boxes = d[["x1", "y1", "x2", "y2"]].values * s
        keep = d.confidence.values >= a.conf
        if keep.sum():
            bt = torch.as_tensor(boxes[keep], dtype=torch.float32, device="cuda")
            bt = pred.transform.apply_boxes_torch(bt, img.shape[:2])
            masks, scores, _ = pred.predict_torch(point_coords=None, point_labels=None,
                                                  boxes=bt, multimask_output=False)
            masks = masks[:, 0].cpu().numpy()
        j = 0
        for r, k in zip(d.itertuples(), keep):
            x1, y1, x2, y2 = r.x1, r.y1, r.x2, r.y2
            if k:
                m = masks[j]; j += 1
                ys, xs = np.where(m)
                if len(xs) > 4:                       # mask found something
                    nx1, ny1, nx2, ny2 = xs.min()/s, ys.min()/s, xs.max()/s, ys.max()/s
                    # only accept if it stays close to the prompt (SAM can latch onto background)
                    w, h = x2 - x1, y2 - y1
                    if (abs(nx1-x1) < 0.5*w and abs(ny1-y1) < 0.5*h and
                            abs(nx2-x2) < 0.5*w and abs(ny2-y2) < 0.5*h and nx2 > nx1 and ny2 > ny1):
                        x1, y1, x2, y2 = nx1, ny1, nx2, ny2; n_ref += 1
                    else:
                        n_kept += 1
                else:
                    n_kept += 1
            out.append((r.image_id, r.class_id, r.confidence, round(x1), round(y1), round(x2), round(y2)))
    res = pd.DataFrame(out, columns=["image_id", "class_id", "confidence", "x1", "y1", "x2", "y2"])
    res = res[(res.x2 > res.x1) & (res.y2 > res.y1)].reset_index(drop=True)
    res.insert(0, "id", range(len(res)))
    res.to_csv(a.out, index=False)
    print(f"images {len(ids)} | refined {n_ref} boxes | rejected {n_kept} SAM masks | wrote {a.out}")


if __name__ == "__main__":
    main()
