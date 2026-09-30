"""Raw PNG mosaics -> normalized 16-band TIFFs + YOLO labels + 5-fold split + dataset YAMLs.

Output (data/prepared/):
  images/{train,test,ranking}/<id>.tiff   uint8, 16 pages (one per band), cube resolution
  labels/train/<id>.txt                   YOLO format: cls cx cy w h (normalized)
  folds.csv                               image_id,fold  (stratified on rare classes)
  fold{k}_train.txt / fold{k}_val.txt     image path lists
  fold{k}.yaml, all.yaml                  ultralytics dataset configs (channels: 16)

With --demosaic N (2 or 4) the cube is phase-correctly upsampled N x (common.demosaic) and written to
images/<split>_dmN/; labels are shared through the symlink labels/train_dmN -> labels/train (normalized
YOLO coordinates do not change), and lists/yamls get the suffix: fold0_dm2.yaml, all_dm2.yaml, ...

Run: python src/prepare.py [--splits train test ranking] [--demosaic 2]
"""
import argparse
import glob
import os
from collections import Counter
from multiprocessing import Pool

import cv2
import numpy as np

from common import (CLASSES, PREP, RANK_IMG, TEST_IMG, TRAIN_IMG, TRAIN_XML, demosaic, denoise, load_cube,
                    normalize, parse_voc)

N_FOLDS = 5
SRC = {"train": TRAIN_IMG, "test": TEST_IMG, "ranking": RANK_IMG}


def sfx(dm, dn=False):
    return ("_dn" if dn else "") + (f"_dm{dm}" if dm > 1 else "")


def convert(args):
    split, path, dm, dn = args
    iid = os.path.splitext(os.path.basename(path))[0]
    cube = normalize(load_cube(path))
    if dn:
        cube = np.clip(denoise(cube.astype(np.float32)) + 0.5, 0, 255).astype(np.uint8)
    H, W = cube.shape[:2]
    if dm > 1:
        cube = demosaic(cube, dm)
    out = os.path.join(PREP, "images", split + sfx(dm, dn), f"{iid}.tiff")
    cv2.imwritemulti(out, [np.ascontiguousarray(cube[..., b]) for b in range(cube.shape[2])])
    if split != "train":
        return iid, None
    _, _, boxes = parse_voc(os.path.join(TRAIN_XML, f"{iid}.xml"))
    if dm == 1:  # demosaiced variants share these labels through a symlinked directory
        lines = []
        for c, x1, y1, x2, y2 in boxes:
            x1, x2 = np.clip([x1, x2], 0, W)
            y1, y2 = np.clip([y1, y2], 0, H)
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            lines.append(f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
        with open(os.path.join(PREP, "labels", split, f"{iid}.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
    return iid, [c for c, *_ in boxes]


def make_folds(img_classes):
    """Greedy iterative stratification: place images holding the rarest classes first."""
    freq = Counter(c for cs in img_classes.values() for c in set(cs))
    order = sorted(img_classes, key=lambda i: (min(freq[c] for c in img_classes[i]), i))
    fold_cls = [Counter() for _ in range(N_FOLDS)]
    fold_n = [0] * N_FOLDS
    rng = np.random.default_rng(0)
    assign = {}
    for iid in order:
        rare = min(set(img_classes[iid]), key=lambda c: freq[c])
        cand = sorted(range(N_FOLDS), key=lambda k: (fold_cls[k][rare], fold_n[k], rng.random()))
        k = cand[0]
        assign[iid] = k
        fold_n[k] += 1
        fold_cls[k].update(set(img_classes[iid]))
    return assign


def write_yaml(name, train_list, val_list):
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    with open(os.path.join(PREP, f"{name}.yaml"), "w") as f:
        f.write(f"path: {PREP}\ntrain: {train_list}\nval: {val_list}\nchannels: 16\nnames:\n{names}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--demosaic", type=int, default=1, help="phase-correct upsampling factor (1 = plain X2Cube)")
    ap.add_argument("--denoise", action="store_true", help="spectral PCA + bilateral denoise -> images/<split>_dn")
    a = ap.parse_args()
    dm, s = a.demosaic, sfx(a.demosaic, a.denoise)

    for split in a.splits:
        os.makedirs(os.path.join(PREP, "images", split + s), exist_ok=True)
        os.makedirs(os.path.join(PREP, "labels", split), exist_ok=True)
        if dm > 1 and split == "train":
            link = os.path.join(PREP, "labels", split + s)
            assert os.listdir(os.path.join(PREP, "labels", split)), "run without --demosaic first (labels)"
            if not os.path.lexists(link):
                os.symlink(os.path.join(PREP, "labels", split), link)
        files = sorted(glob.glob(os.path.join(SRC[split], "**", "*.png"), recursive=True))
        assert files, f"no images in {SRC[split]}"
        with Pool(a.workers) as pool:
            res = pool.map(convert, [(split, p, dm, a.denoise) for p in files], chunksize=8)
        print(f"{split}: {len(res)} images -> {PREP}/images/{split + s}")

        if split == "train":
            img_classes = {iid: cs for iid, cs in res}
            assign = make_folds(img_classes)  # deterministic: identical folds for every variant
            if not s:
                with open(os.path.join(PREP, "folds.csv"), "w") as f:
                    f.write("image_id,fold\n" + "".join(f"{i},{k}\n" for i, k in sorted(assign.items(), key=lambda t: int(t[0]))))
            path = lambda i: os.path.join(PREP, "images", "train" + s, f"{i}.tiff")
            for k in range(N_FOLDS):
                tr = [path(i) for i, f in assign.items() if f != k]
                va = [path(i) for i, f in assign.items() if f == k]
                open(os.path.join(PREP, f"fold{k}{s}_train.txt"), "w").write("\n".join(tr) + "\n")
                open(os.path.join(PREP, f"fold{k}{s}_val.txt"), "w").write("\n".join(va) + "\n")
                write_yaml(f"fold{k}{s}", f"fold{k}{s}_train.txt", f"fold{k}{s}_val.txt")
            # final model: train on everything (val list = fold0 only for monitoring, it is seen in training)
            open(os.path.join(PREP, f"all{s}_train.txt"), "w").write("\n".join(path(i) for i in assign) + "\n")
            write_yaml(f"all{s}", f"all{s}_train.txt", f"fold0{s}_val.txt")
            for k in range(N_FOLDS):
                cnt = Counter(c for i, f in assign.items() if f == k for c in set(img_classes[i]))
                print(f"fold {k}: {sum(1 for f in assign.values() if f == k)} imgs, stone_block imgs={cnt[CLASSES.index('stone_block')]}, "
                      f"min class imgs={min(cnt[c] for c in range(len(CLASSES)))}")


if __name__ == "__main__":
    main()
