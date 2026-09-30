"""Oversample rare street-scene images in a fold's training list (repeat factor per class).

Street classes (people, car, e-bike, stone_block) score far below the lab classes and occur in
few images, so images containing them are repeated. Repeats are symlinks under images/dup/ and
labels/dup/ (ultralytics maps images/ -> labels/ by path), so no extra disk space is used.

  python src/oversample.py --fold 0            # -> data/prepared/fold0_street.yaml
"""
import argparse
import os

from common import CLASSES, PREP, TRAIN_XML, parse_voc

REPEAT = {"people": 2, "car": 2, "e-bike": 2, "stone_block": 3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="0", help="fold index, or 'all'")
    a = ap.parse_args()
    base = "all" if a.fold == "all" else f"fold{a.fold}"
    src_list = os.path.join(PREP, f"{base}_train.txt")
    paths = [l.strip() for l in open(src_list) if l.strip()]
    rep_ids = {CLASSES.index(c): r for c, r in REPEAT.items()}
    img_dup, lab_dup = os.path.join(PREP, "images", "dup"), os.path.join(PREP, "labels", "dup")
    os.makedirs(img_dup, exist_ok=True)
    os.makedirs(lab_dup, exist_ok=True)

    out, n_extra = list(paths), 0
    for p in paths:
        iid = os.path.basename(p)[:-5]
        cls = {c for c, *_ in parse_voc(os.path.join(TRAIN_XML, f"{iid}.xml"))[2]}
        r = max([rep_ids[c] for c in cls if c in rep_ids], default=1)
        for k in range(1, r):
            img = os.path.join(img_dup, f"{iid}_r{k}.tiff")
            lab = os.path.join(lab_dup, f"{iid}_r{k}.txt")
            for link, target in ((img, p), (lab, os.path.join(PREP, "labels", "train", f"{iid}.txt"))):
                if not os.path.lexists(link):
                    os.symlink(target, link)
            out.append(img)
            n_extra += 1

    lst = os.path.join(PREP, f"{base}_street_train.txt")
    open(lst, "w").write("\n".join(out) + "\n")
    val = f"{base}_val.txt" if base != "all" else "fold0_val.txt"
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    with open(os.path.join(PREP, f"{base}_street.yaml"), "w") as f:
        f.write(f"path: {PREP}\ntrain: {os.path.basename(lst)}\nval: {val}\nchannels: 16\nnames:\n{names}\n")
    print(f"{base}: {len(paths)} images + {n_extra} street repeats = {len(out)} -> {base}_street.yaml")


if __name__ == "__main__":
    main()
