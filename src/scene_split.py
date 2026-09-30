"""Group train images into scenes (near-duplicate clusters) and build scene-disjoint splits.

Why: images were shot as short bursts of the same scene, so a random split puts the same scene in
train and val. That makes local val far too easy and our OOF scores anti-correlated with the LB.

  python src/scene_split.py --report          # cluster stats only
  python src/scene_split.py --write           # + scenes.csv, scene fold lists/yamls, clean-val list
"""
import argparse, glob, json, os
from collections import Counter, defaultdict
import cv2, numpy as np
from common import CLASSES, PREP, TRAIN_XML, parse_voc

THUMB = (32, 16)          # w, h  (images are ~2:1)

def thumbs(paths):
    out = np.empty((len(paths), THUMB[1]*THUMB[0]), np.float32)
    for i, p in enumerate(paths):
        ok, fr = cv2.imreadmulti(p, flags=cv2.IMREAD_UNCHANGED)
        im = np.mean(np.stack(fr, 2), 2)                       # band-mean grey image
        t = cv2.resize(im, THUMB, interpolation=cv2.INTER_AREA).ravel().astype(np.float32)
        t -= t.mean(); n = np.linalg.norm(t)
        out[i] = t / (n + 1e-6)                                # zero-mean, unit-norm -> dot = correlation
    return out

class UF:
    def __init__(s, n): s.p = list(range(n))
    def find(s, a):
        while s.p[a] != a: s.p[a] = s.p[s.p[a]]; a = s.p[a]
        return a
    def union(s, a, b):
        ra, rb = s.find(a), s.find(b)
        if ra != rb: s.p[ra] = rb

def cluster(sim, thr):
    uf = UF(len(sim))
    ii, jj = np.where(np.triu(sim >= thr, 1))
    for a, b in zip(ii, jj): uf.union(int(a), int(b))
    lab = defaultdict(list)
    for i in range(len(sim)): lab[uf.find(i)].append(i)
    return list(lab.values())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=0.90)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    paths = sorted(glob.glob(f"{PREP}/images/train/*.tiff"), key=lambda p: int(os.path.basename(p)[:-5]))
    ids = [os.path.basename(p)[:-5] for p in paths]
    T = thumbs(paths)
    sim = T @ T.T
    off = sim[np.triu_indices_from(sim, 1)]
    print(f"{len(ids)} images | pair correlation: p50 {np.percentile(off,50):.3f} "
          f"p99 {np.percentile(off,99):.3f} p99.9 {np.percentile(off,99.9):.3f}")
    for thr in (0.80, 0.85, 0.90, 0.95):
        cl = cluster(sim, thr)
        sz = sorted((len(c) for c in cl), reverse=True)
        singles = sum(1 for s in sz if s == 1)
        print(f"  thr {thr:.2f}: {len(cl):4d} scenes | largest {sz[:5]} | singletons {singles} | "
              f"median size {int(np.median(sz))}")
    if not a.write:
        return

    cl = cluster(sim, a.thr)
    cl.sort(key=len, reverse=True)
    scene_of = {}
    for s, c in enumerate(cl):
        for i in c: scene_of[ids[i]] = s
    cls_of = {i: {c for c, *_ in parse_voc(f"{TRAIN_XML}/{i}.xml")[2]} for i in ids}

    # 5 scene-disjoint folds, greedy: largest scenes first into the fold that is smallest / rarest-class poorest
    folds = [[] for _ in range(5)]; fold_cls = [Counter() for _ in range(5)]
    freq = Counter(c for s in cls_of.values() for c in s)
    for c in cl:
        members = [ids[i] for i in c]
        rare = min((cc for m in members for cc in cls_of[m]), key=lambda x: freq[x], default=0)
        k = min(range(5), key=lambda k: (fold_cls[k][rare], len(folds[k])))
        folds[k] += members; fold_cls[k].update(cc for m in members for cc in cls_of[m])

    os.makedirs(PREP, exist_ok=True)
    with open(f"{PREP}/scenes.csv", "w") as f:
        f.write("image_id,scene\n" + "".join(f"{i},{scene_of[i]}\n" for i in ids))
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    path = lambda i: f"{PREP}/images/train/{i}.tiff"
    for k in range(5):
        va = folds[k]; tr = [i for j in range(5) if j != k for i in folds[j]]
        open(f"{PREP}/scene{k}_train.txt", "w").write("\n".join(map(path, tr)) + "\n")
        open(f"{PREP}/scene{k}_val.txt", "w").write("\n".join(map(path, va)) + "\n")
        with open(f"{PREP}/scene{k}.yaml", "w") as f:
            f.write(f"path: {PREP}\ntrain: scene{k}_train.txt\nval: scene{k}_val.txt\nchannels: 16\nnames:\n{names}\n")
        print(f"scene fold {k}: {len(va):4d} val imgs, {len(set(scene_of[i] for i in va)):3d} scenes, "
              f"classes covered {len({c for i in va for c in cls_of[i]})}/18")

    # clean subset of the OLD fold0_val: val images whose scene never appears in fold0_train
    f0_tr = {os.path.basename(l.strip())[:-5] for l in open(f"{PREP}/fold0_train.txt") if l.strip()}
    f0_va = [os.path.basename(l.strip())[:-5] for l in open(f"{PREP}/fold0_val.txt") if l.strip()]
    seen = {scene_of[i] for i in f0_tr}
    clean = [i for i in f0_va if scene_of[i] not in seen]
    open(f"{PREP}/fold0_val_clean.txt", "w").write("\n".join(map(path, clean)) + "\n")
    print(f"\nfold0_val: {len(f0_va)} images, of which {len(clean)} ({100*len(clean)/len(f0_va):.0f}%) "
          f"are in scenes NEVER seen in fold0_train -> data/prepared/fold0_val_clean.txt")
    json.dump({"thr": a.thr, "n_scenes": len(cl), "clean_val": len(clean)},
              open(f"{PREP}/scenes_meta.json", "w"), indent=1)

if __name__ == "__main__":
    main()
