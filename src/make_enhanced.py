"""Write Retinex+CLAHE+sharpen enhanced copies of the prepared cubes.

  python src/make_enhanced.py --variant mild   --splits train test
  python src/make_enhanced.py --variant strong --splits train test

Reads data/prepared/images/<split>/*.tiff (already X2Cube'd + per-image normalised) and writes
data/prepared/images/<split>_rx<variant>/. Labels/COCO jsons are unchanged, so the DEIM configs
only need img_folder repointed.
"""
import argparse, glob, os
from multiprocessing import Pool
import cv2, numpy as np
from common import PREP
from retinex import enhance_view

PRESET = {"mild": dict(clip=2.0, grid=8, sharpen=0.4), "strong": dict(clip=4.0, grid=16, sharpen=0.8)}


def convert(args):
    src, dst, kw = args
    ok, fr = cv2.imreadmulti(src, flags=cv2.IMREAD_UNCHANGED)
    e = enhance_view(np.stack(fr, 2), **kw)
    cv2.imwritemulti(dst, [np.ascontiguousarray(e[..., b]) for b in range(e.shape[2])])
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(PRESET), required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    kw = PRESET[a.variant]
    for split in a.splits:
        out = os.path.join(PREP, "images", f"{split}_rx{a.variant}")
        os.makedirs(out, exist_ok=True)
        jobs = [(p, os.path.join(out, os.path.basename(p)), kw)
                for p in sorted(glob.glob(f"{PREP}/images/{split}/*.tiff"))]
        with Pool(a.workers) as pool:
            n = sum(pool.map(convert, jobs, chunksize=8))
        print(f"{split}: {n} images -> {out}")


if __name__ == "__main__":
    main()
