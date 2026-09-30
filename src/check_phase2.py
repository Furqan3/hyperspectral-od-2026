"""Validate a Phase-2 CSV before spending one of the nine submissions that count.

The failure that costs the most is silent: a CSV with no ranking rows still uploads and still
scores, it just scores ZERO on half the metric (final = mAP over test + ranking, weighted equally).
Run this on every Phase-2 file first.

  python src/check_phase2.py submissions/sub_phase2_xxx.csv
"""
import glob
import os
import sys

import pandas as pd

from common import CLASSES, PREP

COLS = ["id", "image_id", "class_id", "confidence", "x1", "y1", "x2", "y2"]


def ids_of(split):
    return {int(os.path.splitext(os.path.basename(p))[0])
            for p in glob.glob(os.path.join(PREP, "images", split, "*.tiff"))}


def main(path):
    d = pd.read_csv(path)
    test, rank = ids_of("test"), ids_of("ranking")
    got = set(d.image_id)
    fail = []

    def check(ok, msg):
        print(("  ok   " if ok else "  FAIL ") + msg)
        if not ok:
            fail.append(msg)

    print(f"{path}\n  {len(d)} rows, {d.image_id.nunique()} images")
    check(list(d.columns) == COLS, f"columns == {COLS}")
    check(d["id"].is_unique and (d["id"].values == range(len(d))).all(), "id is 0..N-1")
    check(len(rank) > 0, f"ranking split prepared ({len(rank)} images) - run prepare.py --splits ranking")
    check(not (test - got), f"all {len(test)} test images present (missing {len(test - got)})")
    check(not (rank - got), f"all {len(rank)} ranking images present (missing {len(rank - got)})")
    check(not (got - test - rank), f"no unexpected image_ids ({len(got - test - rank)})")
    check(d.class_id.between(0, len(CLASSES) - 1).all(), "class_id in range")
    check(d.confidence.between(0, 1).all(), "confidence in [0, 1]")
    check(((d.x2 > d.x1) & (d.y2 > d.y1)).all(), "x2 > x1 and y2 > y1")
    n_rank = len(d[d.image_id.isin(rank)])
    check(n_rank > 0, f"RANKING ROWS PRESENT ({n_rank}) - zero scores 0 on half the metric")
    print("\n" + ("PASS - safe to submit" if not fail else f"DO NOT SUBMIT - {len(fail)} problem(s)"))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main(sys.argv[1]))
