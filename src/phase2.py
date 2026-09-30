"""Phase-2 runbook (Sep 23-25): ranking images -> one combined test+ranking submission CSV.

Only submissions made in the ranking window count, so this exists to make that window a matter of
running one command rather than debugging. Every step that can fail silently is asserted instead.

  python src/phase2.py --zip ~/Downloads/ranking_images.zip --prepare
  python src/phase2.py --check                       # ids, counts, collisions - no GPU needed
  python src/phase2.py --predict <ckpt> --config <cfg> --out submissions/phase2.csv

The failure this guards against: predict_deim takes image_id from the TIFF filename and simply
concatenates the splits, so if the ranking set reuses the test numbering we would emit duplicate
image_ids and a silently wrong file. --check refuses to continue in that case.
"""
import argparse
import glob
import os
import subprocess
import sys
import zipfile

sys.path.insert(0, os.path.dirname(__file__))
from common import PREP, RANK_IMG


def ids_in(split):
    return sorted(int(os.path.basename(p)[:-5])
                  for p in glob.glob(os.path.join(PREP, "images", split, "*.tiff")))


def check():
    test, rank = ids_in("test"), ids_in("ranking")
    print(f"test    {len(test):5d} images  ids {min(test)}..{max(test)}" if test else "test     MISSING")
    if not rank:
        print("ranking    0 images  <- not prepared yet")
        return False
    print(f"ranking {len(rank):5d} images  ids {min(rank)}..{max(rank)}")
    clash = set(test) & set(rank)
    if clash:
        print(f"\n!! {len(clash)} image_id COLLISIONS between test and ranking, e.g. {sorted(clash)[:5]}")
        print("!! Do NOT submit a concatenated CSV - the ids are ambiguous. Resolve before predicting.")
        return False
    print(f"\nOK: no id collisions. Combined submission will cover {len(test) + len(rank)} images.")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", help="ranking_images.zip to unpack into data/.../ranking/")
    ap.add_argument("--prepare", action="store_true", help="raw ranking PNGs -> prepared TIFFs")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--predict", metavar="CKPT")
    ap.add_argument("--config")
    ap.add_argument("--out", default="submissions/phase2.csv")
    a = ap.parse_args()

    if a.zip:
        os.makedirs(RANK_IMG, exist_ok=True)
        with zipfile.ZipFile(a.zip) as z:
            z.extractall(RANK_IMG)
        n = len(glob.glob(os.path.join(RANK_IMG, "**", "*.png"), recursive=True))
        print(f"unpacked {n} PNGs into {RANK_IMG}")
        assert n, "no PNGs found - check the archive layout"

    if a.prepare:
        subprocess.check_call([sys.executable, os.path.join(os.path.dirname(__file__), "prepare.py"),
                               "--splits", "ranking"])

    if a.check or a.prepare or a.zip:
        if not check() and a.predict:
            sys.exit("refusing to predict while --check fails")

    if a.predict:
        assert a.config, "--config is required with --predict"
        subprocess.check_call([sys.executable, os.path.join(os.path.dirname(__file__), "predict_deim.py"),
                               a.predict, "--config", a.config, "--split", "test", "ranking", "--out", a.out])
        import pandas as pd
        d = pd.read_csv(a.out)
        want = len(ids_in("test")) + len(ids_in("ranking"))
        print(f"\n{a.out}: {len(d)} rows, {d.image_id.nunique()}/{want} images")
        assert d.image_id.nunique() == want, "some images produced no detections"
        print("ready to submit:  /usr/bin/python3 src/kaggle_submit.py", a.out, '"<description>"')


if __name__ == "__main__":
    main()
