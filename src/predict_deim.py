"""Inference for DEIM / D-FINE checkpoints on the prepared 16-band TIFFs -> submission CSV.

Same output format, checks, pixel rounding and --eval scoring as src/predict.py.

  python src/predict_deim.py runs/deim_f0_dfine_s/best_stg2.pth \
      --config third_party/DEIM/configs/hsi/hsi_deim_dfine_s.yml \
      --split train --ids data/prepared/fold0_val.txt --eval --metrics-out runs/deim_f0_dfine_s/oof_metrics.json
"""
import argparse
import glob
import json
import os
import sys

import pandas as pd
import torch
import torchvision.transforms.v2.functional as TF

from common import PREP, PROJECT
from predict import COLS, check_submission, read_tiff

sys.path.insert(0, os.path.join(PROJECT, "third_party", "DEIM"))
from engine.core import YAMLConfig  # noqa: E402


def load(config, ckpt, device, infer_size=None):
    cfg = YAMLConfig(config)
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False  # weights come from the checkpoint
    if infer_size:  # must happen BEFORE the model is built: the encoder bakes pos-embeds for this size
        cfg.yaml_cfg["eval_spatial_size"] = list(infer_size)
        for k in ("HybridEncoder", "DFINETransformer"):
            if k in cfg.yaml_cfg:
                cfg.yaml_cfg[k]["eval_spatial_size"] = list(infer_size)
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = st["ema"]["module"] if "ema" in st else st["model"]
    if infer_size:
        # decoder.anchors / valid_mask are deterministic grids sized for the training resolution;
        # strict=False does NOT forgive shape mismatches, so drop them and let the model rebuild.
        cur = cfg.model.state_dict()
        drop = [k for k, v in sd.items() if k in cur and cur[k].shape != v.shape]
        for k in drop:
            sd.pop(k)
        cfg.model.load_state_dict(sd, strict=False)
        if drop:
            print(f"rebuilt for {infer_size}: dropped {drop}")
    else:
        cfg.model.load_state_dict(sd)
    model = cfg.model.deploy().to(device).eval()
    post = cfg.postprocessor.deploy().to(device).eval()
    return model, post, tuple(cfg.yaml_cfg["eval_spatial_size"])


@torch.no_grad()
def predict(config, ckpt, paths, batch=16, device="cuda", conf=0.001, round_px=True, infer_size=None, bands=None):
    model, post, (H, W) = load(config, ckpt, device, infer_size)
    if infer_size:
        print(f"inference size overridden to {H}x{W}")
    rows = []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        ims = [read_tiff(p) for p in chunk]  # (h, w, 16) uint8
        # same resize as training (torchvision v2 Resize on the uint8 tensor), then [0, 1]
        x = torch.stack([TF.resize(torch.from_numpy(im).permute(2, 0, 1), [H, W]) for im in ims]).float() / 255.0
        if bands is not None:                      # band ablation: keep these, zero the rest
            keep = torch.zeros(x.shape[1], device=x.device)
            keep[list(bands)] = 1.0
            x = x * keep.view(1, -1, 1, 1)
        sizes = torch.tensor([[im.shape[1], im.shape[0]] for im in ims], device=device)  # (w, h)
        labels, boxes, scores = post(model(x.to(device)), sizes)
        nd = None if round_px else 2  # round(x, None) -> int, written as 486 rather than 486.0
        for p, im, lab, box, sc in zip(chunk, ims, labels, boxes, scores):
            iid = int(os.path.splitext(os.path.basename(p))[0])
            h, w = im.shape[:2]
            keep = sc > conf
            for (x1, y1, x2, y2), c, s in zip(box[keep].tolist(), lab[keep].tolist(), sc[keep].tolist()):
                x1, y1 = round(max(0.0, x1), nd), round(max(0.0, y1), nd)
                x2, y2 = round(min(w, x2), nd), round(min(h, y2), nd)
                if x2 <= x1 or y2 <= y1:
                    continue
                rows.append((iid, int(c), round(float(s), 5), x1, y1, x2, y2))
    df = pd.DataFrame(rows, columns=COLS[1:])
    df.insert(0, "id", range(len(df)))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", nargs="+", default=["test"])
    ap.add_argument("--ids", help="file with image paths/ids to restrict to (e.g. fold0_val.txt)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--metrics-out", default=None)
    ap.add_argument("--no-round", action="store_true")
    ap.add_argument("--bands", default=None, help="comma-separated bands to KEEP; the rest are zeroed")
    ap.add_argument("--infer-size", nargs=2, type=int, default=None, metavar=("H", "W"),
                    help="predict at this resolution instead of the config's eval_spatial_size")
    a = ap.parse_args()

    paths = []
    for s in a.split:
        paths += sorted(glob.glob(os.path.join(PREP, "images", s, "*.tiff")), key=lambda p: int(os.path.basename(p)[:-5]))
    if a.ids:
        keep = {os.path.splitext(os.path.basename(l.strip()))[0] for l in open(a.ids) if l.strip()}
        paths = [p for p in paths if os.path.basename(p)[:-5] in keep]
    ids = [int(os.path.basename(p)[:-5]) for p in paths]

    df = predict(a.config, a.weights, paths, batch=a.batch, device=a.device, round_px=not a.no_round, infer_size=a.infer_size,
                 bands=[int(b) for b in a.bands.split(",")] if a.bands else None)
    check_submission(df, ids)
    if a.out:
        df.to_csv(a.out, index=False)
        print("wrote", a.out)
    if a.eval:
        from evaluate import evaluate
        m = evaluate(df, ids, per_class=True)
        print(json.dumps({k: v for k, v in m.items() if k != "per_class_AP50_95"}, indent=1))
        if a.metrics_out:
            json.dump(m, open(a.metrics_out, "w"), indent=1)


if __name__ == "__main__":
    main()
