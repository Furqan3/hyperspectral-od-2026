"""Train a single YOLO detector on the 16-band cubes.

The first conv of the COCO-pretrained checkpoint expects 3 channels; ultralytics skips it
when shapes differ, leaving it random. `init_first_conv` replaces it with the RGB kernel
averaged over colour and tiled over the 16 bands (scaled by 3/16 so activations keep
their pretrained magnitude) -- applied after the trainer has built optimizer and EMA.

Examples:
  python src/train.py --data fold0 --name fold0_y26s_p2            # CV run
  python src/train.py --data all   --name final_y26s_p2 --epochs 100  # final model on all 3000
  python src/train.py --data fold0 --name smoke --smoke            # tiny CPU pipeline check
"""
import argparse
import os

import numpy as np
import torch
import yaml
from ultralytics import RTDETR, YOLO
from ultralytics.utils.downloads import attempt_download_asset

from common import PREP, PROJECT

WEIGHTS = os.path.join(PROJECT, "weights")


def init_first_conv(pretrained_path):
    ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    sd = (ckpt.get("ema") or ckpt["model"]).float().state_dict()
    # the RGB input conv = first 4-D weight with 3 input channels
    # (YOLO: model.0.conv.weight, RT-DETR HGStem: model.0.stem1.conv.weight)
    key, w_rgb = next((k, v) for k, v in sd.items() if v.ndim == 4 and v.shape[1] == 3)

    def callback(trainer):
        done = None
        for m in [trainer.model] + ([trainer.ema.ema] if trainer.ema else []):
            p = dict(m.named_parameters()).get(key)
            if p is None or p.shape[1] == w_rgb.shape[1] or p.shape[0] != w_rgb.shape[0]:
                continue
            c_in = p.shape[1]
            with torch.no_grad():
                p.copy_((w_rgb.mean(1, keepdim=True).repeat(1, c_in, 1, 1) * (3.0 / c_in)).to(p))
            done = c_in
        print(f"init_first_conv: {key} initialised from RGB kernels -> {done} bands" if done
              else f"init_first_conv: WARNING no {key} with non-RGB input found")

    return callback


def patch_multispectral_photometric(gain=0.25, noise=6.0, p=0.5):
    """Spectrally-safe photometric augmentation for 16-band input.

    ultralytics' RandomHSV returns early for non-3-channel images, so 16-band training gets no
    photometric augmentation at all. This replaces it, for non-RGB images only, with one random
    gain shared by all bands (keeps the spectral shape) plus Gaussian sensor noise (dark frames
    are noisy after per-image normalisation). Dataloader workers are forked, so they inherit it.
    """
    from ultralytics.data import augment
    rgb_call = augment.RandomHSV.__call__

    def __call__(self, labels):
        img = labels["img"]
        if img.shape[-1] == 3:
            return rgb_call(self, labels)
        if np.random.rand() < p:
            x = img.astype(np.float32) * np.random.uniform(1 - gain, 1 + gain)
            if noise:
                x += np.random.randn(*x.shape).astype(np.float32) * np.random.uniform(0, noise)
            labels["img"] = np.clip(x + 0.5, 0, 255).astype(np.uint8)
        return labels

    augment.RandomHSV.__call__ = __call__
    print(f"photometric aug: p={p} gain=±{gain} noise σ≤{noise}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="fold0", help="fold0..fold4 | all  (data/prepared/<name>.yaml)")
    ap.add_argument("--model", default="yolo26s-p2.yaml", help="architecture yaml (P2 head for tiny objects)")
    ap.add_argument("--pretrained", default="yolo26s.pt", help="COCO checkpoint to transfer (declare it!)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--photometric", action="store_true", help="band-shared gain + noise augmentation")
    ap.add_argument("--hyp", nargs="*", default=[], help="extra ultralytics args as key=value, e.g. box=10 close_mosaic=20")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run to validate the pipeline")
    a = ap.parse_args()

    if a.photometric:
        patch_multispectral_photometric()
    os.makedirs(WEIGHTS, exist_ok=True)
    pre = attempt_download_asset(os.path.join(WEIGHTS, a.pretrained))
    is_rtdetr = "rtdetr" in a.model.lower()
    model = (RTDETR if is_rtdetr else YOLO)(a.model).load(pre)
    model.add_callback("on_pretrain_routine_end", init_first_conv(pre))
    if is_rtdetr:
        # RTDETRDecoder caches anchors keyed on feature shapes. If they were built under torch.inference_mode
        # (warm-up, or a validation batch with the training shape) the next training step crashes with
        # "Inference tensors cannot be saved for backward". Invalidate the cache before every training batch.
        def reset_decoder_anchors(trainer):
            for m in trainer.model.modules():
                if hasattr(m, "valid_mask") and hasattr(m, "shapes"):
                    m.shapes = []

        model.add_callback("on_train_batch_start", reset_decoder_anchors)

    cfg = dict(
        data=os.path.join(PREP, f"{a.data}.yaml"),
        project=os.path.join(PROJECT, "runs"), name=a.name or a.data, exist_ok=True,
        epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, workers=a.workers, device=a.device,
        seed=0, deterministic=True, cos_lr=True, close_mosaic=10, patience=30,
        # geometry only: HSV is auto-skipped for 16-band input, colour jitter would corrupt spectra anyway
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, fliplr=0.5, flipud=0.0, mosaic=1.0,
        scale=0.5, translate=0.1, degrees=0.0, mixup=0.0,
        cache=False, plots=True,
    )
    for kv in a.hyp:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)  # typed: 10 -> int, 0.5 -> float, true -> bool
    if a.smoke:
        cfg.update(epochs=1, imgsz=320, batch=4, workers=2, device="cpu", fraction=0.02, plots=False, amp=False)
    print("train cfg:", {k: v for k, v in cfg.items() if k not in ("data", "project")})
    model.train(**cfg)


if __name__ == "__main__":
    main()
