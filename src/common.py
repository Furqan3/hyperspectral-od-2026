"""Shared paths, class list, demosaicing, normalization and VOC parsing."""
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(PROJECT, "data", "hyperspectral-object-detection-challenge-2026")
TRAIN_IMG = os.path.join(RAW, "data_train/data_train/VIS")
TRAIN_XML = os.path.join(RAW, "data_train/data_train/Annotations/VIS")
TEST_IMG = os.path.join(RAW, "data_test/data_test/VIS")
RANK_IMG = os.path.join(RAW, "ranking")  # unzip ranking_images.zip here on Sep 23 (PNGs found recursively)
PREP = os.path.join(PROJECT, "data", "prepared")

CLASSES = [l.strip() for l in open(os.path.join(RAW, "class.txt")) if l.strip()]

# Mean raw value per band over the train set (notebooks/eda_out/summary.json).
# Dividing by it equalizes the sensor's per-band response (band 4 and 15 are ~2x brighter).
BAND_MEAN = np.array([40.5, 34.3, 35.2, 33.5, 63.7, 38.8, 39.4, 40.6,
                      42.5, 43.4, 38.0, 38.9, 36.0, 46.4, 52.6, 80.4], np.float32)


def train_procs():
    """Command lines of running `python ... src/train.py ...` processes.

    Parses /proc instead of `pgrep -f`, which also matches shells whose command *text*
    merely mentions src/train.py (e.g. the shell that wrote a launch script).
    """
    out = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            argv = open(f"/proc/{pid}/cmdline", "rb").read().decode(errors="ignore").split("\0")
        except OSError:
            continue
        if argv and os.path.basename(argv[0]).startswith("python") and "src/train.py" in argv[1:3]:
            out.append(" ".join(argv) + " ")
    return out


def x2cube(img, c=4):
    """Fast equivalent of the official X2Cube (pixel-unshuffle, band = row*4 + col)."""
    M, N = img.shape
    img = img[: M // c * c, : N // c * c]
    return img.reshape(M // c, c, N // c, c).transpose(0, 2, 1, 3).reshape(M // c, N // c, c * c)


def demosaic(cube, f=2, c=4):
    """Phase-correct upsampling of an X2Cube cube (H, W, 16) -> (f*H, f*W, 16).

    X2Cube treats all 16 bands of a 4x4 sensor block as if sampled at the block centre, but band
    b = row*4 + col was really sampled at offset (row+0.5, col+0.5)/4 inside it (cube units). Each band is
    bilinearly resampled from its true sample positions onto an f-times finer grid, which keeps the
    sub-pixel edge information the plain cube (and plain upscaling) throws away. Output pixel k covers
    cube-unit position (k + 0.5) / f, so boxes in cube coordinates scale by exactly f.
    """
    H, W, B = cube.shape
    out = np.empty((H * f, W * f, B), cube.dtype)
    for b in range(B):
        r, cc = divmod(b, c)
        # dst pixel k -> position (k+0.5)/f; band sample j lies at j + (cc+0.5)/c  =>  src index = (k+0.5)/f - (cc+0.5)/c
        M = np.float32([[1 / f, 0, 0.5 / f - (cc + 0.5) / c], [0, 1 / f, 0.5 / f - (r + 0.5) / c]])
        out[..., b] = cv2.warpAffine(cube[..., b], M, (W * f, H * f),
                                     flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    return out


def denoise(cube, k=6, d=5, sigma_color=12, sigma_space=2):
    """Spectral PCA (keep k of 16 band-components) + per-band edge-preserving bilateral filter.

    Measured on 14 images: noise in flat regions -44%, box edge contrast -0.1% (a Gaussian of
    equivalent strength costs -1.9%). Real-vs-counterfeit spectral separability is unchanged
    (nearest-centroid 0.786 -> 0.779). Applied identically to train and test, so it narrows the
    train/test input difference rather than fitting the training scenes better.
    """
    X = cube.reshape(-1, cube.shape[2]).astype(np.float32)
    mu = X.mean(0)
    Xc = X - mu
    w, V = np.linalg.eigh((Xc.T @ Xc) / len(Xc))
    Vk = V[:, -k:]
    out = ((Xc @ Vk) @ Vk.T + mu).reshape(cube.shape)
    return np.stack([cv2.bilateralFilter(out[..., b], d, sigma_color, sigma_space)
                     for b in range(out.shape[2])], -1)


def load_cube(path):
    """Raw 16-bit mosaic PNG -> float32 cube (H, W, 16)."""
    return x2cube(np.array(Image.open(path))).astype(np.float32)


def normalize(cube, pct=99.5, target=230.0):
    """Per-band gain equalization + per-image exposure normalization -> uint8 (H, W, 16).

    Exposure varies ~100x between frames, so every image is scaled so its 99.5th
    percentile maps to `target`. The same scale is applied to all bands, which keeps
    the spectral shape (the real-vs-counterfeit cue) intact.
    """
    x = cube / BAND_MEAN
    s = np.percentile(x, pct)
    x = x * (target / max(s, 1e-3))
    return np.clip(x + 0.5, 0, 255).astype(np.uint8)


def parse_voc(path):
    """-> (width, height, [(class_id, x1, y1, x2, y2), ...]) in cube pixel coords."""
    r = ET.parse(path).getroot()
    s = r.find("size")
    w, h = int(s.find("width").text), int(s.find("height").text)
    boxes = []
    for o in r.findall("object"):
        b = o.find("bndbox")
        x1, y1, x2, y2 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append((CLASSES.index(o.find("name").text.strip()), x1, y1, x2, y2))
    return w, h, boxes
