# Hyperspectral Object Detection Challenge 2026 — 6th of 309 teams

Solution code for Track 1 of the 2nd Hyperspectral Remote Sensing Data Processing and Application Challenge
([Kaggle](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026)).
**Final ranking: 6th of 309 teams** (single-member team). Phase 1 `0.64927`, Phase 2 `0.68221`, Final `0.66574`.

A non-technical one-page summary is in [`docs/`](docs/HOD26_Achievement_OnePager.pdf).
The model weights are too large for GitHub; see [`weights/README.md`](weights/README.md).

**One detection model, one forward pass per image. No ensembling, no WBF, no k-fold merging,
no TTA, no test-time model averaging, no pseudo-labelling, no hand-labelling.**

---

## 1. Team information

| | |
|---|---|
| Team name | **Furqan Ahmad** (TeamId 16871666) |
| Team size | **1 — single-member team.** The captain is the only member. |
| Captain | Furqan Ahmad — Independent participant (no institutional affiliation) |

Only one Kaggle account was used to register and compete.

---

## 2. Submitted CSV and its sha256

The file scored in Phase 2 is shipped in `submission/`:

| | |
|---|---|
| File | `phase2_final.csv` |
| **sha256** | `c0cfe03202a2cc8fab31ee5700855f3b604bc90ed69de960595a3be7b7f0589f` |
| Rows | 599,998 (2,000 images: 1,000 test + 1,000 ranking) |
| Submitted | 2026-09-26 09:43:58 UTC |
| Kaggle submission ID | 56575980 |
| Phase-2 score | 0.68221 |

This is the file the organizing committee confirmed as the scored Phase-2 submission.

Reproduce it with `./reproduce.sh` (section 5). The script recomputes the sha256 and compares.

Three Phase-2 CSVs were submitted on 2026-09-26, all from this same single architecture and
recipe. The organizing committee confirmed (submission ID 56575980) that the **first** of the
three is the one scored for Phase 2, and this package documents that file:

| CSV | submitted (UTC) | Phase-2 score | sha256 |
|---|---|---|---|
| **`phase2_final.csv`** (scored, in this package) | 09:43:58 | **0.68221** | `c0cfe03202a2cc8fab31ee5700855f3b604bc90ed69de960595a3be7b7f0589f` |
| `phase2_e72_stg2_local081.csv` (not scored) | 10:08 | — | `54e98453fc9951ce0c5183dffcf225be7da8776a107cda57e6da2f4cfd275fea` |
| `phase2_e72_ep47.csv` (not scored) | 10:12 | — | `9386ea2662d1c1b96313d54064320d821b08930acf458415ad9d3d8ff58950ab` |

The weights, config and reproduction script in this package all correspond to the scored file.

`weights/final_model.pth` sha256: `a669b87909f22f20e8a4cc053120ba3299bfa6d2baf696c63f2656c7900d4a47`

---

## 3. Model

| | |
|---|---|
| Architecture | DEIM / D-FINE-S — HGNetv2-B0 backbone, HybridEncoder, D-FINE decoder with FDR + GO-LSD |
| Input | 16-band hyperspectral cube, 768 x 1536, uint8 -> float32 /255 |
| **Parameters** | **10,244,141 total / 10,244,139 trainable (10.24 M)** |
| **FLOPs** | **68.1961 GFLOPS (33.9219 GMACs)** at `1 x 16 x 768 x 1536` |
| Checkpoint | epoch 35 of the 40-epoch all-data run (seed 0), EMA weights, 27,000 EMA updates |
| Config | `third_party/DEIM/configs/hsi/hsi_deim_dfine_s_ms1536_all_e40.yml` |

Regenerate the two numbers above with `python report_flops.py` (uses `calflops` on the exact
inference config).

The 40-epoch schedule is the winning recipe re-timed: stages scaled 72/36/8 -> 40/20/4 with
`checkpoint_freq 4`. `last.pth`, `checkpoint0035.pth` and `best_stg1.pth` of this run are
byte-identical (same EMA tensors, 27,000 updates) because the solver only writes `last.pth` while
`epoch < stop_epoch` and this recipe sets `stop_epoch: 36`; the shipped file is that checkpoint.

---

## 4. Compliance statement (Phase 2 Supplementary Notice §5.5)

### 4.1 Training data

The only training data is the competition's **3,000 Phase-1 training images** with the organizers'
VOC annotations (`data/prepared/coco/all_train.json`: 3,000 images, 10,066 annotations, 18
categories). **No external images. No external labels. No test or ranking images were used in
training in any form.**

"COCO format" in `data/prepared/coco/*.json` refers to the annotation *file format* only — these
are our own labels converted from the provided VOC XML. No COCO data is used.

### 4.2 Pretrained weights — sources and licences

| checkpoint | source | pre-trained on | licence |
|---|---|---|---|
| `dfine_s_obj2coco.pth` | D-FINE (`Peterande/D-FINE`) | Objects365 -> MS-COCO 2017 | Apache-2.0 |

This is the **only** pretrained weight used by the submitted model. Its RGB input convolution is
replaced by a 16-channel convolution whose weights are the mean of the RGB kernel tiled across all
16 bands (`src/make_16ch_weights.py`, `src/train.py:init_first_conv`). Every other parameter is
fine-tuned on competition data only.

Code licences: DEIM / D-FINE (`third_party/DEIM`, patched for 16 bands — see
`third_party/DEIM/HSI_PATCHES.md`) is Apache-2.0. All Python dependencies in section 6 are
OSI-licensed without commercial-use restrictions.

### 4.3 Pseudo-labelling / self-training / distillation

**None of these were used in the submitted model.** There is no teacher model, no pseudo-labelled
image, no confidence threshold and no iteration count to report, because no such stage exists in
the pipeline that produced `phase2_e72_ep47.csv`.

For full transparency: pseudo-labelling **was** evaluated during Phase 1 as a separate experiment
(`src/make_pseudo.py`) and was **rejected** — it scored 0.63332 / 0.63165 on the public
leaderboard versus 0.64807 for the matched control, and no pseudo-labelled checkpoint was ever
submitted in Phase 2. It forms no part of this submission. `src/make_pseudo.py` is included only
because this package ships the complete repository.

---

## 5. Reproducing the submitted CSV

Place the competition data under this package:

```
data/hyperspectral-object-detection-challenge-2026/
    class.txt
    data_train/data_train/VIS/            data_train/data_train/Annotations/VIS/
    data_test/data_test/VIS/
    ranking/                              <- the 1000 Phase-2 ranking PNGs (found recursively)
```

Then:

```bash
./reproduce.sh                 # or: ./reproduce.sh /path/to/python
```

It runs preprocessing -> inference -> structural validation -> sha256 comparison. Equivalently, by
hand:

```bash
python src/prepare.py --splits test ranking
python src/predict_deim.py weights/final_model.pth \
  --config third_party/DEIM/configs/hsi/hsi_deim_dfine_s_ms1536_all_e40.yml \
  --split test ranking --out submission/reproduced.csv
python src/check_phase2.py submission/reproduced.csv
sha256sum submission/reproduced.csv
```

### Training from scratch

```bash
python src/prepare.py --splits train test          # 16-band TIFFs + labels + folds
python src/make_16ch_weights.py                    # RGB stem -> 16-band stem
cd third_party/DEIM
python train.py -c configs/hsi/hsi_deim_dfine_s_ms1536_all_e40.yml \
                -t ../../weights/dfine_s_obj2coco_16ch.pth --use-amp --seed 0
# checkpoint_freq writes checkpoint0035.pth == last.pth == weights/final_model.pth (the submitted model)
```

Config paths in this package are **relative to `third_party/DEIM`**, so training must be launched
from that directory as shown.

---

## 6. Environment — exact versions

```
Python 3.10.12

torch==2.14.0+cu126          torchvision==0.29.0+cu126
numpy==2.2.6                 opencv-python==5.0.0.93
pillow==12.3.0               tifffile==2025.5.10
pycocotools==2.0.11          faster-coco-eval==1.8.0
pandas==2.3.3                scipy==1.15.3
PyYAML==6.0.3                transformers==5.17.0
calflops==0.3.2              tensorboard==2.21.0
matplotlib==3.10.9           ultralytics==8.4.147
```

`ultralytics` is imported by `src/predict.py` only (an unused alternative predictor); it is not on
the code path that produced the submitted CSV. `calflops` is needed only by `report_flops.py`.

Hardware used: training on 1 x NVIDIA A100 80GB; inference verified on 1 x NVIDIA RTX 3090 24GB
(CUDA 12.6). Inference over the 2,000 test+ranking images takes about 5 minutes on a 3090.

---

## 7. Preprocessing (`src/common.py`)

1. `x2cube` — pixel-unshuffle the 4x4 snapshot mosaic into a 250 x 495 x 16 cube (band = row*4 + col).
2. Per-band gain equalisation: divide by the per-band train-set mean (`BAND_MEAN`); bands 4 and 15
   are ~2x brighter than the rest on this sensor.
3. Per-image exposure normalisation: scale so the 99.5th percentile maps to 230. Exposure varies
   ~100x between frames. One scalar for all 16 bands, so the spectral ratios that separate real
   objects from their counterfeits (`apple`/`apple_plastic`, `car`/`car_toy`, ...) are preserved.
4. Write as a 16-page uint8 TIFF.

`--round-px` (default on) rounds output box coordinates to integers, worth about +0.005 mAP: the
ground-truth boxes lie on an integer pixel grid, so sub-pixel outputs can only lose IoU.

---

## 8. Determinism

Inference is deterministic: `reproduce.sh` regenerated the submitted CSV bit-identically
(sha256 `c0cfe032...`) from `weights/final_model.pth` on the verification machine. If your sha256
differs, the usual causes are a different `opencv`/`pillow` version altering PNG decode, or a
different `torchvision` interpolation kernel; the row count and image count should still match.
