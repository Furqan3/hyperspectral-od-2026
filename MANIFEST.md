# File list — HOD26 Track 1 code package (team: Furqan Ahmad, TeamId 16871666)

| path | what |
|---|---|
| `README.md` | team info, submitted CSV + sha256, params/FLOPs, exact dependency versions, run commands, compliance statement (§5.5) |
| `reproduce.sh` | one command: preprocess -> inference -> validate -> sha256 compare |
| `report_flops.py` | regenerates the parameter count and FLOPs quoted in the README |
| `SHA256SUMS.txt` | sha256 of the submitted CSV and of the model weights |
| `submission/phase2_final.csv` | **the scored Phase-2 CSV** (submission ID 56575980, 599,998 rows, 2,000 images, Phase-2 score 0.68221) |
| `weights/final_model.pth` | the trained model weights (EMA, epoch 35 of the 40-epoch all-data run, 27,000 updates) |
| `src/` | our source: preprocessing, training, inference, validation, evaluation (23 files) |
| `src/common.py` | X2Cube demosaic, per-band gain, per-image exposure normalisation, VOC parsing |
| `src/prepare.py` | raw 4x4 mosaic PNG -> 16-band TIFF + labels + folds (`--splits train test ranking`) |
| `src/make_16ch_weights.py` | RGB pretrained stem -> 16-band stem |
| `src/train.py` | training entry point (16-band stem init) |
| `src/predict_deim.py` | **inference -> submission CSV** (`--split test ranking` writes one combined file) |
| `src/check_phase2.py` | validates a Phase-2 CSV; fails loudly if ranking rows are missing |
| `src/evaluate.py` | local mAP@[.5:.95], maxDets 300 |
| `third_party/DEIM/` | DEIM / D-FINE, Apache-2.0, patched for 16-band input — see `HSI_PATCHES.md` |
| `third_party/DEIM/configs/hsi/hsi_deim_dfine_s_ms1536_all_e40.yml` | the exact config of the scored model |

Config paths inside `third_party/DEIM/configs/` are relative to `third_party/DEIM`.
Competition data is **not** included; see README section 5 for where to place it.
