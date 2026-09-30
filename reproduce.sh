#!/usr/bin/env bash
# HOD26 Track 1 - reproduce the submitted CSV from the shipped weights.
#
# Prerequisite: place the competition data under this package as
#   data/hyperspectral-object-detection-challenge-2026/
#       class.txt
#       data_train/data_train/{VIS,Annotations/VIS}
#       data_test/data_test/VIS
#       ranking/                     <- the 1000 Phase-2 ranking PNGs (found recursively)
#
# Usage:  ./reproduce.sh [python]
set -euo pipefail
PY="${1:-python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

EXPECT_SHA=c0cfe03202a2cc8fab31ee5700855f3b604bc90ed69de960595a3be7b7f0589f
OUT=submission/reproduced.csv

echo "== 1/4  preprocess: 4x4 mosaic PNG -> 16-band uint8 TIFF"
"$PY" src/prepare.py --splits test ranking

echo "== 2/4  inference: one model, one forward pass per image, test + ranking in ONE csv"
"$PY" src/predict_deim.py weights/final_model.pth \
  --config third_party/DEIM/configs/hsi/hsi_deim_dfine_s_ms1536_all_e40.yml \
  --split test ranking --out "$OUT"

echo "== 3/4  validate structure"
"$PY" src/check_phase2.py "$OUT"

echo "== 4/4  compare against the submitted CSV"
GOT=$(sha256sum "$OUT" | cut -d' ' -f1)
echo "expected sha256 : $EXPECT_SHA"
echo "reproduced sha256: $GOT"
if [ "$GOT" = "$EXPECT_SHA" ]; then
  echo "MATCH - reproduction is bit-identical to the submitted file."
else
  echo "DIFFER - see README section 8 (determinism notes)."
  exit 1
fi
