# Local patches for 16-band hyperspectral input

Upstream: https://github.com/ShihuaHuang95/DEIM (Apache-2.0), commit `09d35d5` (2026-03-24).
All changed lines are marked `[HSI patch]`.

| file | change |
|---|---|
| `engine/data/dataset/coco_dataset.py` | `_load_image`: multi-page TIFF → `tv_tensors.Image` (16, H, W) uint8 (PIL only keeps page 1). `_image_hw()` replaces PIL `.size` calls so tensors work. |
| `engine/data/transforms/_transforms.py` | `ConvertTensorImage` (uint8 tensor → float [0, 1]; `ConvertPILImage` is PIL-only). Compatibility fallback for torchvision ≥ 0.21, which renamed `_transform`→`transform` and `_get_params`→`make_params`. |
| `engine/backbone/hgnetv2.py` | `in_chans` argument (stem input channels). An RGB stem in a backbone checkpoint is averaged and tiled to `in_chans` (× 3/in_chans). |
| `engine/misc/profiler_utils.py` | FLOPs dummy input uses the model's real input channels and `eval_spatial_size` (h, w) instead of a square RGB image. |
| `configs/hsi/*.yml` | Competition configs: 18 classes, 512×1024 input (keeps the cube's 1:2 aspect), no RGB photometric aug / PIL mosaic / zoom-out, square multi-scale off. |

**Gotcha:** `configs/base/deim.yml` sets the D-FINE decoder to `silu`, but the official **D-FINE** checkpoints use `relu`.
Fine-tuning a D-FINE checkpoint under the DEIM recipe loads without errors (same shapes). However, 46% of the first-step predicted
boxes then have negative width or height, and training crashes in `generalized_box_iou`'s assert. `hsi_deim_dfine_s.yml` pins
`DFINETransformer.activation/mlp_act: relu` (0/1200 invalid boxes).

Detector checkpoints: `weights/dfine_{s,m}_obj2coco_16ch.pth` are the official D-FINE Objects365→COCO
checkpoints with `backbone.stem.stem1.conv.weight` converted from 3 to 16 input channels (RGB mean, tiled, × 3/16).
Fine-tune with `-t`: class heads (80 → 18) and size-dependent anchors are re-initialised automatically.
