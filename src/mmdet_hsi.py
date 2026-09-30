"""mmdetection plugin for the 16-band hyperspectral cubes.

Registers `LoadHyperspectralTiff`, which reads the prepared multi-page TIFFs (16 uint8 pages,
one per spectral band) as an (H, W, 16) array. mmdet's LoadImageFromFile goes through
mmcv.imfrombytes, which keeps at most 4 channels, so the default loader cannot be reused.

Enable it from a config with:
    custom_imports = dict(imports=['src.mmdet_hsi'], allow_failed_imports=False)
and run training from the project root.
"""
import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadHyperspectralTiff(BaseTransform):
    """Load a 16-band TIFF cube into results['img'] as (H, W, 16) uint8/float32."""

    def __init__(self, to_float32: bool = False):
        self.to_float32 = to_float32

    def transform(self, results: dict) -> dict:
        path = results.get('img_path')
        ok, frames = cv2.imreadmulti(path, flags=cv2.IMREAD_UNCHANGED)
        assert ok and frames, f'failed to read {path}'
        img = np.stack(frames, axis=2)
        if self.to_float32:
            img = img.astype(np.float32)
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(to_float32={self.to_float32})'
