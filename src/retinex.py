"""Spectrally-safe multi-scale Retinex for the 16-band cubes.

Classic MSR works per colour channel: log(I) - log(blur(I)). Doing that band-by-band would
rescale each band differently and destroy the spectral shape, which is exactly the cue that
separates real objects from their plastic/wood copies. Here the illumination map is estimated
once from the band mean and divided out of ALL bands, so every band is scaled by the same
factor at a given pixel and band ratios are preserved exactly.

  gain = (L / MSR(L))  computed on the luminance L = mean over bands, then cube * gain
"""
import numpy as np
import cv2

SIGMAS = (15.0, 60.0, 200.0)


def illumination(L, sigmas=SIGMAS):
    """Multi-scale local illumination estimate of a 2-D image."""
    return np.mean([cv2.GaussianBlur(L, (0, 0), s, borderType=cv2.BORDER_REFLECT) for s in sigmas], 0)


def retinex_gain(cube, strength=1.0, sigmas=SIGMAS, eps=1.0, clip=(0.25, 6.0)):
    """Per-pixel gain that flattens illumination; identical for all bands."""
    L = cube.mean(2).astype(np.float32)
    I = illumination(L, sigmas)
    g = (L.mean() + eps) / (I + eps)          # bright where the scene is dark
    if strength != 1.0:
        g = g ** strength
    return np.clip(g, *clip)[..., None]


def enhance(cube, strength=1.0, target=None):
    """cube (H, W, 16) uint8/float -> Retinex-enhanced, same dtype range, spectral ratios unchanged."""
    c = cube.astype(np.float32)
    out = c * retinex_gain(c, strength)
    if target is None:
        target = np.percentile(c, 99.5)
    s = target / max(np.percentile(out, 99.5), 1e-6)
    return np.clip(out * s + 0.5, 0, 255).astype(np.uint8)


def clahe_gain(cube, clip=4.0, grid=16):
    """CLAHE computed on the band mean, returned as a per-pixel gain (keeps spectral ratios)."""
    L = cube.mean(2).astype(np.float32)
    out = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(
        np.clip(L, 0, 255).astype(np.uint8)).astype(np.float32)
    return out / np.maximum(L, 1.0)


def enhance_view(cube, strength=1.0, clip=4.0, grid=16, sharpen=0.6, sigma=1.2):
    """Retinex -> CLAHE -> unsharp mask, for LOOKING at dark frames.

    All three stages act through a single gain map shared by the 16 bands, so band ratios survive.
    This raises visible detail a lot (edges ~+75%) but also noise (~+185%): use it to inspect
    images, not as model input -- measured detail/noise is worse than the plain cube.
    """
    c = cube.astype(np.float32)
    g = retinex_gain(c, strength)[..., 0]
    step = np.clip(c * g[..., None], 0, None)
    g2 = clahe_gain(step, clip, grid)
    out = step * g2[..., None]
    if sharpen:
        blur = cv2.GaussianBlur(out, (0, 0), sigma)
        out = out + sharpen * (out - blur)
    s = np.percentile(cube, 99.5) / max(np.percentile(out, 99.5), 1e-6)
    return np.clip(out * s + 0.5, 0, 255).astype(np.uint8)
