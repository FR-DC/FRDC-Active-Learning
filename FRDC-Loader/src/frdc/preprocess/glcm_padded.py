from __future__ import annotations

from typing import Hashable, Collection

import numpy as np
from glcm_cupy import glcm, Features

from frdc.conf import ROOT_DIR
from frdc.utils.file_cache import file_cache


def make_hashable(x: object) -> Hashable:
    return x if isinstance(x, np.ndarray) else str(x)


def glcm_padded(
    ar: np.ndarray,
    *,
    bin_from: int,
    bin_to: int,
    radius: int,
    step_size: int = 1,
    features: Collection[Features] | None = None,
    **kwargs,
) -> np.ndarray:
    """A wrapper for glcm-cupy's glcm. This pads the GLCM automatically

    Notes:
        This function is not cached. Use glcm_padded_cached instead if you want
        to cache the GLCM.

    Args:
        ar: Array to compute GLCM on. Must be of shape (H, W, C)
        bin_from: The upper bounds integer for the input
        bin_to: The resolution of the GLCM
        radius: Radius of each GLCM window
        step_size: Step size of each GLCM window
        features: The GLCM features to compute
        **kwargs: Additional arguments to pass to glcm-cupy's glcm

    Returns:
        Given an input (H, W, C), returns a
        GLCM of shape (H, W, C, GLCM Features)
    """
    pad = radius + step_size
    ar_pad = np.pad(
        ar, pad_width=((pad,), (pad,), (0,)), constant_values=np.nan
    )
    g = glcm(
        ar_pad,
        bin_from=bin_from,
        bin_to=bin_to,
        radius=radius,
        step_size=step_size,
        features=features
        if features
        else (
            Features.HOMOGENEITY,
            Features.CONTRAST,
            Features.ASM,
            Features.MEAN,
            Features.VARIANCE,
            Features.CORRELATION,
            Features.DISSIMILARITY,
        ),
        **kwargs,
    )
    return g[..., features] if features else g


@file_cache(
    fn_cache_fp=lambda x: ROOT_DIR / ".cache" / f"glcm_{x}.npy",
    fn_load_object=np.load,
    fn_save_object=np.save,
    fn_make_hashable=make_hashable,
)
def glcm_padded_cached(
    ar: np.ndarray,
    *,
    bin_from: int,
    bin_to: int,
    radius: int,
    step_size: int = 1,
    features: Collection[Features] | None = None,
    **kwargs,
):
    """A wrapper for glcm-cupy's glcm. This pads the GLCM automatically

    Notes:
        This function is also cached. The cache is invalidated if any of the
        arguments change.

    Args:
        ar: Array to compute GLCM on. Must be of shape (H, W, C)
        bin_from: The upper bounds integer for the input
        bin_to: The resolution of the GLCM
        radius: Radius of each GLCM window
        step_size: Step size of each GLCM window
        features: The GLCM features to compute
        **kwargs: Additional arguments to pass to glcm-cupy's glcm

    Returns:
        Given an input (H, W, C), returns a
        GLCM of shape (H, W, C, GLCM Features)
    """
    return glcm_padded(
        ar,
        bin_from=bin_from,
        bin_to=bin_to,
        radius=radius,
        step_size=step_size,
        features=features,
        **kwargs,
    )


def append_glcm_padded_cached(
    ar: np.ndarray,
    *,
    bin_from: int,
    bin_to: int,
    radius: int,
    step_size: int = 1,
    features: Collection[Features] | None = None,
    **kwargs,
) -> np.ndarray:
    """This is a wrapper around glcm_padded_cached that appends the GLCM
    features to the channels.

    Notes:
        This function is also cached. The cache is invalidated if any of the
        arguments change.

        I wanted to avoid this function as it may seem redundant, but it
        significantly improves the readability of the code.

    Returns:
        Given an input (H, W, C), returns a (H, W, C * GLCM Features) array
    """
    ar_glcm = glcm_padded_cached(
        ar,
        step_size=step_size,
        bin_from=bin_from,
        bin_to=bin_to,
        radius=radius,
        features=features,
        **kwargs,
    )

    # : (H, W, C), s_glcm: (H, W, C, GLCM Features)
    # Concatenate the GLCM features to the channels
    return np.concatenate(
        [
            # (H, W, C)
            ar,
            # ar_glcm is of shape (H, W, C, GLCM Features)
            # We reshape it to (H, W, C * GLCM Features)
            ar_glcm.reshape(*ar_glcm.shape[:-2], -1),
        ],
        axis=-1,
    )
