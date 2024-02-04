from __future__ import annotations

from typing import Callable

import numpy as np

from frdc.conf import BAND_MAX_CONFIG


def _fn_per_band(ar: np.ndarray, fn: Callable[[np.ndarray], np.ndarray]):
    """Runs an operation for each band in an NDArray."""
    ar = ar.copy()
    ar_bands = []
    for band in range(ar.shape[-1]):
        ar_band = ar[:, :, band]
        ar_band = fn(ar_band)
        ar_bands.append(ar_band)

    return np.stack(ar_bands, axis=-1)


def scale_0_1_per_band(
    ar: np.ndarray, epsilon: float | bool = False
) -> np.ndarray:
    """Scales an NDArray from 0 to 1 for each band independently

    Args:
        ar: NDArray of shape (H, W, C), where C is the number of bands.
        epsilon: If True, then we add a small epsilon to the denominator to
            avoid division by zero. If False, then we do not add epsilon.
            If a float, then we add that float as epsilon.
    """
    epsilon = 1e-7 if epsilon is True else epsilon

    return _fn_per_band(
        ar,
        lambda x: (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + epsilon),
    )


def scale_normal_per_band(ar: np.ndarray) -> np.ndarray:
    """Scales an NDArray to zero mean and unit variance for each band
        independently

    Args:
        ar: NDArray of shape (H, W, C), where C is the number of bands.
    """
    return _fn_per_band(ar, lambda x: (x - np.nanmean(x)) / np.nanstd(x))


def scale_static_per_band(
    ar: np.ndarray,
    order: list[str],
    bounds_config: dict[str, tuple[int, int]] = BAND_MAX_CONFIG,
) -> np.ndarray:
    """This scales statically per band, using the bounds_config.

    Args:
        ar: NDArray of shape (H, W, C), where C is the number of bands.
        order: The order of the bands.
        bounds_config: The bounds config, see BAND_MAX_CONFIG for an example.

    Examples:
        If you've retrieved the data from `get_ar_bands`, then you can use the
        order returned from that function. This is the recommended way to use
        this function.

        >>> ar, order = get_ar_bands()
        >>> scale_static_per_band(ar, order)

        If you need more control over the order, you can specify it manually.
        Given that you have an ar of shape (H, W, 3) with order
        ['WB', 'WG', 'WR']

        >>> scale_static_per_band(
        >>>     ar, ['WB', 'WG', 'WR']
        >>>     bounds_config={'WB': (0, 256), 'WG': (0, 256), 'WR': (0, 256)}
        >>> )

    Returns:
        The scaled array.
    """

    for e, band in enumerate(order):
        ar_min, ar_max = bounds_config[band]
        ar[..., e] = (ar[..., e] - ar_min) / (ar_max - ar_min)

    return ar
