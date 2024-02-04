import logging

import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.segmentation import watershed

logger = logging.getLogger(__name__)


def compute_labels(
    ar: np.ndarray,
    nir_band_ix: int = -1,
    nir_threshold_value=90 / 256,
    min_crown_size=1000,
    min_crown_hole=1000,
    connectivity=2,
    peaks_footprint=200,
    watershed_compactness=0,
) -> np.ndarray:
    """Automatically segments crowns from an NDArray with a series of image
        processing operations.

    Args:
        ar: NDArray of shape (H, W, C), where C is the number of bands
        nir_band_ix: Index of NIR Band.
        nir_threshold_value: Threshold value for the NIR band.
        min_crown_size: Minimum crown size in pixels.
        min_crown_hole: Minimum crown hole size in pixels.
        connectivity: Connectivity for morphological operations.
        peaks_footprint: Footprint for peak_local_max.
        watershed_compactness: Compactness for watershed.

    Returns:
        A tuple of (background, crowns), background is the background image and
        crowns is a list of np.ndarray crowns.
        Background is of shape (H, W, C), where C is the number of bands
        Crowns is a list[np.ndarray] crowns, each of shape (H, W, C).
    """
    # Raise deprecation warning
    logger.warning(
        "This function is to be deprecated. use functions separately instead "
        "for more control over the parameters. This function assumes the NIR "
        "band is the last band.",
        DeprecationWarning,
    )

    from frdc.preprocess.scale import scale_0_1_per_band

    ar = scale_0_1_per_band(ar)
    ar_mask = threshold_binary_mask(ar, nir_band_ix, nir_threshold_value)
    ar_mask = remove_small_objects(
        ar_mask, min_size=min_crown_size, connectivity=connectivity
    )
    ar_mask = remove_small_holes(
        ar_mask, area_threshold=min_crown_hole, connectivity=connectivity
    )
    ar_labels = binary_watershed(
        ar_mask, peaks_footprint, watershed_compactness
    )
    return ar_labels


def threshold_binary_mask(
    ar: np.ndarray, band_ix: int, threshold_value: float = 90 / 256
) -> np.ndarray:
    """Creates a binary mask array from an NDArray by thresholding a band.

    Notes:
         For the band argument, use Band.NIR for the NIR band,
         Band.RED for the RED band, etc.

    Returns:
        A binary mask array of shape (H, W).
        True for values above the threshold, False otherwise.
    """
    return ar[:, :, band_ix] > threshold_value


def binary_watershed(
    ar_mask: np.ndarray,
    peaks_footprint: int = 200,
    watershed_compactness: float = 0,
) -> np.ndarray:
    """Watershed segmentation of a binary mask.

    Notes:
        This function is used internally by `segment_crowns`.

    Args:
        ar_mask: Binary mask array of shape (H, W).
        peaks_footprint: Footprint for peak_local_max.
        watershed_compactness: Compactness for watershed.

    Returns:
        A watershed segmentation of the binary mask.
    """

    # Watershed
    # For watershed, we need:
    #   Image Depth: The distance from the background
    #   Image Basins: The local maxima of the image depth.
    #                 i.e. points that are the deepest in the image.

    # The ar distance is the distance from the background.
    ar_distance = distance_transform_edt(ar_mask)

    # We find the basins by the local maxima of the negative image depth.
    ar_watershed_basin_coords = peak_local_max(
        ar_distance,
        footprint=np.ones((peaks_footprint, peaks_footprint)),
        # min_distance=1,
        exclude_border=0,
        # p_norm=2,
        labels=ar_mask,
    )
    ar_watershed_basins = np.zeros(ar_distance.shape, dtype=bool)
    ar_watershed_basins[tuple(ar_watershed_basin_coords.T)] = True
    ar_watershed_basins, _ = ndimage.label(ar_watershed_basins)

    # TODO: I noticed that low watershed compactness values produces miniblobs,
    #  which can be indicative of redundant crowns.
    #  We should investigate this further.
    return watershed(
        image=-ar_distance,
        # We use the negative so that "peaks" become "troughs"
        markers=ar_watershed_basins,
        mask=ar_mask,
        # Enable this to see the watershed lines
        # watershed_line=True,
        compactness=watershed_compactness,
    )
