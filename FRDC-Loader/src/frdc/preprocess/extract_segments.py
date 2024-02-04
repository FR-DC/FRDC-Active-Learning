import logging
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

from frdc.utils import Rect


def remove_small_segments_from_labels(
    ar_labels: np.ndarray, min_height: int = 10, min_width: int = 10
) -> np.ndarray:
    """Removes small segments from a label image.

    Args:
        ar_labels: Labelled Image, where each integer value is a segment mask.
        min_height: Minimum height of a segment to be considered "small".
        min_width: Minimum width of a segment to be considered "small".

    Returns:
        A labelled image with small segments removed.
    """
    ar_labels = ar_labels.copy()
    for i in np.unique(ar_labels):
        coords = np.argwhere(ar_labels == i)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        height = y1 - y0
        width = x1 - x0
        if height < min_height or width < min_width:
            logging.info(f"Removing segment {i} with shape {height}x{width}")
            ar_labels[ar_labels == i] = 0
    return ar_labels


def extract_segments_from_labels(
    ar: np.ndarray, ar_labels: np.ndarray, cropped: bool = True
) -> list[np.ndarray]:
    """Extracts segments as a list from a label image.

    Examples:
        Given an image::

            0  0  1
            0  0  1


        Then, if cropped = True. 2 images will be returned::

            0  0
            0  0,

            1
            1,

            ...

        If cropped = False, 2 images will be returned::

            0  0  nan
            0  0  nan,

            nan nan 1
            nan nan 1,

    Args:
        ar: The source image to extract segments from.
        ar_labels: Labelled Image, where each integer value is a segment mask.
        cropped: Whether to crop the segments to the smallest possible size.

    Returns:
        A list of segments, each segment is of shape (H, W, C).

    """
    ar_segments = []
    for segment_ix in np.unique(ar_labels):
        if cropped:
            coords = np.argwhere(ar_labels == segment_ix)
            x0, y0 = coords.min(axis=0)
            x1, y1 = coords.max(axis=0) + 1
            ar_segment_cropped_mask = ar_labels[x0:x1, y0:y1] == segment_ix
            ar_segment_cropped = ar[x0:x1, y0:y1]
            ar_segment_cropped = np.where(
                ar_segment_cropped_mask[..., None], ar_segment_cropped, np.nan
            )
            ar_segments.append(ar_segment_cropped)
        else:
            ar_segment_mask = np.array(ar_labels == segment_ix)
            ar_segment = np.where(ar_segment_mask[..., None], ar, np.nan)
            ar_segments.append(ar_segment)
    return ar_segments


def extract_segments_from_bounds(
    ar: np.ndarray, bounds: Iterable[Rect], cropped: bool = True
) -> list[np.ndarray]:
    """Extracts segments as a list from bounds

    Examples:
        Given an image::

            0  1  2
            3  4  5

        With bounds::

            1  1  0
            1  1  0

        Then, if cropped = True::

            0  1
            3  4

        If cropped = False::

            0  1  nan
            3  4  nan


    Args:
        ar: The source image to extract segments from.
        bounds: The bounds of the segment to extract. (x0, y0, x1, y1)
        cropped: Whether to crop the segments to the smallest possible size.

    Returns:
        A list of segments, each segment is of shape (H, W, C).

    """
    ar_segments = []
    for b in bounds:
        x0, y0, x1, y1 = b.x0, b.y0, b.x1, b.y1
        if cropped:
            ar_segments.append(ar[y0:y1, x0:x1])
        else:
            ar_segment_mask = np.zeros(ar.shape[:2], dtype=bool)
            ar_segment_mask[y0:y1, x0:x1] = True
            ar_segment = np.where(ar_segment_mask[..., None], ar, np.nan)
            ar_segments.append(ar_segment)
    return ar_segments


def extract_segments_from_polybounds(
    ar: np.ndarray,
    polybounds: Iterable[tuple[int, int]],
    cropped: bool = True,
    polycropped: bool = True,
) -> list[np.ndarray]:
    """Extracts segments from polygon bounds.

    Args:
        ar: The source image to extract segments from.
        polybounds: The bounds of the segment to extract. A list of points
            [(x0, y0), (x1, y1), ...].
        cropped: Whether to crop the segments to the smallest possible size.
        polycropped: Whether to further mask out the cropped image with the
            polygon mask. The mask will create nan values in the cropped image.

    Examples:
        Given an image::

            0   1   2   3
            4   5   6   7
            8   9   10  11
            12  13  14  15

        With polygon mask::

            1   1   1   0
            1   1   1   0
            1   1   0   0
            0   0   0   0

        Then, if cropped = False::

            0   1   2   nan
            4   5   6   nan
            8   9   nan nan
            nan nan nan nan

        If cropped = True, polycropped = True::

            0   1   2
            4   5   6
            8   9   nan

        If cropped = True, polycropped = False::

            0   1   2
            4   5   6
            8   9   10


    Returns:
        A list of segments, each segment is of shape (H, W, C).
        If not cropped, there will be nan values in the segments.
        Same goes with polycropped.

    """

    ar_segments = []
    for pts in polybounds:
        if cropped:
            x, y = list(zip(*pts))
            min_x, max_x, min_y, max_y = min(x), max(x), min(y), max(y)
            width, height = max_x - min_x, max_y - min_y

            # Create a mask used to mask out the cropped image
            # NOTE: We use the width and height of the segment as we'll crop out the
            #       image before we mask it.
            im_mask = Image.new("L", (width, height), 0)

            # We need to zero the points as the points is relative to the image
            # and not the cropped image
            pts_zeroed = [(x - min_x, y - min_y) for x, y in pts]
            ImageDraw.Draw(im_mask).polygon(pts_zeroed, fill=1)

            # Prepare the original image for cropping
            im_crop = ar[min_y:max_y, min_x:max_x]

            if polycropped:
                # Mask out the cropped image with our polygon mask
                im_pcrop = np.where(
                    np.array(im_mask)[..., np.newaxis] == 0, np.nan, im_crop
                )
                ar_segments.append(im_pcrop)
            else:
                # If we don't want to polycrop, then just return the cropped image
                ar_segments.append(im_crop)
        else:
            # Replicate the original image with 0s
            im_mask = Image.new("L", (ar.shape[1], ar.shape[0]), 0)

            # Create the polygon mask
            ImageDraw.Draw(im_mask).polygon(pts, fill=1)

            # Mask out the original image with our polygon mask
            im_crop = np.where(
                np.array(im_mask)[..., np.newaxis] == 0, np.nan, ar
            )
            ar_segments.append(im_crop)
    return ar_segments
