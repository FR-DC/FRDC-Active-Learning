from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Callable, Any

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset

from frdc.conf import (
    BAND_CONFIG,
    LABEL_STUDIO_CLIENT,
)
from frdc.load.gcs import download
from frdc.load.label_studio import get_task
from frdc.preprocess.extract_segments import (
    extract_segments_from_bounds,
    extract_segments_from_polybounds,
)
from frdc.utils import Rect

logger = logging.getLogger(__name__)


class FRDCConcatDataset(ConcatDataset):
    """ConcatDataset for FRDCDataset.

    Notes:
        This handles concatenating the targets when you add two datasets
        together, furthermore, implements the addition operator to
        simplify the syntax.

    Examples:
        If you have two datasets, ds1 and ds2, you can concatenate them::

            ds = ds1 + ds2

        `ds` will be a FRDCConcatDataset, which is a subclass of ConcatDataset.

        You can further add to a concatenated dataset::

            ds = ds1 + ds2
            ds = ds + ds3

        Finallu, all concatenated datasets have the `targets` property, which
        is a list of all the targets in the datasets::

            (ds1 + ds2).targets == ds1.targets + ds2.targets
    """

    def __init__(self, datasets: list[FRDCDataset]):
        super().__init__(datasets)
        self.datasets: list[FRDCDataset] = datasets

    @property
    def targets(self):
        return [t for ds in self.datasets for t in ds.targets]

    def __add__(self, other: FRDCDataset) -> FRDCConcatDataset:
        return FRDCConcatDataset([*self.datasets, other])


@dataclass
class FRDCDataset(Dataset):
    def __init__(
        self,
        site: str,
        date: str,
        version: str | None,
        transform: Callable[[list[np.ndarray]], Any] = None,
        target_transform: Callable[[list[str]], list[str]] = None,
        use_legacy_bounds: bool = False,
    ):
        """Initializes the FRDC Dataset.

        Notes:
            We recommend to check FRDCDatasetPreset if you want to use a
            pre-defined dataset.

            You can concatenate datasets using the addition operator, e.g.::

                ds = FRDCDataset(...) + FRDCDataset(...)

            This will return a FRDCConcatDataset, see FRDCConcatDataset for
            more information.

        Args:
            site: The site of the dataset, e.g. "chestnut_nature_park".
            date: The date of the dataset, e.g. "20201218".
            version: The version of the dataset, e.g. "183deg".
            transform: The transform to apply to each segment.
            target_transform: The transform to apply to each label.
            use_legacy_bounds: Whether to use the legacy bounds.csv file.
                This will automatically be set to True if LABEL_STUDIO_CLIENT
                is None, which happens when Label Studio cannot be connected
                to.
        """
        self.site = site
        self.date = date
        self.version = version

        self.ar, self.order = self.get_ar_bands()
        self.targets = None

        if use_legacy_bounds or (LABEL_STUDIO_CLIENT is None):
            bounds, self.targets = self.get_bounds_and_labels()
            self.ar_segments = extract_segments_from_bounds(self.ar, bounds)
        else:
            bounds, self.targets = self.get_polybounds_and_labels()
            self.ar_segments = extract_segments_from_polybounds(
                self.ar, bounds, cropped=True, polycropped=False
            )
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.ar_segments)

    def __getitem__(self, idx):
        return (
            self.transform(self.ar_segments[idx])
            if self.transform
            else self.ar_segments[idx],
            self.target_transform(self.targets[idx])
            if self.target_transform
            else self.targets[idx],
        )

    @property
    def dataset_dir(self):
        """Returns the path format of the dataset."""
        return Path(
            f"{self.site}/{self.date}/"
            f"{self.version + '/' if self.version else ''}"
        )

    def get_ar_bands_as_dict(
        self,
        bands: Iterable[str] = BAND_CONFIG.keys(),
    ) -> dict[str, np.ndarray]:
        """Gets the bands from the dataset as a dictionary of (name, image)

        Notes:
            Use get_ar_bands to get the bands as a concatenated numpy array.
            This is used to preserve the bands separately as keys and values.

        Args:
            bands: The bands to get, e.g. ['WB', 'WG', 'WR']. By default, this
                get all bands in BAND_CONFIG.

        Examples:
            >>> get_ar_bands_as_dict(['WB', 'WG', 'WR']])

            Returns

            >>> {'WB': np.ndarray, 'WG': np.ndarray, 'WR': np.ndarray}

        Returns:
            A dictionary of (KeyName, image) pairs.
        """
        d = {}
        fp_cache = {}

        try:
            config = OrderedDict({k: BAND_CONFIG[k] for k in bands})
        except KeyError:
            raise KeyError(
                f"Invalid band name. Valid band names are {BAND_CONFIG.keys()}"
            )

        for name, (glob, transform) in config.items():
            fp = download(fp=self.dataset_dir / glob)

            # We may use the same file multiple times, so we cache it
            if fp in fp_cache:
                logging.debug(f"Cache hit for {fp}, using cached image...")
                im = fp_cache[fp]
            else:
                logging.debug(f"Cache miss for {fp}, loading...")
                im = self._load_image(fp)
                fp_cache[fp] = im

            d[name] = transform(im)

        return d

    def get_ar_bands(
        self,
        bands: Iterable[str] = BAND_CONFIG.keys(),
    ) -> tuple[np.ndarray, list[str]]:
        """Gets the bands as a numpy array, and the band order as a list.

        Notes:
            This is a wrapper around get_bands, concatenating the bands.

        Args:
            bands: The bands to get, e.g. ['WB', 'WG', 'WR']. By default, this
                get all bands in BAND_CONFIG.

        Examples
            >>> get_ar_bands(['WB', 'WG', 'WR'])

            Returns

            >>> (np.ndarray, ['WB', 'WG', 'WR'])

        Returns:
            A tuple of (ar, band_order), where ar is a numpy array of shape
            (H, W, C) and band_order is a list of band names.
        """

        d: dict[str, np.ndarray] = self.get_ar_bands_as_dict(bands)
        return np.concatenate(list(d.values()), axis=-1), list(d.keys())

    def get_bounds_and_labels(
        self,
        file_name="bounds.csv",
    ) -> tuple[list[Rect], list[str]]:
        """Gets the bounds and labels from the bounds.csv file.

        Notes:
            In the context of np.ndarray, to slice with x, y coordinates,
            you need to slice with [y0:y1, x0:x1]. Which is different from the
            bounds.csv file.

        Args:
            file_name: The name of the bounds.csv file.

        Returns:
            A tuple of (bounds, labels), where bounds is a list of
            (x0, y0, x1, y1) and labels is a list of labels.
        """
        logger.warning(
            "Using legacy bounds.csv file for dataset."
            "This is pending to be deprecated in favour of pulling "
            "annotations from Label Studio."
        )
        fp = download(fp=self.dataset_dir / file_name)
        df = pd.read_csv(fp)
        return (
            [Rect(i.x0, i.y0, i.x1, i.y1) for i in df.itertuples()],
            df["name"].tolist(),
        )

    def get_polybounds_and_labels(self):
        """Gets the bounds and labels from Label Studio."""
        return get_task(
            Path(f"{self.dataset_dir}/result.jpg")
        ).get_bounds_and_labels()

    @staticmethod
    def _load_image(path: Path | str) -> np.ndarray:
        """Loads an Image from a path into a 3D numpy array. (H, W, C)

        Notes:
            If the image has only 1 channel, then it will be (H, W, 1) instead

        Args:
            path: Path to image. pathlib.Path is preferred, but str is also
                accepted.

        Returns:
            3D Image as numpy array.
        """

        im = Image.open(Path(path).as_posix())
        ar = np.asarray(im)
        return np.expand_dims(ar, axis=-1) if ar.ndim == 2 else ar

    def __add__(self, other) -> FRDCConcatDataset:
        return FRDCConcatDataset([self, other])


class FRDCUnlabelledDataset(FRDCDataset):
    """An implementation of FRDCDataset that masks away the labels.

    Notes:
        If you already have a FRDCDataset, you can simply set __class__ to
        FRDCUnlabelledDataset to achieve the same behaviour::

            ds.__class__ = FRDCUnlabelledDataset

        This will replace the __getitem__ method with the one below.

        However, it's also perfectly fine to initialize this directly::

            ds_unl = FRDCUnlabelledDataset(...)
    """

    def __getitem__(self, item):
        return (
            self.transform(self.ar_segments[item])
            if self.transform
            else self.ar_segments[item]
        )
