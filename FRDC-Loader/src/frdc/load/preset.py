from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Any

import numpy as np
import torch
from torchvision.transforms.v2 import (
    Compose,
    ToImage,
    ToDtype,
    Resize,
)

from frdc.load.dataset import FRDCDataset, FRDCUnlabelledDataset

logger = logging.getLogger(__name__)


# This curries the FRDCDataset class, so that we can shorthand the preset
# definitions.
@dataclass
class FRDCDatasetPartial:
    """Partial class for FRDCDataset.

    Notes:
        This is used internally by FRDCDatasetPreset to define the presets
        in a more concise manner::

            # Instead of
            lambda *args, **kwargs:
                FRDCDataset("chestnut_nature_park", "20201218", None,
                            *args, **kwargs)

            # Using partial, we can do this instead
            FRDCDatasetPartial("chestnut_nature_park", "20201218", None)(
                *args, **kwargs
            )

        See FRDCDatasetPreset for usage.
    """

    site: str
    date: str
    version: str | None

    def __call__(
        self,
        transform: Callable[[list[np.ndarray]], Any] = None,
        target_transform: Callable[[list[str]], list[str]] = None,
        use_legacy_bounds: bool = False,
    ):
        """Alias for labelled()."""
        return self.labelled(
            transform,
            target_transform,
            use_legacy_bounds,
        )

    def labelled(
        self,
        transform: Callable[[list[np.ndarray]], Any] = None,
        target_transform: Callable[[list[str]], list[str]] = None,
        use_legacy_bounds: bool = False,
    ):
        """Returns the Labelled Dataset."""
        return FRDCDataset(
            self.site,
            self.date,
            self.version,
            transform,
            target_transform,
            use_legacy_bounds,
        )

    def unlabelled(
        self,
        transform: Callable[[list[np.ndarray]], Any] = None,
        target_transform: Callable[[list[str]], list[str]] = None,
        use_legacy_bounds: bool = False,
    ):
        """Returns the Unlabelled Dataset.

        Notes:
            This simply masks away the labels during __getitem__.
            The same behaviour can be achieved by setting __class__ to
            FRDCUnlabelledDataset, but this is a more convenient way to do so.
        """
        return FRDCUnlabelledDataset(
            self.site,
            self.date,
            self.version,
            transform,
            target_transform,
            use_legacy_bounds,
        )


@dataclass
class FRDCDatasetPreset:
    """Presets for the FRDCDataset.

    Examples:
        Each variable is a preset for the FRDCDataset.

        You can use it like this::

            FRDCDatasetPreset.chestnut_20201218()

        Which returns a FRDCDataset.

        Furthermore, if you're interested in the unlabelled dataset, you can
        use::

            FRDCDatasetPreset.chestnut_20201218.unlabelled()

        Which returns a FRDCUnlabelledDataset.

        If you'd like to keep the syntax consistent for labelled and unlabelled
        datasets, you can use::

            FRDCDatasetPreset.chestnut_20201218.labelled()
            FRDCDatasetPreset.chestnut_20201218.unlabelled()

        The `labelled` method is simply an alias for the `__call__` method.

        The DEBUG dataset is a special dataset that is used for debugging,
        which pulls from GCS a small cropped image and dummy label + bounds.

    """

    chestnut_20201218 = FRDCDatasetPartial(
        "chestnut_nature_park", "20201218", None
    )
    chestnut_20210510_43m = FRDCDatasetPartial(
        "chestnut_nature_park", "20210510", "90deg43m85pct255deg"
    )
    chestnut_20210510_60m = FRDCDatasetPartial(
        "chestnut_nature_park", "20210510", "90deg60m84.5pct255deg"
    )
    casuarina_20220418_183deg = FRDCDatasetPartial(
        "casuarina_nature_park", "20220418", "183deg"
    )
    casuarina_20220418_93deg = FRDCDatasetPartial(
        "casuarina_nature_park", "20220418", "93deg"
    )
    DEBUG = lambda resize=299: FRDCDatasetPartial(
        site="DEBUG", date="0", version=None
    )(
        transform=Compose(
            [
                ToImage(),
                ToDtype(torch.float32),
                Resize((resize, resize)),
            ]
        ),
        target_transform=None,
    )
