from __future__ import annotations

import base64
import hashlib
import logging
from io import BytesIO
from pathlib import Path
from typing import TextIO

import pandas as pd
from PIL import Image

from frdc.conf import (
    LOCAL_DATASET_ROOT_DIR,
    GCS_BUCKET,
)


def download(
    fp: str | Path,
    local_cache_dir: Path = LOCAL_DATASET_ROOT_DIR,
) -> Path:
    """Downloads a file from Google Cloud Storage. If the file already
        exists locally, and the hashes match, it will not download the file

    Args:
        fp: Path Glob to the file in GCS. This must only match one file.
        local_cache_dir: The local cache directory to download the file to.

    Examples:
        If our file in GCS is in
        gs://<BUCKET>/casuarina/20220418/183deg/result_Blue.tif
        then we can download it with:

        >>> download(
        >>>     fp=Path("casuarina/20220418/183deg/result_Blue.tif")
        >>> )

    Raises:
        ValueError: If there are multiple blobs that match the path_glob.
        FileNotFoundError: If the file does not exist in GCS.
        FileExistsError: If the file already exists locally and the hashes
            match.

    Returns:
        The local path to the downloaded file.
    """

    gcs_blobs = list(GCS_BUCKET.list_blobs(match_glob=Path(fp).as_posix()))

    if len(gcs_blobs) > 1:
        raise ValueError(f"Multiple blobs found for {fp}: {gcs_blobs}")
    elif len(gcs_blobs) == 0:
        raise FileNotFoundError(f"No blobs found for {fp}")

    # Get the local path and the GCS blob
    gcs_blob = gcs_blobs[0]
    local_path = local_cache_dir / gcs_blob.name

    # If locally exists & hashes match, return False
    if local_path.exists():
        gcs_blob.reload()  # Necessary to get the md5_hash
        gcs_hash = base64.b64decode(gcs_blob.md5_hash).hex()
        local_hash = hashlib.md5(open(local_path, "rb").read()).hexdigest()
        logging.debug(f"Local hash: {local_hash}, GCS hash: {gcs_hash}")
        if gcs_hash == local_hash:
            return local_path

    # Else, download
    logging.info(f"Downloading {gcs_blob.name} to {local_path}...")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    gcs_blob.download_to_filename(local_path.as_posix())
    return local_path


def open_file(
    fp: str | Path,
    mode: str = "r",
    local_cache_dir: Path = LOCAL_DATASET_ROOT_DIR,
) -> TextIO | BytesIO:
    """Opens a file from Google Cloud Storage.

    Notes:
        Internally, this wraps around download() and open(), it's just a
        shortcut.

    Args:
        fp: Path Glob to the file in GCS. This must only match one file.
        mode: The mode to open the file in, see open() for more details.
        local_cache_dir: The local cache directory to download the file to.

    Returns:
        A file object.
    """
    local_fp = download(fp, local_cache_dir)
    return open(local_fp, mode)


def open_image(fp: str | Path) -> Image:
    """Opens an image from Google Cloud Storage.

    Notes:
        Internally, this wraps around open_file() and Image.open().

    Args:
        fp: Path Glob to the file in GCS. This must only match one file.

    Returns:
        A PIL Image.
    """
    return Image.open(open_file(fp, "rb"))


def list_gcs_datasets(
    anchor="result_Red.tif",
) -> pd.DataFrame:
    """Lists all datasets from Google Cloud Storage.

    Args:
        anchor: The anchor file to find the dataset.
                This is used to find the dataset. For example, if we want
                to find the dataset for
                "chestnut_nature_park/20201218/183deg/result_Red.tif",
                then we can use "result_Red.tif" as the anchor file.

    Returns:
        An iterator of all blobs that match the anchor file.
    """

    # The anchor file to find the dataset
    # E.g. "result_Red.tif"
    df = (
        # The list of all blobs in the bucket that contains the anchor file
        # E.g. "chestnut_nature_park/20201218/183deg/result_Red.tif"
        pd.Series(
            [
                blob.name
                for blob in GCS_BUCKET.list_blobs(match_glob=f"**/{anchor}")
            ]
        )
        # Remove the anchor file name
        # E.g. "chestnut_nature_park/20201218/183deg"
        .str.replace(f"/{anchor}", "")
        .rename("dataset_dir")
        .drop_duplicates()
    )

    return df
