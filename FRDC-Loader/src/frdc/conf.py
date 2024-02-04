import logging
import os
from collections import OrderedDict
from pathlib import Path

import label_studio_sdk as label_studio
import requests
from google.cloud import storage as gcs

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parents[2]
LOCAL_DATASET_ROOT_DIR = ROOT_DIR / "rsc"
os.environ["GOOGLE_CLOUD_PROJECT"] = "frmodel"
GCS_PROJECT_ID = "frmodel"
GCS_BUCKET_NAME = "frdc-ds"
GCS_CREDENTIALS = None
LABEL_STUDIO_HOST = os.environ.get("LABEL_STUDIO_HOST", "localhost")
LABEL_STUDIO_URL = f"http://{LABEL_STUDIO_HOST}:8080"

if not (LABEL_STUDIO_API_KEY := os.environ.get("LABEL_STUDIO_API_KEY", None)):
    logger.warning("LABEL_STUDIO_API_KEY not set")

BAND_CONFIG = OrderedDict(
    {
        "WB": ("*result.tif", lambda x: x[..., 2:3]),
        "WG": ("*result.tif", lambda x: x[..., 1:2]),
        "WR": ("*result.tif", lambda x: x[..., 0:1]),
        "NB": ("result_Blue.tif", lambda x: x),
        "NG": ("result_Green.tif", lambda x: x),
        "NR": ("result_Red.tif", lambda x: x),
        "RE": ("result_RedEdge.tif", lambda x: x),
        "NIR": ("result_NIR.tif", lambda x: x),
    }
)

BAND_MAX_CONFIG: dict[str, tuple[int, int]] = {
    "WR": (0, 2**8),
    "WG": (0, 2**8),
    "WB": (0, 2**8),
    "NR": (0, 2**14),
    "NG": (0, 2**14),
    "NB": (0, 2**14),
    "RE": (0, 2**14),
    "NIR": (0, 2**14),
}

try:
    logger.info("Connecting to GCS...")
    GCS_CLIENT = gcs.Client(
        project=GCS_PROJECT_ID,
        credentials=GCS_CREDENTIALS,
    )
    GCS_BUCKET = GCS_CLIENT.bucket(GCS_BUCKET_NAME)
    logger.info("Connected to GCS.")
except Exception as e:
    logger.warning(
        "Could not connect to GCS. Will not be able to download files. "
        "Check that you've (1) Installed the GCS CLI and (2) Set up the"
        "ADC with `gcloud auth application-default login`. "
        "GCS_CLIENT will be None."
    )
    GCS_CLIENT = None
    GCS_BUCKET = None

try:
    logger.info("Connecting to Label Studio...")
    requests.get(LABEL_STUDIO_URL)
    LABEL_STUDIO_CLIENT = label_studio.Client(
        url=LABEL_STUDIO_URL,
        api_key=LABEL_STUDIO_API_KEY,
    )
    logger.info("Connected to Label Studio.")
    try:
        logger.info("Attempting to Get Label Studio Project...")
        LABEL_STUDIO_CLIENT.get_project(1)
    except requests.exceptions.HTTPError:
        logger.warning(
            f"Could not get main annotation project. "
            f"Pulling annotations may not work. "
            f"It's possible that your API Key is incorrect, "
            f"or somehow your .netrc is preventing you from "
            f"accessing the project. "
        )
except requests.exceptions.ConnectionError:
    logger.warning(
        f"Could not connect to Label Studio at {LABEL_STUDIO_URL}. "
        f"LABEL_STUDIO_CLIENT will be None."
    )
    LABEL_STUDIO_CLIENT = None
