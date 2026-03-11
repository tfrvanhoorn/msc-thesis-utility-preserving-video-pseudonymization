from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

import logging


logger = logging.getLogger(__name__)


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        array = np.asarray(img.convert("RGB"))
    logger.debug(
        "Loaded image %s shape=%s dtype=%s range=(%s-%s)",
        path,
        array.shape,
        array.dtype,
        array.min(),
        array.max(),
    )
    return array
