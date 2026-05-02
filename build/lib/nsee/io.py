from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image


@lru_cache(maxsize=32)
def load_image(path: str):
    return np.asarray(Image.open(path))


def save_image(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr).save(path)
