from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: Path) -> np.ndarray:
    stat = path.stat()
    return _load_image_cached(path, stat.st_mtime_ns)


@lru_cache(maxsize=32)
def _load_image_cached(path: Path, mtime_ns: int) -> np.ndarray:
    print("Loading:", path)
    with Image.open(path) as im:
        return np.asarray(im)


def save_image(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr).save(path)
