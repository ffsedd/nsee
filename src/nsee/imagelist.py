from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from .io import load_image


DEFAULT_EXTS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
)


def _scan_images(directory: Path, exts: Iterable[str]) -> list[Path]:
    exts_l = {e.lower() for e in exts}

    paths = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in exts_l
    )

    if not paths:
        raise ValueError(f"No images in {directory}")

    return paths


def _wrap_index(i: int, n: int) -> int:
    return i % n


@dataclass(slots=True)
class ImageList:
    directory: Path
    exts: frozenset[str] = DEFAULT_EXTS

    _paths: list[Path] = field(init=False, repr=False)
    _index: int = field(default=0, init=False, repr=False)

    # ---------------------------
    # lifecycle
    # ---------------------------

    def __post_init__(self) -> None:
        self.refresh()

    # ---------------------------
    # filesystem
    # ---------------------------

    def refresh(self, current: Path | None = None) -> None:
        paths = _scan_images(self.directory, self.exts)

        if current is not None:
            current = current.resolve()
            for i, p in enumerate(paths):
                if p.resolve() == current:
                    self._index = i
                    break
            else:
                self._index = min(self._index, len(paths) - 1)
        else:
            self._index = min(self._index, len(paths) - 1)

        self._paths = paths

    # ---------------------------
    # properties
    # ---------------------------

    @property
    def paths(self) -> tuple[Path, ...]:
        return tuple(self._paths)

    @property
    def index(self) -> int:
        return self._index

    @property
    def current(self) -> Path:
        return self._paths[self._index]

    # ---------------------------
    # navigation
    # ---------------------------

    def _move(self, step: int) -> Path:
        self._index = _wrap_index(self._index + step, len(self._paths))
        return self.current

    def next(self) -> Path:
        return self._move(+1)

    def prev(self) -> Path:
        return self._move(-1)

    # ---------------------------
    # IO
    # ---------------------------

    def load(self) -> np.ndarray:
        return load_image(self.current)
