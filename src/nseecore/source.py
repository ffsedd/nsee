from __future__ import annotations

from typing import Hashable, Protocol

import numpy as np


class ImageSource(Protocol):
    """Read-only interface for a 2D image or channel-last raster."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> np.dtype: ...

    @property
    def revision(self) -> Hashable: ...

    @property
    def cache_identity(self) -> Hashable: ...

    def read_region(self, rows: slice, columns: slice) -> np.ndarray: ...


class NumpySource:
    """Wrap an ndarray or memmap without copying the source pixels."""

    def __init__(self, array: np.ndarray) -> None:
        if array.ndim not in (2, 3):
            raise ValueError("image array must be 2D or channel-last 3D")
        if array.shape[0] < 0 or array.shape[1] < 0:
            raise ValueError("image dimensions must be nonnegative")
        self._array = array
        self._revision = 0
        self._cache_identity = object()

    @property
    def shape(self) -> tuple[int, ...]:
        return self._array.shape

    @property
    def dtype(self) -> np.dtype:
        return self._array.dtype

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def cache_identity(self) -> Hashable:
        return self._cache_identity

    def mark_dirty(self) -> None:
        """Invalidate derived frames after the wrapped array is modified."""
        self._revision += 1

    def read_region(self, rows: slice, columns: slice) -> np.ndarray:
        if self._array.ndim == 2:
            return self._array[rows, columns]
        return self._array[rows, columns, :]
