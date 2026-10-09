from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable

import numpy as np


class FrameCache:
    """Byte-bounded LRU for owned, converted viewport arrays."""

    def __init__(self, max_bytes: int = 64 * 1024 * 1024) -> None:
        if max_bytes < 0:
            raise ValueError("max_bytes must be nonnegative")
        self.max_bytes = max_bytes
        self._frames: OrderedDict[Hashable, np.ndarray] = OrderedDict()
        self._bytes = 0

    @property
    def bytes_used(self) -> int:
        return self._bytes

    def get(self, key: Hashable) -> np.ndarray | None:
        frame = self._frames.get(key)
        if frame is not None:
            self._frames.move_to_end(key)
        return frame

    def put(self, key: Hashable, frame: np.ndarray) -> np.ndarray:
        """Cache an owned copy when it fits, returning the cached or input array."""
        byte_size = int(frame.nbytes)
        if byte_size == 0 or byte_size > self.max_bytes:
            return frame

        previous = self._frames.pop(key, None)
        if previous is not None:
            self._bytes -= int(previous.nbytes)

        owned = np.array(frame, copy=True, order="C")
        while self._frames and self._bytes + byte_size > self.max_bytes:
            _, evicted = self._frames.popitem(last=False)
            self._bytes -= int(evicted.nbytes)
        self._frames[key] = owned
        self._bytes += int(owned.nbytes)
        return owned

    def clear(self) -> None:
        self._frames.clear()
        self._bytes = 0
