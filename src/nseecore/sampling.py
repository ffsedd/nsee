from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass

import numpy as np

from .cache import FrameCache
from .geometry import visible_image_region
from .models import Pose, ViewerState
from .source import ImageSource

ArrayConverter = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class ViewportFrame:
    """Source/display pixels for just the visible part of an image."""

    array: np.ndarray
    draw: Pose
    source_rows: slice
    source_columns: slice
    zoom: int
    source_revision: Hashable
    display_key: Hashable | None


def _slice_key(value: slice) -> tuple[int | None, int | None, int | None]:
    return value.start, value.stop, value.step


def make_viewport_frame(
    source: ImageSource,
    canvas_size: Pose,
    state: ViewerState,
    *,
    converter: ArrayConverter | None = None,
    cache: FrameCache | None = None,
    display_key: Hashable | None = None,
) -> ViewportFrame:
    """Slice and optionally convert only the active viewport region.

    Without a converter, ``array`` is normally a zero-copy NumPy view. When a
    converter and cache are supplied, only its viewport-sized output is cached.
    ``display_key`` must identify converter settings such as a contrast window.
    """
    rows, columns, draw = visible_image_region(source.shape, canvas_size, state)
    key: Hashable | None = None
    if converter is not None and cache is not None:
        key = (
            source.cache_identity,
            source.revision,
            _slice_key(rows),
            _slice_key(columns),
            draw,
            state.zoom,
            id(converter),
            display_key,
        )
        cached = cache.get(key)
        if cached is not None:
            return ViewportFrame(
                cached,
                draw,
                rows,
                columns,
                state.zoom,
                source.revision,
                display_key,
            )

    visible = source.read_region(rows, columns)
    display = visible if converter is None else converter(visible)
    if not isinstance(display, np.ndarray):
        raise TypeError("display converter must return a NumPy array")
    if key is not None and cache is not None:
        display = cache.put(key, display)
    return ViewportFrame(
        display,
        draw,
        rows,
        columns,
        state.zoom,
        source.revision,
        display_key,
    )
