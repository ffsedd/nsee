from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Pose:
    """Integer YX coordinate pair."""

    y: int
    x: int

    def __add__(self, other: Pose) -> Pose:
        return Pose(self.y + other.y, self.x + other.x)

    def __sub__(self, other: Pose) -> Pose:
        return Pose(self.y - other.y, self.x - other.x)

    def __floordiv__(self, divisor: int) -> Pose:
        return Pose(self.y // divisor, self.x // divisor)

    def __mul__(self, factor: int) -> Pose:
        return Pose(self.y * factor, self.x * factor)


@dataclass(slots=True)
class ViewerState:
    """Viewport coordinates; ``zoom`` is source pixels per canvas pixel."""

    mouse: Pose = Pose(0, 0)
    zoom: int = 2
    sample_anchor_source: Pose = Pose(0, 0)
    sample_anchor_canvas: Pose = Pose(0, 0)
    max_zoom: int = 100
    sel_start: Pose | None = None
    sel_end: Pose | None = None
    _pan_pointer: Pose | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.zoom < 1:
            raise ValueError("zoom must be at least 1")
        if self.max_zoom < 1:
            raise ValueError("max_zoom must be at least 1")
        self.zoom = min(self.zoom, self.max_zoom)

    def image_pixel(self, pointer: Pose | None = None) -> Pose:
        """Return the source sample under a canvas point, without clipping."""
        canvas_point = self.mouse if pointer is None else pointer
        return self.sample_anchor_source + (
            canvas_point - self.sample_anchor_canvas
        ) * self.zoom

    def set_mouse(self, pointer: Pose) -> None:
        self.mouse = pointer

    def zoom_at(
        self,
        pointer: Pose,
        direction: int,
        image_shape: tuple[int, ...],
    ) -> bool:
        """Zoom one stride step, keeping the source pixel under ``pointer`` fixed.

        ``direction=+1`` zooms in and ``direction=-1`` zooms out. Return whether
        the zoom level changed. At a zoom limit, viewport anchors are unchanged.
        """
        if direction not in (-1, 1):
            raise ValueError("direction must be +1 (in) or -1 (out)")
        if len(image_shape) < 2:
            raise ValueError("image_shape must include height and width")
        height, width = image_shape[:2]
        if height < 1 or width < 1:
            raise ValueError("cannot zoom an empty image")

        new_zoom = max(1, self.zoom - 1) if direction > 0 else min(
            self.max_zoom, self.zoom + 1
        )
        self.mouse = pointer
        if new_zoom == self.zoom:
            return False

        anchor = self.image_pixel(pointer)
        self.sample_anchor_source = Pose(
            min(max(anchor.y, 0), height - 1),
            min(max(anchor.x, 0), width - 1),
        )
        self.sample_anchor_canvas = pointer
        self.zoom = new_zoom
        return True

    def begin_pan(self, pointer: Pose) -> None:
        self.mouse = pointer
        self._pan_pointer = pointer

    def pan_to(self, pointer: Pose) -> None:
        self.mouse = pointer
        if self._pan_pointer is None:
            return
        self.sample_anchor_canvas = self.sample_anchor_canvas + (
            pointer - self._pan_pointer
        )
        self._pan_pointer = pointer

    def end_pan(self, pointer: Pose) -> None:
        self.pan_to(pointer)
        self._pan_pointer = None

    def reset_view(self, zoom: int = 2) -> None:
        if zoom < 1:
            raise ValueError("zoom must be at least 1")
        self.zoom = min(zoom, self.max_zoom)
        self.sample_anchor_source = Pose(0, 0)
        self.sample_anchor_canvas = Pose(0, 0)
        self._pan_pointer = None
        self.sel_start = None
        self.sel_end = None

    def selection_bounds(
        self, image_shape: tuple[int, ...] | None = None
    ) -> tuple[int, int, int, int] | None:
        """Return clipped half-open source bounds ``(y0, x0, y1, x1)``."""
        if self.sel_start is None or self.sel_end is None:
            return None
        y0 = min(self.sel_start.y, self.sel_end.y)
        x0 = min(self.sel_start.x, self.sel_end.x)
        y1 = max(self.sel_start.y, self.sel_end.y) + 1
        x1 = max(self.sel_start.x, self.sel_end.x) + 1
        if image_shape is not None:
            height, width = image_shape[:2]
            y0, y1 = max(0, y0), min(height, y1)
            x0, x1 = max(0, x0), min(width, x1)
        if y0 >= y1 or x0 >= x1:
            return None
        return y0, x0, y1, x1
