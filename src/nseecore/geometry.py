from __future__ import annotations

from .models import Pose, ViewerState


def _ceil_div(numerator: int, denominator: int) -> int:
    return -((-numerator) // denominator)


def _axis_region(
    source_length: int,
    canvas_length: int,
    source_anchor: int,
    canvas_anchor: int,
    step: int,
) -> tuple[slice, int, int]:
    if source_length < 0 or canvas_length < 0:
        raise ValueError("image and canvas dimensions must be nonnegative")
    if source_length == 0 or canvas_length == 0:
        return slice(0, 0, step), 0, 0

    first_canvas = max(0, canvas_anchor + _ceil_div(-source_anchor, step))
    last_canvas = min(
        canvas_length,
        canvas_anchor + (source_length - 1 - source_anchor) // step + 1,
    )
    if last_canvas <= first_canvas:
        return slice(0, 0, step), 0, 0

    source_start = source_anchor + (first_canvas - canvas_anchor) * step
    sample_count = last_canvas - first_canvas
    source_stop = source_start + sample_count * step
    return slice(source_start, source_stop, step), first_canvas, sample_count


def visible_image_region(
    image_shape: tuple[int, ...],
    canvas_size: Pose,
    state: ViewerState,
) -> tuple[slice, slice, Pose]:
    """Return active Y/X slices and canvas position of the first visible sample.

    The returned NumPy view is bounded by the canvas dimensions. Its sampling
    phase passes through ``state.sample_anchor_source`` so zoom can preserve the
    source pixel under the pointer exactly.
    """
    if len(image_shape) < 2:
        raise ValueError("image_shape must include height and width")
    if state.zoom < 1:
        raise ValueError("zoom must be at least 1")
    height, width = image_shape[:2]
    rows, draw_y, out_height = _axis_region(
        height,
        canvas_size.y,
        state.sample_anchor_source.y,
        state.sample_anchor_canvas.y,
        state.zoom,
    )
    columns, draw_x, out_width = _axis_region(
        width,
        canvas_size.x,
        state.sample_anchor_source.x,
        state.sample_anchor_canvas.x,
        state.zoom,
    )
    if out_height == 0 or out_width == 0:
        return slice(0, 0, state.zoom), slice(0, 0, state.zoom), Pose(
            draw_y, draw_x
        )
    return rows, columns, Pose(draw_y, draw_x)


def selection_canvas_bounds(
    state: ViewerState,
    image_shape: tuple[int, ...] | None = None,
) -> tuple[int, int, int, int] | None:
    """Project half-open source selection bounds to canvas pixel edges.

    Return ``(left, top, right, bottom)``. The far edges use ceiling division
    so a selection that falls between sampled source pixels still covers the
    displayed cell containing its source pixels.
    """
    bounds = state.selection_bounds(image_shape)
    if bounds is None:
        return None
    y0, x0, y1, x1 = bounds
    step = state.zoom
    left = state.sample_anchor_canvas.x + (
        x0 - state.sample_anchor_source.x
    ) // step
    top = state.sample_anchor_canvas.y + (
        y0 - state.sample_anchor_source.y
    ) // step
    right = state.sample_anchor_canvas.x + _ceil_div(
        x1 - state.sample_anchor_source.x, step
    )
    bottom = state.sample_anchor_canvas.y + _ceil_div(
        y1 - state.sample_anchor_source.y, step
    )
    return left, top, right, bottom
