import numpy as np
import pytest

from nseecore import (
    FrameCache,
    NumpySource,
    Pose,
    ViewerState,
    make_viewport_frame,
    selection_canvas_bounds,
    visible_image_region,
)


def test_visible_region_is_bounded_to_canvas_and_is_a_view() -> None:
    image = np.arange(100 * 120, dtype=np.uint32).reshape(100, 120)
    state = ViewerState(zoom=2)

    rows, columns, draw = visible_image_region(image.shape, Pose(8, 10), state)
    visible = image[rows, columns]

    np.testing.assert_array_equal(visible, image[0:16:2, 0:20:2])
    assert visible.shape == (8, 10)
    assert np.shares_memory(image, visible)
    assert draw == Pose(0, 0)


def test_randomized_slices_match_anchor_coordinate_mapping() -> None:
    rng = np.random.default_rng(428)
    for _ in range(250):
        height, width = (int(value) for value in rng.integers(1, 80, size=2))
        canvas_height, canvas_width = (
            int(value) for value in rng.integers(0, 30, size=2)
        )
        step = int(rng.integers(1, 12))
        source_anchor = Pose(
            int(rng.integers(0, height)), int(rng.integers(0, width))
        )
        canvas_anchor = Pose(
            int(rng.integers(-20, 40)), int(rng.integers(-20, 40))
        )
        state = ViewerState(
            zoom=step,
            sample_anchor_source=source_anchor,
            sample_anchor_canvas=canvas_anchor,
        )
        image = np.arange(height * width).reshape(height, width)

        rows, columns, draw = visible_image_region(
            image.shape, Pose(canvas_height, canvas_width), state
        )
        visible = image[rows, columns]
        assert visible.shape[0] <= canvas_height
        assert visible.shape[1] <= canvas_width
        for row in range(visible.shape[0]):
            for column in range(visible.shape[1]):
                canvas_y, canvas_x = draw.y + row, draw.x + column
                source_y = source_anchor.y + (canvas_y - canvas_anchor.y) * step
                source_x = source_anchor.x + (canvas_x - canvas_anchor.x) * step
                assert visible[row, column] == image[source_y, source_x]


def test_visible_region_keeps_zoom_anchor_sampling_phase() -> None:
    image = np.arange(100 * 100, dtype=np.int32).reshape(100, 100)
    state = ViewerState(
        zoom=3,
        sample_anchor_source=Pose(20, 22),
        sample_anchor_canvas=Pose(2, 1),
    )

    rows, columns, draw = visible_image_region(image.shape, Pose(6, 7), state)
    visible = image[rows, columns]

    np.testing.assert_array_equal(visible, image[14:32:3, 19:40:3])
    assert visible.shape == (6, 7)
    assert draw == Pose(0, 0)
    # The sample anchor remains in the returned view at its canvas coordinate.
    assert visible[2, 1] == image[20, 22]


def test_zoom_keeps_pixel_under_pointer_at_same_canvas_coordinate() -> None:
    state = ViewerState(
        zoom=4,
        sample_anchor_source=Pose(20, 22),
        sample_anchor_canvas=Pose(2, 1),
    )
    pointer = Pose(5, 4)
    source_pixel = state.image_pixel(pointer)

    assert state.zoom_at(pointer, direction=1, image_shape=(100, 100))

    assert state.zoom == 3
    assert state.sample_anchor_source == source_pixel
    assert state.sample_anchor_canvas == pointer
    assert state.image_pixel(pointer) == source_pixel


def test_zoom_clamps_source_anchor_and_obeys_limits() -> None:
    state = ViewerState(zoom=1, max_zoom=3)
    assert not state.zoom_at(Pose(0, 0), direction=1, image_shape=(10, 10))

    state.zoom = 3
    assert not state.zoom_at(Pose(0, 0), direction=-1, image_shape=(10, 10))

    state = ViewerState(zoom=2)
    assert state.zoom_at(Pose(-2, 15), direction=1, image_shape=(10, 10))
    assert state.sample_anchor_source == Pose(0, 9)

    with pytest.raises(ValueError):
        state.zoom_at(Pose(0, 0), direction=0, image_shape=(10, 10))


def test_left_drag_moves_canvas_anchor_without_changing_source_anchor() -> None:
    state = ViewerState(
        zoom=3,
        sample_anchor_source=Pose(30, 45),
        sample_anchor_canvas=Pose(10, 20),
    )
    source_anchor = state.sample_anchor_source
    state.begin_pan(Pose(4, 8))
    state.pan_to(Pose(7, 6))
    state.end_pan(Pose(9, 10))

    assert state.sample_anchor_canvas == Pose(15, 22)
    assert state.sample_anchor_source == source_anchor
    assert state._pan_pointer is None


def test_selection_bounds_are_clipped_half_open_numpy_bounds() -> None:
    state = ViewerState()
    state.sel_start = Pose(8, 9)
    state.sel_end = Pose(2, 3)

    assert state.selection_bounds((7, 8)) == (2, 3, 7, 8)
    assert state.selection_bounds() == (2, 3, 9, 10)


def test_selection_overlay_projects_half_open_edges() -> None:
    state = ViewerState(
        zoom=3,
        sample_anchor_source=Pose(10, 10),
        sample_anchor_canvas=Pose(5, 5),
    )
    state.sel_start = Pose(8, 8)
    state.sel_end = Pose(10, 10)

    assert selection_canvas_bounds(state) == (4, 4, 6, 6)


@pytest.mark.parametrize(
    ("anchor_source", "anchor_canvas", "expected_draw", "expected_values"),
    [
        (Pose(4, 4), Pose(-1, -1), Pose(0, 0), [66, 86]),
        (Pose(0, 0), Pose(2, 2), Pose(2, 2), [0, 2]),
    ],
)
def test_visible_region_clips_at_canvas_edges(
    anchor_source: Pose,
    anchor_canvas: Pose,
    expected_draw: Pose,
    expected_values: list[int],
) -> None:
    image = np.arange(10 * 10).reshape(10, 10)
    state = ViewerState(
        zoom=2,
        sample_anchor_source=anchor_source,
        sample_anchor_canvas=anchor_canvas,
    )

    rows, columns, draw = visible_image_region(image.shape, Pose(4, 4), state)
    view = image[rows, columns]

    assert draw == expected_draw
    if anchor_canvas.y < 0:
        assert view[:, 0].tolist() == expected_values
    else:
        assert view[0, :].tolist() == [0, 2]


def test_viewport_entirely_outside_image_returns_empty_slices() -> None:
    image = np.ones((10, 10), dtype=np.uint8)
    state = ViewerState(
        zoom=2,
        sample_anchor_source=Pose(0, 0),
        sample_anchor_canvas=Pose(20, 20),
    )

    rows, columns, _ = visible_image_region(image.shape, Pose(5, 5), state)
    assert image[rows, columns].size == 0


def test_invalid_geometry_is_rejected() -> None:
    with pytest.raises(ValueError):
        visible_image_region((10,), Pose(5, 5), ViewerState())
    with pytest.raises(ValueError):
        visible_image_region((10, 10), Pose(-1, 5), ViewerState())
    with pytest.raises(ValueError):
        ViewerState(zoom=0)


def test_numpy_source_wraps_without_copying_and_tracks_revision() -> None:
    image = np.arange(12).reshape(3, 4)
    source = NumpySource(image)
    view = source.read_region(slice(0, 3, 2), slice(1, 4, 2))

    assert source.shape == image.shape
    assert source.dtype == image.dtype
    assert np.shares_memory(image, view)
    assert source.revision == 0
    source.mark_dirty()
    assert source.revision == 1


def test_numpy_source_accepts_memmap(tmp_path) -> None:
    path = tmp_path / "image.dat"
    mapped = np.memmap(path, mode="w+", dtype=np.uint16, shape=(8, 10))
    mapped[:] = np.arange(80).reshape(8, 10)
    mapped.flush()
    source = NumpySource(mapped)

    view = source.read_region(slice(2, 8, 2), slice(1, 9, 3))
    np.testing.assert_array_equal(view, mapped[2:8:2, 1:9:3])
    assert np.shares_memory(mapped, view)


def test_make_viewport_frame_returns_only_visible_source_view() -> None:
    image = np.arange(100 * 200, dtype=np.uint16).reshape(100, 200)
    source = NumpySource(image)
    state = ViewerState(zoom=4)

    frame = make_viewport_frame(source, Pose(20, 30), state)

    assert frame.array.shape == (20, 30)
    assert np.shares_memory(image, frame.array)
    assert frame.zoom == 4
    assert frame.source_revision == 0


def test_numpy_source_preserves_channel_last_arrays() -> None:
    image = np.zeros((12, 16, 4), dtype=np.uint16)
    source = NumpySource(image)
    state = ViewerState(zoom=2)

    frame = make_viewport_frame(source, Pose(4, 5), state)

    assert frame.array.shape == (4, 5, 4)
    assert frame.array.dtype == np.uint16
    assert np.shares_memory(image, frame.array)


def test_converted_viewport_cache_hits_and_invalidates_on_source_revision() -> None:
    source = NumpySource(np.arange(100, dtype=np.uint8).reshape(10, 10))
    state = ViewerState(zoom=2)
    cache = FrameCache(max_bytes=1024)
    calls = 0

    def converter(view: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return view.copy()

    first = make_viewport_frame(
        source, Pose(4, 5), state, converter=converter, cache=cache,
        display_key=("uint8",),
    )
    second = make_viewport_frame(
        source, Pose(4, 5), state, converter=converter, cache=cache,
        display_key=("uint8",),
    )
    assert calls == 1
    assert first.array is second.array
    assert cache.bytes_used == first.array.nbytes

    source.mark_dirty()
    third = make_viewport_frame(
        source, Pose(4, 5), state, converter=converter, cache=cache,
        display_key=("uint8",),
    )
    assert calls == 2
    assert third.source_revision == 1

    state.sample_anchor_canvas = Pose(1, 0)
    make_viewport_frame(
        source, Pose(4, 5), state, converter=converter, cache=cache,
        display_key=("uint8",),
    )
    assert calls == 3
    make_viewport_frame(
        source, Pose(4, 5), state, converter=converter, cache=cache,
        display_key=("uint8", "changed-window"),
    )
    assert calls == 4


def test_frame_cache_is_lru_and_respects_byte_limit() -> None:
    cache = FrameCache(max_bytes=8)
    first = np.zeros((2, 2), dtype=np.uint8)
    second = np.ones((2, 2), dtype=np.uint8)
    third = np.full((2, 2), 2, dtype=np.uint8)
    cache.put("first", first)
    cache.put("second", second)
    assert cache.get("first") is not None  # First becomes most recently used.
    cache.put("third", third)

    assert cache.bytes_used == 8
    assert cache.get("first") is not None
    assert cache.get("second") is None
    assert cache.get("third") is not None

    cache.put("too-large", np.zeros((3, 3), dtype=np.uint8))
    assert cache.get("too-large") is None
    assert cache.bytes_used == 8
