# nseecore: implementation specification

## 1. Purpose

Build a small, GUI-independent Python core for image viewers that need fast
navigation of large raster images. The first consumers are `nsee` and
`rdxkit.tile_viewer`, but the core must not depend on either application.

The core owns viewport math, source-region selection, integer nearest-neighbor
sampling, and bounded display-frame caching. Each application keeps its own
window layout, commands, file navigation, image-format policy, and GUI toolkit
adapter.

The performance rule is: **work on the visible viewport only**. Panning or
zooming must never resize, copy, or convert the whole source image. For a
NumPy-backed source, derive the active region with array slicing and hand only
that slice to the display adapter.

## 2. Goals

- Keep navigation responsive for images much larger than the viewport.
- Preserve fast integer zoom-out by using NumPy stepped slices, not a general
  image-resize operation.
- Make the visible source region explicit and testable.
- Zoom toward or away from the mouse pointer with the mouse wheel.
- Pan by dragging with the left mouse button.
- Work with ordinary NumPy arrays and `numpy.memmap` without requiring a
  full-resolution display copy.
- Keep source dtype and pixel values intact until an application-supplied
  display conversion is requested.
- Keep memory used by derived frames bounded by the configured cache limit.
- Expose pure Python/NumPy interfaces that can be adapted to Tk, Qt, or another
  renderer.

## 3. Non-goals for the first version

- GUI widgets, event-loop ownership, menus, dialogs, or keyboard bindings.
- File dialogs, directory scanning, image saving, JPEG rotation, or format
  discovery.
- Smooth/interpolated zoom. The first version uses nearest-neighbor sampling.
- A multiresolution pyramid, tile server, GPU renderer, or asynchronous decode
  scheduler. Add these only if measured source-read times require them.
- A requirement that every source be a single in-memory NumPy array. The source
  interface must leave room for region-backed sources without implementing a
  file-format plugin framework in v1.

## 4. Package boundaries

`nseecore` is a pure core. It may depend on Python's standard library and NumPy.
It must not import Tkinter, Pillow, Qt, Matplotlib, SciPy, or code from either
viewer application.

Suggested modules:

```text
nseecore/
    __init__.py
    source.py       # source protocol and NumPy/memmap adapter
    viewport.py     # state, coordinates, pan, wheel zoom, visible bounds
    sampling.py     # active-region NumPy slicing and display-frame creation
    cache.py        # byte-bounded cache for derived viewport frames
```

Keep the public API small. Avoid one class per concept where a dataclass or
function is sufficient. Public symbols should be re-exported from
`nseecore.__init__`.

### Application responsibilities

- **nsee:** Pillow loading/saving, image-list navigation, crop/save/rotate
  commands, status bar, and its Tk adapter.
- **rdxkit:** PBF decoding, image selection, grayscale normalization, histogram
  and statistics panels, and its Tk adapter.
- **each adapter:** map toolkit events to core calls; create/update the toolkit
  image object from the returned viewport frame.

The core may provide a display conversion hook, but it must not impose one
normalization policy. For example, uint8 color can pass through, while uint16
or uint32 scientific images need an app-selected display window.

## 5. Coordinate and zoom model

Use YX order for `Pose` values and public point arguments, matching the current
viewer state. NumPy arrays also use `(row, column)` for spatial indexing. Name
scalar arguments explicitly (`pointer_y`, `pointer_x`) and convert from GUI XY
coordinates at the adapter boundary.

The viewport stores:

- `sample_step`: positive integer source pixels represented by one displayed
  pixel. `1` is native 1:1; larger values zoom out by sampling more sparsely.
- `sample_anchor_source`: source pixel under the last zoom pointer.
- `sample_anchor_canvas`: canvas position where that source pixel is shown.
- `canvas_width`, `canvas_height`: current drawable area in screen pixels.
- `min_sample_step = 1` and a configurable finite `max_sample_step` (default
  `100`). No magnification beyond native resolution is required in v1.

The integer step is intentional: each screen-pixel step from the anchor samples
`sample_step` source pixels away. Increasing `sample_step` zooms out; decreasing
it zooms in. A platform-independent wheel method accepts a signed direction
rather than a Tk event object. Wheel-up decreases `sample_step` by one, clamped
to 1; wheel-down increases it by one, clamped to `max_sample_step`.

The sampling grid is anchored to the source pixel under the last zoom pointer,
not globally to source coordinate `(0, 0)`. This phase shift ensures that the
anchored pixel is included in the new stride grid and remains at the same canvas
coordinate after zoom. Panning moves the anchor's canvas position without
changing the source anchor or sampling phase.

### Pointer-anchored zoom

Zoom must keep the source pixel under the mouse at the same canvas coordinate.
Before changing `sample_step`, map the pointer to its source pixel using the
current anchor and stride. Clamp that source pixel to image bounds if the
pointer maps outside the image. Then set `sample_anchor_source` to that pixel
and `sample_anchor_canvas` to the pointer coordinate before rendering at the
new stride. Because the sampling phase moves with this anchor, that exact source
pixel remains in the new sample grid at the same canvas position. A wheel event
at a stride limit leaves the viewport state unchanged.

### Left-button drag

The application adapter calls `begin_pan(pointer)`, `pan_to(pointer)`, and
`end_pan(pointer)` (or equivalent functions). Each pointer delta moves the
image by the same number of screen pixels:

```text
sample_anchor_canvas.x += current_pointer_x - previous_pointer_x
sample_anchor_canvas.y += current_pointer_y - previous_pointer_y
```

The core does not depend on mouse-button names; the adapter binds these methods
to left-button press, motion, and release. Panning is unrestricted in v1; the
adapter may choose to constrain movement later.

### Selection and crop bounds

Represent rectangular selections as half-open source bounds
`(y0, x0, y1, x1)`, matching NumPy slicing. Normalize drag direction with min/max;
because mouse endpoints identify pixels, set each exclusive upper bound to the
larger endpoint plus one. Clip bounds to `(height, width)` before extracting:

```python
crop = image[y0:y1, x0:x1]
```

The width and height are `x1 - x0` and `y1 - y0`. Keep the selection bounds in
source coordinates; project the four pixel-edge boundaries to canvas space for
the overlay. Do not mix inclusive pixel indices with half-open crop bounds.

## 6. Source interface

Define a minimal `ImageSource` protocol:

- `shape -> tuple[int, ...]`: at least `(height, width)`; optional trailing
  channel dimensions are preserved.
- `dtype -> numpy.dtype`: source dtype, without implicit conversion.
- `revision -> Hashable`: changes whenever source pixels or interpretation
  settings change. It participates in cache invalidation.
- `read_region(row_slice, column_slice) -> ndarray`: returns only the requested
  source region, ideally as a view. A source may support NumPy slice steps
  directly; if not, core sampling may request a unit-step bounded region and
  then apply the NumPy step.

Provide `NumpySource` for an ndarray, including `numpy.memmap`. It must not copy
the array at construction. Its region method is equivalent to:

```python
array[row_slice, column_slice, ...]
```

For 2D grayscale and 3D channel-last arrays, preserve all channel dimensions.
Validate unsupported layouts at the boundary and raise a clear `ValueError`;
do not silently reinterpret planar/channel-first arrays.

Mutable sources must expose an explicit revision update (`mark_dirty()` or a
caller-provided revision). The core must not hash image contents to detect
changes.

## 7. Visible-region calculation and NumPy sampling

The viewport calculation returns a `VisibleRegion` containing:

- `source_rows: slice`
- `source_columns: slice`
- `destination_x`, `destination_y`: top-left of the returned frame in canvas
  coordinates
- `sample_step`
- `sampled_width`, `sampled_height`
- `is_empty`

Calculate each axis independently. Let source length be `L`, canvas length be
`V`, source anchor be `a`, canvas anchor be `p`, and sample step be `s`. Integer
ceiling division is `ceil_div(n, s) = -((-n) // s)`. The first and exclusive
last canvas coordinates whose samples fall inside the source are:

```text
c0 = max(0, p + ceil_div(-a, s))
c1 = min(V, p + ((L - 1 - a) // s) + 1)
```

If `c1 <= c0`, that axis is empty and the visible region is empty. Otherwise:

```text
source_start = a + (c0 - p) * s
source_stop  = source_start + (c1 - c0) * s
source_slice = slice(source_start, source_stop, s)
destination  = c0
```

Apply this once to Y (`H`, `Vh`, anchor `.y`) and once to X (`W`, `Vw`,
anchor `.x`). The first sample may not be divisible by `s`: its phase is
anchored to the source pixel under the last zoom pointer. This phase-aligned
slice is still a basic NumPy slice and does not allocate a pixel buffer.

Array access must use those bounded slices:

```python
frame = source.read_region(region.source_rows, region.source_columns)
```

This makes the number of sampled spatial pixels no greater than
`canvas_width * canvas_height`, regardless of source dimensions. For
`NumpySource`, the resulting array should share memory with the source where
possible. A contiguous display buffer may be created only for this visible
frame when the renderer requires it.

The visible-region calculation is a core public function and must not require
allocating or scanning the source image.

## 8. Display conversion and caching

The sampled source frame remains in source dtype by default. An optional,
application-supplied converter may map it to renderer-compatible pixels. The
converter receives only the active region and its display settings; it must
not receive or materialize the full source by default.

Cache only derived viewport frames, never a full resized copy of the source.
Cache identity must include:

- source identity and `revision`
- visible source slices and destination dimensions (including the zoom-anchor
  sampling phase)
- `sample_step`
- converter identity and display settings (for example intensity window,
  channel mapping, and output dtype)

Use a byte-bounded LRU with a conservative default (for example 64 MiB), and
allow applications to disable or configure it. Evict before exceeding the
budget. Keep the current displayed frame alive until the adapter replaces it;
do not let cache eviction invalidate an object still being displayed.

Mouse motion used only for status/pixel inspection must not trigger a frame
conversion. Selection overlays must be rendered separately from the image
frame so changing an ROI does not invalidate sampled image data.

## 9. Rendering contract

Core output is a `ViewportFrame` with:

- a NumPy array containing only the visible sampled region
- the destination screen position
- the source slices and sample step used to create it
- source revision and display-settings identity for diagnostics/cache checks

The adapter is responsible for copying/converting this bounded frame into its
native display object. It should update an existing canvas image item where
possible instead of deleting/recreating unrelated canvas items. Empty frames
hide or clear the displayed image without raising an error.

The core must not force the adapter to allocate a full-image texture. If an
adapter chooses to cache a larger raster for its toolkit, that cache must be
separate, bounded, and optional.

## 10. Error handling and validation

- Reject zero or negative image dimensions, canvas dimensions, or sample steps
  with explicit errors (zero-sized canvas may instead return an empty frame if
  it can occur during initial GUI layout; choose and document one policy).
- Clamp wheel zoom to configured limits.
- Return an empty region for a viewport entirely outside the image.
- Handle a source smaller than the viewport without padding allocations; the
  adapter displays its bounded frame at the specified destination.
- Reject source channel layouts that the display converter does not support.
- Propagate source read failures; do not convert them into black or stale
  pixels.
- A source revision change invalidates all frames derived from the old revision.

## 11. Performance requirements

### Hard requirements

- Pan and zoom may read/process no more than the active viewport's sampled
  region, except for a source backend's documented minimum read granularity.
- No full-image `np.repeat`, `resize`, normalization, `astype`, or `ascontiguousarray`
  in the navigation path.
- For ndarray and memmap sources, region selection is slicing-based and does
  not make a copy of the full source.
- Derived frame/cache memory is bounded by viewport dimensions and the cache
  byte limit, not by full source image dimensions.
- Updating status text or selection geometry does not recalculate pixel data.

### Benchmark targets

Benchmark on a documented reference machine using both an in-memory ndarray
and a memmap source. Use a synthetic image at least `100,000 x 100,000` where
the storage backend supports it and a `1920 x 1080` viewport.

- Visible-region calculation: p95 under 1 ms; it must be independent of source
  image area.
- In-memory uint8 viewport sampling plus frame assembly: p95 under 33 ms at a
  1920 x 1080 viewport, excluding GUI toolkit upload.
- Pan event handling: target at least 30 rendered frames per second for an
  in-memory uint8 source on the reference machine. If full toolkit upload
  misses this target, coalesce motion events and render at most once per frame.
- Peak derived-frame memory: no more than two viewport-sized output buffers
  plus the explicitly configured cache budget.

Record actual results and hardware/software versions. Treat the 33 ms target as
a benchmark gate, not a claim about every GUI backend or storage device.

## 12. Public API sketch

Names may change during implementation, but the first release should provide
equivalent functionality:

```python
source = NumpySource(array)  # accepts ndarray or memmap without copying
viewport = ViewportState(
    canvas_width=canvas_width,
    canvas_height=canvas_height,
    sample_step=1,
    max_sample_step=100,
    sample_anchor_source=Pose(0, 0),
    sample_anchor_canvas=Pose(0, 0),
)

viewport.zoom_at(pointer_y, pointer_x, direction=+1, source_shape=source.shape)
viewport.begin_pan(pointer_y, pointer_x)
viewport.pan_to(pointer_y, pointer_x)
viewport.end_pan(pointer_y, pointer_x)

region = visible_region(source.shape, viewport)
frame = make_viewport_frame(source, viewport, region=region, converter=converter)
```

`direction=+1` means wheel-up / zoom-in (decrease `sample_step`); `-1` means
wheel-down / zoom-out (increase it). `zoom_at()` must retain the exact source
pixel at the input canvas coordinate. Avoid accepting GUI event objects in this
API. The sketch uses YX arguments to match `Pose`; adapters convert toolkit XY
coordinates at the boundary.

## 13. Required tests

Tests must cover:

1. Exact source slices and destination positions for an image larger than the
   viewport, at each image edge, with positive and negative offsets.
2. Empty views when panned entirely outside each image edge.
3. Exact NumPy sample values for steps `1`, `2`, and a larger step; grayscale,
   RGB, and channel-last RGBA shapes.
4. Odd image dimensions where `ceil(width / step)` and
   `ceil(height / step)` matter.
5. `NumpySource` wrapping ndarray and memmap without source-sized copies.
6. Pointer-anchored zoom: the same source pixel maps to the same canvas
   coordinate before and after each stride change, even when the anchor is not
   divisible by the new stride; wheel direction, min/max clamping, and boundary
   no-ops.
7. Left-drag pan deltas, including negative offsets.
8. Cache hit on an identical visible frame; cache miss after source revision,
   zoom, slice, or display-setting changes; eviction at the byte limit.
9. A large synthetic/memmap source proving returned frame dimensions never
   exceed the viewport and sampling does not call a whole-image conversion.
10. Invalid dimensions, unsupported layouts, and source-read errors.

Prefer deterministic assertions for slice coordinates, output shape, and
memory sharing. Keep wall-clock performance checks in an opt-in benchmark, not
timing-sensitive unit tests.

## 14. v1 completion criteria

- The package can be imported without a GUI installation.
- A viewer can wrap an existing ndarray and render a viewport without changing
  the source array or copying it in full.
- Wheel zoom is anchored at the pointer and uses integer NumPy sampling.
- Left-button drag is implemented by the app adapter through core pan methods.
- Only the active viewport region is sampled and converted.
- Unit tests cover all required coordinate, slicing, zoom, pan, cache, and
  large-source cases.
- A short benchmark report records slicing/frame performance and memory use on
  the reference machine.
