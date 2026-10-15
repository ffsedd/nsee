# `nseecore` — NumPy-first Technical Specification v2.1

**Status:** implementation specification; performance goals are hypotheses until measured  
**Supersedes:** v1.1 and the earlier v2.0 architecture proposal  
**Baseline:** the existing `Pose`, `ViewerState`, and `visible_image_region()` implementation  
**Python:** ≥3.11; **runtime dependencies:** NumPy only  
**Primary domain:** 2D scientific rasters, including X-ray images and large `numpy.memmap` arrays

## 1. Executive decision

**Keep the current integer-stride rendering model.** The default data path is precisely:

```python
rows, cols, draw = visible_image_region(state.img_origin, canvas_size, state.zoom)
visible = image[rows, cols]  # Basic NumPy slicing: no pixel copy
```

No eager contiguous read, pyramid, tile scheduler, LRU cache, display converter, or worker is required for the MVP. Each is a separately justified extension **only after representative profiling**.

The core's job is to compute source-aligned view slices, preserve input/selection coordinates, and give a UI adapter access to a NumPy view and its canvas origin. Display-buffer creation and event-loop integration remain application concerns.

### Goals

1. Provide exact, predictable row/column-to-canvas integer math.
2. Create a source-aligned NumPy view in constant time independent of total image dimensions.
3. Restrict view *shape* to at most the viewport dimensions (source data clipped by NumPy).
4. Support ordinary `numpy.ndarray` and `numpy.memmap`, without copying source pixels.
5. Preserve a small, testable, GUI-independent API and existing YX coordinate conventions.
6. Make edge cases and observable performance measurable, not assumptions.

### Non-goals for MVP

- Continuous or fractional zoom, interpolation, magnification past 1:1, rotation.
- Mandatory tiles/pyramids, GPU code, caching, background threads, color mapping.
- File decoding, histogram calculation, image processing, ROI statistics, GUI code.
- Automatic stacks/volumes handling: caller selects a 2D plane first.
- Guaranteed FPS for uncached memory maps, network filesystems, or slow GUI conversions.

## 2. Terminology and coordinate invariants

Use **YX order everywhere within the core**. `Pose(y, x)` is an integer pair. GUI libraries that expose `(x, y)` convert at the adapter boundary.

| Name | Definition | Units |
|---|---|---|
| `image` | Source array, `(H, W)` or `(H, W, C)` | source pixels |
| `canvas_size` | `Pose(height, width)`; each ≥ 0 | canvas pixels |
| `mouse` | `Pose(canvas_y, canvas_x)` | canvas pixels |
| `img_origin` | Canvas position of source sample `(0, 0)` | canvas pixels |
| `zoom` | Positive integer **source pixels per canvas pixel**; sampling stride `s` | source px / canvas px |
| `selected` | Source pixel used to anchor the next zoom | source pixels |
| `sel_start`, `sel_end` | Source-space selection endpoints | source pixels |
| `draw` | Canvas position of the returned view's first sample | canvas pixels |

**Naming:** preserve the public field `zoom` for compatibility, but document it as **sampling stride**, not a graphical magnification factor. `zoom = 1` means one source pixel per displayed pixel; `zoom = 4` means every fourth source pixel is displayed. Increasing `zoom` zooms *out*.

For either axis, with `s = zoom` and canvas coordinate `c`:

```text
source_sample(c) = (c - img_origin) * s
canvas_cell(i)   = img_origin + (i // s)
```

The forward mapping selects the source index sampled by a canvas pixel. The inverse maps an arbitrary source index to the canvas **cell** representing its stride group. These are deliberately not exact algebraic inverses for every integer source index when `s > 1`.

**Required invariants**

- `zoom >= 1`; negative/zero step is invalid.
- All core coordinates, slice boundaries, and sample steps are integers.
- Source sampling uses the **global grid** `0, s, 2s, ...` on both axes, regardless of pan/crop.
- Panning does not resample, interpolate, or alter image values.
- Any returned `image[rows, cols]` uses basic slicing, preserving trailing channel dimensions.
- An off-screen viewport produces a zero-sized view in at least one dimension; no negative slice *stop* may be used to express an empty view.
- Scientific measurements always use the original image, not the navigational sample view.

## 3. Package structure and ownership

Minimal initial structure:

```text
src/nseecore/
    __init__.py        # Deliberately small stable public exports
    models.py          # Pose and ViewerState
    geometry.py        # visible_image_region, selection_canvas_bounds
    sampling.py        # Optional one-line NumPy extraction convenience

tests/
    test_geometry.py
    test_state.py
    test_sampling.py

benchmarks/
    bench_view.py       # Manual or opt-in reproducible performance probes
```

| Responsibility | Core | Application adapter |
|---|---|---|
| Integer viewport and selection math | Owns | Calls |
| Source-aligned basic slicing | Owns | Supplies array |
| File loading/memmap lifetime | Does not own | Owns |
| uint16/float contrast, lookup tables, normalization | Does not own | Owns |
| Toolkit image, texture, paint calls | Excluded | Owns |
| Mouse events, wheel sensitivity, timers | Excluded | Owns |
| Frame coalescing/stale results | Optional later | Initially owns |

Core operations must not import Qt, Tk, matplotlib, Pillow, or an event-loop framework. A `ViewportFrame` dataclass and an `ImageSource` protocol are **not mandatory** until there is a proven second backend or a need to return metadata beyond `(view, draw)`.

## 4. Normative minimal API

The following is the **reference implementation**. It preserves the supplied field names and method behavior except for three fixes: simplified coordinate conversion, nonnegative viewport extent, and correct global-grid selection projection. Validation is added where an invalid value would cause nonsensical behavior.

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Pose:
    y: int
    x: int

    def __add__(self, other: Pose) -> Pose:
        return Pose(self.y + other.y, self.x + other.x)

    def __sub__(self, other: Pose) -> Pose:
        return Pose(self.y - other.y, self.x - other.x)

    def __floordiv__(self, k: int) -> Pose:
        return Pose(self.y // k, self.x // k)

    def __mul__(self, k: int) -> Pose:
        return Pose(self.y * k, self.x * k)


@dataclass(slots=True)
class ViewerState:
    """Mutable YX coordinates and source-space selection state."""

    mouse: Pose = Pose(0, 0)
    selected: Pose = Pose(0, 0)
    img_origin: Pose = Pose(0, 0)
    zoom: int = 2
    sel_start: Pose | None = None
    sel_end: Pose | None = None

    @property
    def selection_bounds(self) -> tuple[int, int, int, int] | None:
        """Inclusive (top, left, bottom, right) source coordinates."""
        if self.sel_start is None or self.sel_end is None:
            return None
        return (
            min(self.sel_start.y, self.sel_end.y),
            min(self.sel_start.x, self.sel_end.x),
            max(self.sel_start.y, self.sel_end.y),
            max(self.sel_start.x, self.sel_end.x),
        )

    def image_pixel(self) -> Pose:
        """Source sample index under mouse; may be outside the image."""
        if self.zoom < 1:
            raise ValueError("zoom must be >= 1")
        return (self.mouse - self.img_origin) * self.zoom

    def select_anchor(self, image_shape: tuple[int, ...]) -> None:
        """Clamp current source sample to the image as a zoom anchor."""
        height, width = image_shape[:2]
        if height < 1 or width < 1:
            raise ValueError("cannot select an anchor in an empty image")
        pos = self.image_pixel()
        self.selected = Pose(
            min(max(pos.y, 0), height - 1),
            min(max(pos.x, 0), width - 1),
        )

    def update_origin(self) -> None:
        """Position the selected source stride cell under the mouse."""
        if self.zoom < 1:
            raise ValueError("zoom must be >= 1")
        self.img_origin = self.mouse - (self.selected // self.zoom)


def visible_image_region(
    origin: Pose, canvas_size: Pose, zoom: int
) -> tuple[slice, slice, Pose]:
    """Source row/column slices and canvas YX origin of first sample."""
    if zoom < 1:
        raise ValueError("zoom must be >= 1")
    if canvas_size.y < 0 or canvas_size.x < 0:
        raise ValueError("canvas dimensions must be >= 0")

    draw = Pose(max(origin.y, 0), max(origin.x, 0))
    crop = Pose(max(-origin.y, 0), max(-origin.x, 0))
    view = Pose(
        max(canvas_size.y - draw.y, 0),
        max(canvas_size.x - draw.x, 0),
    )
    start = crop * zoom
    stop = (crop + view) * zoom
    return (
        slice(start.y, stop.y, zoom),
        slice(start.x, stop.x, zoom),
        draw,
    )


def selection_canvas_bounds(
    state: ViewerState,
) -> tuple[int, int, int, int] | None:
    """Inclusive (left, top, right, bottom) canvas cell endpoints."""
    bounds = state.selection_bounds
    if bounds is None:
        return None
    if state.zoom < 1:
        raise ValueError("zoom must be >= 1")

    y1, x1, y2, x2 = bounds
    origin = state.img_origin
    step = state.zoom
    return (
        origin.x + x1 // step,
        origin.y + y1 // step,
        origin.x + x2 // step,
        origin.y + y2 // step,
    )
```

The reference is short enough to be covered completely by pytest and strict static typing. The actual package may split it across `models.py` and `geometry.py`; split by responsibility, not by a new abstraction for every function.

### NumPy extraction contract

```python
rows, cols, draw = visible_image_region(state.img_origin, canvas_size, state.zoom)
visible = image[rows, cols]  # View for ordinary ndarray/memmap basic slicing

# The GUI adapter owns all materialization and display conversion.
# Example: display = converter(visible)
```

`image` must be a 2D array or a 3D HWC array. Channel count is a concern for the adapter's converter, not for slicing. For a 3D array the two indices leave the channel axis intact. A zero-sized image may be rendered as empty, but cannot be used with `select_anchor()`.

## 5. Visible-region clipping, precisely

For each axis, define:

```text
draw  = max(origin, 0)
crop  = max(-origin, 0)
count = max(canvas_size - draw, 0)
start = crop * zoom
stop  = (crop + count) * zoom
slice(start, stop, zoom)
```

For a source index `i` produced by the slice, its canvas coordinate is:

```text
c = draw + (i - start) // zoom
```

This is equivalent to `c = origin + i // zoom` for all returned samples.

**Do not** calculate `count = canvas_size - draw` without clamping to zero: an entirely off-screen image can otherwise create a slice with a negative `stop`, which NumPy interprets relative to the **end of the source dimension**, leading to unexpectedly large reads.

### Worked edge cases

Assume 1D canvas length 4, stride 2, and source length 20:

| `origin` | `draw` | Source indices | Interpretation |
|---:|---:|---|---|
| 0 | 0 | 0, 2, 4, 6 | exactly fits viewport |
| +2 | 2 | 0, 2 | image starts inside viewport |
| -1 | 0 | 2, 4, 6, 8 | skip obscured first sample |
| +5 | 5 | none | entirely beyond viewport |
| -20 | 0 | none | entirely before viewport |

NumPy clamps out-of-source indices naturally. If the source ends before the requested stop, the view is truncated without creating extra pixels. The canvas destination may lie outside the canvas for an empty view; the adapter must simply skip painting an empty view. It may separately clip destination drawing to the widget.

### Slicing alignment

Do **not** use `image[start:stop][::zoom]` if `start` could be an arbitrary nonmultiple of `zoom`. The required sample grid is anchored at source `(0, 0)`. In the reference implementation `start` is always a multiple of `zoom`, so direct `image[rows, cols]` is unambiguous and pan-stable.

## 6. Navigation and selection semantics

### Pan

The adapter updates `img_origin` by a canvas-pixel delta. No source pixels need to be accessed during the event:

```python
state.img_origin += Pose(delta_y, delta_x)
```

Do not recompute the image origin from a stale `selected` anchor on every pan event.

### Pointer-anchored discrete zoom

For an in-image pointer and new positive integer stride:

```python
state.mouse = pointer_yx
state.select_anchor(image.shape)   # Save current sampled source index
state.zoom = new_zoom
state.update_origin()              # Anchor selected source stride cell
```

Calling `select_anchor()` **before** changing `zoom` is essential. A selected source index need not be a multiple of the new stride, so the anchor is held to the **containing canvas cell**, not at subpixel precision. This is stable integer-grid behavior, **not** continuous, exact inverse zoom. If the mouse is outside the image, clamping intentionally pulls the nearest image edge toward the pointer on zoom; adapter policy can instead ignore wheel events outside the image.

Repeated zoom cycles must not be described as mathematically drift-free. A refreshed anchor is quantized to the old sampling grid on each event. For exact source-point preservation under arbitrary repeated zoom, a richer continuous model would be required; that is out of MVP scope.

### Source-space selection

- `sel_start` and `sel_end` are **source pixel indices** (not canvas pixels).
- `selection_bounds` returns `(top, left, bottom, right)` with **inclusive** endpoints.
- `selection_canvas_bounds()` returns **canvas (left, top, right, bottom)** endpoints of the cells containing those source indices; it does not clamp them to the canvas.
- To extract an ROI for analysis, use `image[top:bottom + 1, left:right + 1]` after clipping the selection to image bounds.
- If the UI draws a full rectangular pixel-edge outline rather than two selected cell endpoints, it needs a separate explicit half-open boundary transformation and corner/edge rounding policy.
- Because striding discards pixels, a source ROI can contain pixels not visible on the screen. Never interpret a sampled view as the full ROI.

## 7. Source data and display pipeline

**Default supported arrays:** `np.ndarray` and `np.memmap`, read-only from the viewer's perspective. Input must be `(H, W)` or `(H, W, C)`; other dimensions require the application to select a plane. There is no conversion to `uint8` inside the geometry/slicing core.

For scientific arrays (`uint16`, `float32`, `float64`, etc.), distinguish:

1. **Source values:** original, unchanged numerical data.
2. **Navigation view:** strided read-only view of those values.
3. **Display buffer:** possibly contiguous, normalized/clipped/colored bytes created by the adapter.

```text
source ndarray/memmap
       |
       v
visible_image_region -> basic NumPy view  [nseecore]
       |
       v
contrast / LUT / normalize / contiguous  [application adapter]
       |
       v
GUI bitmap / texture / paint            [application adapter]
```

The adapter must keep any display buffer alive as long as the toolkit references its memory; a temporary array passed to an asynchronous painter may become invalid. The core does not take responsibility for an array or memmap being closed/replaced concurrently with display conversion.

### Sampling fidelity

`image[::zoom, ::zoom]` is *point sampling*: it can miss small bright features, isolated defective pixels, thin cracks, etc. A fast preview is not a faithful scientific reduction. If needed in a later phase, add **explicitly named** display policies (`stride`, `max`, `min`, `mean`) with benchmarked cost and boundary semantics. Never silently substitute an averaging or maximum-intensity algorithm for point sampling.

## 8. Memmap and I/O: no speculative optimization

`numpy.memmap` supports basic slice views; creating a view usually incurs little work. **Reading the selected pixels later** can trigger disk page faults, especially on a cold map or slow storage. The expression below is **not** a 2-stage contiguous fetch:

```python
window = image[y0:y1, x0:x1]  # View; no eager read
visible = window[::step, ::step]  # Still strided when touched
```

`window.copy()` *does* materialize the full window, but can read much more data than the pixels required by a strided viewport. Neither strategy is universally faster; decide using measurements on representative local SSD, cached memmap, and (if relevant) network-backed files.

Required benchmark strategies:

1. **Direct stride:** `image[rows, cols]`, then materialize the *visible* view.
2. **Contiguous-window copy:** copy the full bounding source region first, then stride (comparison only).
3. **Row/strip reads:** optional experiment if the filesystem or access pattern justifies it.

Benchmarks must report separate values for view creation, actual pixel reads/materialization, and display conversion. Do not claim `O(canvas)` **physical I/O**, a cold NVMe `p95 < 20 ms`, or guaranteed FPS solely from slice creation timing.

## 9. Cache, events, drafts, threads: deferred by default

### GUI event coalescing

Implement first in the adapter: mutate `ViewerState` on input, then schedule at most one pending repaint. Redraw from the **latest state**, rather than queueing one render per mouse event. A refresh timer or GUI-native update request is sufficient for a cheap synchronous path.

If measured display conversion/memmap reads block the GUI, introduce a later optional worker with these contracts:

- At most one running request and one replaceable pending request.
- Tag requests with monotonically increasing IDs; discard completed results older than current state.
- Worker never calls GUI APIs; presentation is marshalled to the GUI thread.
- Source replacement/revision invalidates old requests.
- No assertions that Python threads can cancel an ongoing OS page fault or NumPy operation.

### Draft frames

**Do not change `ViewerState.zoom` to create a low-resolution preview.** This changes the displayed field of view and moves cursor-anchored coordinates. If profiling calls for drafts, treat draft quality as a display-only policy that preserves viewport geometry and overlays; establish its alignment and fidelity separately.

### Cache

No cache is required while view creation and conversion are sufficiently fast. If a cache is added, cache **materialized display buffers**, not raw NumPy view objects as though their `nbytes` represented newly allocated memory. Cache keys must include source identity/generation, content revision (if mutable), geometry, sampling mode, and display-converter settings. Memory ceilings apply to owned buffers, not the operating system's page cache.

### Optional multiresolution storage

Add pyramids/tiles only if full-resolution point sampling performs poorly at common overview zoom levels, or if faithful block reductions are required. A pyramid is a distinct storage backend rather than a mandatory dependency of the ordinary ndarray code path.

## 10. Errors, edge cases, and validation

| Situation | Required behavior |
|---|---|
| `zoom <= 0` | `ValueError` |
| Negative canvas dimension | `ValueError` |
| Canvas dimension 0 | Empty slice/view |
| Image outside canvas | Empty view; never a negative slice stop |
| Canvas larger than image | NumPy truncates source slice; no padding in core |
| Partially clipped image | Correct global sample grid and draw offset |
| Empty image + `select_anchor()` | `ValueError` |
| Empty image + slicing | Empty view (allowed) |
| Mouse outside image | `image_pixel()` may return out-of-bounds; `select_anchor()` clamps |
| Selection not initialized | `None` bounds |
| Selection reversed | Bounds normalized using min/max |
| Selection lies offscreen | Return unbounded canvas endpoints; adapter clips |
| Noncontiguous or transposed source | Basic slicing still valid; no contiguity guarantee |
| Source changes during conversion | Adapter is responsible for safe lifetime/versioning |

The MVP keeps the public `Pose` operators as supplied: they accept an `int` scalar for multiplication/division. Callers must not pass floating-point zoom levels.

## 11. Acceptance tests

Tests must exercise **both values and mapping**, not only returned shape. At minimum:

### Geometry examples

```python
import numpy as np

from nseecore import Pose, visible_image_region


def test_visible_slice_is_globally_aligned() -> None:
    image = np.arange(20 * 30).reshape(20, 30)
    rows, cols, draw = visible_image_region(Pose(-1, 2), Pose(4, 5), 2)
    np.testing.assert_array_equal(image[rows, cols], image[2:10:2, 0:6:2])
    assert draw == Pose(0, 2)


def test_offscreen_does_not_read_tail_of_source() -> None:
    image = np.arange(100 * 100).reshape(100, 100)
    rows, cols, _ = visible_image_region(Pose(10, 0), Pose(4, 4), 2)
    assert image[rows, cols].size == 0
    assert rows.stop == 0


def test_zero_size_canvas() -> None:
    image = np.zeros((100, 100), dtype=np.uint8)
    rows, cols, _ = visible_image_region(Pose(-10, -10), Pose(0, 20), 2)
    assert image[rows, cols].size == 0
```

### State and selection examples

```python
from nseecore import Pose, ViewerState, selection_canvas_bounds


def test_global_grid_selection_projection() -> None:
    state = ViewerState(img_origin=Pose(-1, -1), zoom=2)
    state.sel_start = Pose(3, 3)
    state.sel_end = Pose(6, 8)
    assert selection_canvas_bounds(state) == (0, 0, 3, 2)


def test_zoom_anchor_cell() -> None:
    state = ViewerState(mouse=Pose(10, 12), img_origin=Pose(0, 2), zoom=2)
    state.select_anchor((100, 100))
    anchor = state.selected
    state.zoom = 3
    state.update_origin()
    assert state.mouse == state.img_origin + (anchor // state.zoom)
```

### Required property-based or randomized tests

- Random source `(H, W)`, canvas `(Vh, Vw)`, signed origins, and positive strides: compare returned pixels/draw position with a slow reference loop that checks each canvas coordinate independently.
- Verify every sampled source index is divisible by `zoom`.
- Verify image view dimensions never exceed corresponding canvas dimensions.
- Verify empty views for every entirely off-screen geometry.
- Verify `selection_canvas_bounds()` equals `origin + source // zoom` on both axes, including negative origins.
- Verify 2D grayscale and 3D HWC arrays preserve dtype and channels and (for basic NumPy slicing) share source memory when nonempty.
- Verify anchor behavior at boundaries, near source edges, and with out-of-image pointer positions.
- Optional: assert direct memmap slicing creates a view, and independently time touching/copying its pixels.

## 12. Benchmark methodology and acceptance gates

Use stable hardware, a fixed viewport (e.g., 1920×1080), several source shapes, zoom levels 1/2/4/8/16/64, and report NumPy version, dtype, C/F contiguity, page-cache state, storage type, and conversion policy. Benchmark creation and materialization separately.

| Stage | Measurement | Initial target / gate |
|---|---|---|
| Geometry | `visible_image_region()` latency | p95 < 0.1 ms on representative workstation; record actual |
| View creation | `image[rows, cols]` latency | p95 < 0.1 ms in warm normal cases; record actual |
| Visible view size | `view.shape[:2]` | ≤ canvas height × width, always |
| Slicing memory | Python-owned pixel-buffer allocations | Zero newly allocated pixel buffers on basic slicing |
| Display materialization | `np.ascontiguousarray(view)` or actual converter | Report warm/cold p50/p95 and bytes copied; no universal bound |
| Full render | View + conversion + GUI handoff | Application-defined frame budget; initial goal >30 FPS when data are warm |
| Navigation burst | ≥100 events in one refresh interval | No unbounded render queue; last state wins |
| Offline data integrity | Image equality/checksum before and after | Original array unchanged |

*Measurement note:* `.nbytes` on a NumPy **view** is its logical span, not bytes allocated by slicing. RSS deltas and explicit copy counts must distinguish views from owned buffers. OS page-cache memory and filesystem I/O are separate budgets.

A CPU/view benchmark alone is not evidence for a disk throughput claim. Cold-I/O benchmarks should clearly state how cold cache was established and avoid flushing unrelated system caches on shared workstations.

## 13. Implementation sequence

**Phase 1 — Extract the existing viewer core (required).**

1. Move `Pose` and `ViewerState` to `models.py`, retaining field/method names.
2. Put the two pure geometry helpers in `geometry.py`.
3. Apply the explicit off-screen clipping and selection projection fixes.
4. Export only the four symbols needed by the existing GUI: `Pose`, `ViewerState`, `visible_image_region`, `selection_canvas_bounds`.
5. Add deterministic and randomized tests plus strict typing/linting.

**Phase 2 — Measure (required before new infrastructure).**

1. Capture timings in the *existing GUI* for state update, view creation, memmap access, converter, and paint.
2. Compare cold/warm memmap, source dtype, viewport size, and zoom stride.
3. Decide whether the bottleneck is storage, conversion, or GUI repaint.

**Phase 3 — Add only a proven optimization (optional).**

- Too many paint requests → adapter coalescing.
- Slow scientific dtype conversion → vectorized LUT/normalization or cached converted frames.
- Cold sparse I/O at high stride → benchmark an optional pyramid or alternative storage layout.
- GUI blocked by long conversion → latest-request-wins worker with stale-result suppression.
- Overview loses important fine features → explicit max/min/mean display-reduction mode.

### Definition of done

- Existing viewer can import the extracted API without changing its YX coordinate model.
- Correct rendering for positive, negative, clipped, and fully off-screen origins.
- Selection projection is correct under negative origins and arbitrary positive stride.
- All tests pass under `uv run pytest`; `uv run ruff check .`; `uv run mypy src` with the project's configured strictness.
- The default path creates only a NumPy view, not a full-image or full-source-window copy.
- Representative benchmark numbers are recorded; unsupported fixed-FPS/disk claims are absent.

## 14. Change log vs previous proposals

| Previous proposal | v2.1 decision | Rationale |
|---|---|---|
| 2-stage contiguous memmap I/O always | **Remove** | A basic slice does not eagerly fetch data; extra copy may waste I/O |
| Float-scale transform as mandatory | **Defer** | Breaks the existing fast, integer-aligned sampling model |
| Pyramid/tiled reader by default | **Defer** | Additional storage/build complexity without an established bottleneck |
| Built-in `ViewportFrame` and cache | **Defer** | View and draw position already meet MVP needs |
| Forced core-owned GUI frame control | **Defer** | Event-loop scheduling belongs to the application initially |
| Auto-double `zoom` for draft dragging | **Remove** | Changes field of view and breaks pointer-anchored navigation |
| “Drift-free” integer zoom | **Correct description** | Integer cell anchoring has quantization by design |
| Source-size-independent physical I/O | **Withdraw guarantee** | View creation is O(1); cold page-fault behavior depends on storage |
| Fixed cold NVMe p95 gate | **Replace with measurements** | Not portable across data layouts/hardware |

---

**Guiding constraint:** preserve the simple, fast `image[rows, cols]` pipeline until profiling demonstrates that a more complex implementation is worth its maintenance cost.
