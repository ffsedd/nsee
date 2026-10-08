from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Pose:
    y: int
    x: int

    def __add__(self, o: "Pose") -> "Pose":
        return Pose(self.y + o.y, self.x + o.x)

    def __sub__(self, o: "Pose") -> "Pose":
        return Pose(self.y - o.y, self.x - o.x)

    def __floordiv__(self, k: int) -> "Pose":
        return Pose(self.y // k, self.x // k)

    def __mul__(self, k: int) -> "Pose":
        return Pose(self.y * k, self.x * k)


@dataclass(slots=True)
class ViewerState:
    """Coordinates and selection state shared by input and rendering."""

    mouse: Pose = Pose(0, 0)
    selected: Pose = Pose(0, 0)
    img_origin: Pose = Pose(0, 0)
    zoom: int = 2
    sel_start: Pose | None = None
    sel_end: Pose | None = None

    @property
    def selection_bounds(self) -> tuple[int, int, int, int] | None:
        if self.sel_start is None or self.sel_end is None:
            return None
        return (
            min(self.sel_start.y, self.sel_end.y),
            min(self.sel_start.x, self.sel_end.x),
            max(self.sel_start.y, self.sel_end.y),
            max(self.sel_start.x, self.sel_end.x),
        )

    def image_pixel(self) -> Pose:
        crop = Pose(max(-self.img_origin.y, 0), max(-self.img_origin.x, 0))
        base = Pose(max(self.img_origin.y, 0), max(self.img_origin.x, 0))
        return (self.mouse - base + crop) * self.zoom

    def select_anchor(self, image_shape: tuple[int, ...]) -> None:
        pos = self.image_pixel()
        height, width = image_shape[:2]
        self.selected = Pose(
            min(max(pos.y, 0), height - 1),
            min(max(pos.x, 0), width - 1),
        )

    def update_origin(self) -> None:
        self.img_origin = self.mouse - (self.selected // self.zoom)


def visible_image_region(
    origin: Pose, canvas_size: Pose, zoom: int
) -> tuple[slice, slice, Pose]:
    """Return sampled image slices and their canvas position."""
    draw = Pose(max(origin.y, 0), max(origin.x, 0))
    crop = Pose(max(-origin.y, 0), max(-origin.x, 0))
    view = Pose(canvas_size.y - draw.y, canvas_size.x - draw.x)
    y0, x0 = crop.y * zoom, crop.x * zoom
    y1, x1 = (crop.y + view.y) * zoom, (crop.x + view.x) * zoom
    return slice(y0, y1, zoom), slice(x0, x1, zoom), draw


def selection_canvas_bounds(state: ViewerState) -> tuple[int, int, int, int] | None:
    bounds = state.selection_bounds
    if bounds is None:
        return None
    y1, x1, y2, x2 = bounds
    draw = Pose(max(state.img_origin.y, 0), max(state.img_origin.x, 0))
    crop = Pose(max(-state.img_origin.y, 0), max(-state.img_origin.x, 0))
    return (
        (x1 - crop.x) // state.zoom + draw.x,
        (y1 - crop.y) // state.zoom + draw.y,
        (x2 - crop.x) // state.zoom + draw.x,
        (y2 - crop.y) // state.zoom + draw.y,
    )
