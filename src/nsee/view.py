from __future__ import annotations

import tkinter as tk
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk

from nseecore import FrameCache, ImageSource, ViewerState, make_viewport_frame
from nseecore import Pose, selection_canvas_bounds


class ImageView:
    """Tk adapter demonstrating viewport-only rendering with ``nseecore``."""

    def __init__(self, root: tk.Tk, size: Pose) -> None:
        self.root = root
        self.canvas_size = size
        self.canvas = tk.Canvas(
            root,
            width=size.x,
            height=size.y,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(fill="both", expand=True)
        self.status = tk.Label(
            root,
            anchor="w",
            relief="sunken",
            padx=6,
            pady=2,
            font=("TkDefaultFont", 9),
        )
        self.status.pack(side="bottom", fill="x")
        self._image_id: int | None = None
        self._tk_image: ImageTk.PhotoImage | None = None
        self._source_identity: object | None = None
        self._frame_cache = FrameCache(max_bytes=64 * 1024 * 1024)
        self._converter = self._to_display_array

    def set_title(self, path: Path) -> None:
        self.root.title(f"{path.name} — {path.parent}")

    @staticmethod
    def _to_display_array(array: np.ndarray) -> np.ndarray:
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 1)
            array = (array * 255).astype(np.uint8)
        return np.ascontiguousarray(array)

    @staticmethod
    def _to_photo(array: np.ndarray) -> ImageTk.PhotoImage:
        return ImageTk.PhotoImage(Image.fromarray(array))

    def render(self, source: ImageSource, state: ViewerState) -> None:
        if source.cache_identity is not self._source_identity:
            self._frame_cache.clear()
            self._source_identity = source.cache_identity

        frame = make_viewport_frame(
            source,
            self.canvas_size,
            state,
            converter=self._converter,
            cache=self._frame_cache,
            display_key=("uint8", "nsee-display-v1"),
        )

        if frame.array.size:
            self._tk_image = self._to_photo(frame.array)
            if self._image_id is None:
                self._image_id = self.canvas.create_image(
                    frame.draw.x,
                    frame.draw.y,
                    anchor="nw",
                    image=self._tk_image,
                    tags="image",
                )
            else:
                self.canvas.coords(self._image_id, frame.draw.x, frame.draw.y)
                self.canvas.itemconfig(
                    self._image_id, image=self._tk_image, state="normal"
                )
        elif self._image_id is not None:
            self.canvas.itemconfig(self._image_id, state="hidden")

        self._draw_selection(state, source.shape)
        self.update_status(state, source.shape)

    def update_status(
        self, state: ViewerState, image_shape: tuple[int, ...]
    ) -> None:
        pixel = state.image_pixel()
        bounds = state.selection_bounds(image_shape)
        if bounds is None:
            selection_text = "None"
        else:
            y0, x0, y1, x1 = bounds
            selection_text = f"{y0}:{x0} → {y1}:{x1} | size=({y1 - y0},{x1 - x0})"
        self.status.config(
            text=(
                f"mouse=({state.mouse.y},{state.mouse.x}) | "
                f"img_px=({pixel.y},{pixel.x}) | "
                f"sel_rect={selection_text} | "
                f"anchor=({state.sample_anchor_source.y},{state.sample_anchor_source.x}) | "
                f"zoom_step={state.zoom}"
            )
        )

    def _draw_selection(
        self, state: ViewerState, image_shape: tuple[int, ...]
    ) -> None:
        self.canvas.delete("selection")
        bounds = selection_canvas_bounds(state, image_shape)
        if bounds is None:
            return
        x0, y0, x1, y1 = bounds
        self.canvas.create_rectangle(
            x0,
            y0,
            x1,
            y1,
            outline="red",
            width=1,
            tags="selection",
        )
