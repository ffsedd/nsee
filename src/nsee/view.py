from __future__ import annotations

import tkinter as tk
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk

from .geometry import Pose, ViewerState, selection_canvas_bounds


class ImageView:
    """Tk canvas that displays the image, selection, and status line."""

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
        self._source_image: np.ndarray | None = None
        self._cached_zoom: int | None = None

    def set_title(self, path: Path) -> None:
        self.root.title(f"{path.name} — {path.parent}")

    @staticmethod
    def _to_photo(array: np.ndarray) -> ImageTk.PhotoImage:
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 1)
            array = (array * 255).astype(np.uint8)
        return ImageTk.PhotoImage(Image.fromarray(array))

    def render(self, image: np.ndarray, state: ViewerState) -> None:
        state.update_origin()
        if image is not self._source_image or state.zoom != self._cached_zoom:
            self._tk_image = self._to_photo(image[:: state.zoom, :: state.zoom])
            self._source_image = image
            self._cached_zoom = state.zoom

        if (
            self._tk_image is not None
            and self._tk_image.width()
            and self._tk_image.height()
        ):
            if self._image_id is None:
                self._image_id = self.canvas.create_image(
                    state.img_origin.x,
                    state.img_origin.y,
                    anchor="nw",
                    image=self._tk_image,
                    tags="image",
                )
            else:
                self.canvas.coords(
                    self._image_id, state.img_origin.x, state.img_origin.y
                )
                self.canvas.itemconfig(
                    self._image_id, image=self._tk_image, state="normal"
                )
        elif self._image_id is not None:
            self.canvas.itemconfig(self._image_id, state="hidden")

        self._draw_selection(state)
        self.update_status(state)

    def update_status(self, state: ViewerState) -> None:
        pixel = state.image_pixel()
        bounds = state.selection_bounds
        if bounds is None:
            selection_text = "None"
        else:
            y1, x1, y2, x2 = bounds
            selection_text = f"{y1}:{x1} → {y2}:{x2} | size=({y2 - y1},{x2 - x1})"
        self.status.config(
            text=(
                f"mouse=({state.mouse.y},{state.mouse.x}) | "
                f"img_px=({pixel.y},{pixel.x}) | "
                f"selected=({state.selected.y},{state.selected.x}) | "
                f"sel_rect={selection_text} | "
                f"origin=({state.img_origin.y},{state.img_origin.x}) | "
                f"zoom={state.zoom}"
            )
        )

    def _draw_selection(self, state: ViewerState) -> None:
        self.canvas.delete("selection")
        bounds = selection_canvas_bounds(state)
        if bounds is None:
            return
        x1, y1, x2, y2 = bounds
        self.canvas.create_rectangle(
            x1,
            y1,
            x2,
            y2,
            outline="red",
            width=1,
            tags="selection",
        )
