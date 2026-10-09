from __future__ import annotations

import argparse
import logging
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

from .imagelist import ImageList
from .io import save_image
from .logger import setup_logger
from .ops import rotate_jpeg_lossless
from .view import ImageView
from nseecore import NumpySource, Pose, ViewerState

log = setup_logger("nsee")

CANVAS_SIZE = Pose(600, 1000)
TEST_IMAGE = Path.home() / ".local/share/icons/hicolor/128x128/apps/nsee.png"


class App:
    """Coordinate user actions and image data; rendering lives in ImageView."""

    def __init__(self, root: tk.Tk, fpath: str | Path) -> None:
        path = Path(fpath).expanduser()
        directory = path if path.is_dir() else path.parent
        self.imagelist = ImageList(directory)
        if path.is_file():
            self.imagelist.refresh(current=path)

        self.image_path = self.imagelist.current
        self.image = self.imagelist.load()
        self.source = NumpySource(self.image)
        self.state = ViewerState()
        self.view = ImageView(root, CANVAS_SIZE)
        self.canvas = self.view.canvas
        self._bind_mouse()
        self._bind_keys()
        self.view.set_title(self.image_path)
        log.info("Initialized: %s", self.image_path)
        self._render()

    def _bind_mouse(self) -> None:
        self.canvas.bind("<Button-1>", self._on_down)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_up)
        self.canvas.bind("<Button-3>", self._on_right_down)
        self.canvas.bind("<B3-Motion>", self._on_right_drag)
        self.canvas.bind("<ButtonRelease-3>", self._on_right_up)
        self.canvas.bind("<Motion>", self._on_move)
        self.canvas.bind("<Configure>", self._on_resize)

    def _bind_keys(self) -> None:
        root = self.view.root
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", self._on_wheel)
        self.canvas.bind("<Button-5>", self._on_wheel)
        root.bind("<Left>", self._on_prev_image)
        root.bind("<Right>", self._on_next_image)
        root.bind("c", self._on_crop)
        root.bind("<Control-s>", self._on_save)
        root.bind("s", self._on_save_as)
        root.bind("r", self._on_rotate_right)
        self.canvas.focus_set()

    def _update_mouse(self, event) -> None:
        self.state.set_mouse(
            Pose(int(self.canvas.canvasy(event.y)), int(self.canvas.canvasx(event.x)))
        )

    def _on_down(self, event) -> None:
        self._update_mouse(event)
        self.state.begin_pan(self.state.mouse)

    def _on_drag(self, event) -> None:
        self._update_mouse(event)
        self.state.pan_to(self.state.mouse)
        self._schedule_render()

    def _on_up(self, event) -> None:
        self._update_mouse(event)
        self.state.end_pan(self.state.mouse)
        self._render()

    def _on_move(self, event) -> None:
        self._update_mouse(event)
        self.view.update_status(self.state, self.image.shape)

    def _on_wheel(self, event) -> None:
        self._update_mouse(event)
        if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
            direction = 1
        else:
            direction = -1
        if self.state.zoom_at(self.state.mouse, direction, self.image.shape):
            self._render()
        else:
            self.view.update_status(self.state, self.image.shape)

    def _on_resize(self, event) -> None:
        self.view.canvas_size = Pose(event.height, event.width)
        self._schedule_render()

    def _on_right_down(self, event) -> None:
        self._update_mouse(event)
        point = self._clamped_image_pixel()
        self.state.sel_start = point
        self.state.sel_end = point

    def _on_right_drag(self, event) -> None:
        self._update_mouse(event)
        self.state.sel_end = self._clamped_image_pixel()
        self._schedule_render()

    def _on_right_up(self, event) -> None:
        self._update_mouse(event)
        self.state.sel_end = self._clamped_image_pixel()
        self._render()

    def _clamped_image_pixel(self) -> Pose:
        point = self.state.image_pixel()
        height, width = self.image.shape[:2]
        return Pose(
            min(max(point.y, 0), height - 1),
            min(max(point.x, 0), width - 1),
        )

    def _on_crop(self, event=None) -> None:
        bounds = self.state.selection_bounds(self.image.shape)
        if bounds is None:
            return
        y0, x0, y1, x1 = bounds
        self.image = self.image[y0:y1, x0:x1].copy()
        self.source = NumpySource(self.image)
        self.state.reset_view(self.state.zoom)
        self._render()

    def _on_save(self, event=None) -> None:
        self._save()

    def _on_save_as(self, event=None) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All", "*.*")],
        )
        if path:
            self._save(Path(path))
            self.image_path = Path(path)
            self.view.set_title(self.image_path)

    def _save(self, path: Path | None = None) -> None:
        target = path or self.image_path
        try:
            save_image(self.image, target)
        except (OSError, ValueError) as exc:
            log.exception("Unable to save image: %s", target)
            messagebox.showerror("Save failed", str(exc), parent=self.view.root)
            return
        log.info("Saved: %s", target)

    def _on_rotate_right(self, event=None) -> None:
        path = self.image_path
        try:
            rotated = rotate_jpeg_lossless(path)
        except (OSError, RuntimeError) as exc:
            log.exception("Unable to rotate image: %s", path)
            messagebox.showerror("Rotate failed", str(exc), parent=self.view.root)
            return
        if not rotated:
            log.warning("Rotate is only supported for JPEG images: %s", path)
            return
        self._load_current()

    def _on_prev_image(self, event=None) -> None:
        self._navigate(-1)

    def _on_next_image(self, event=None) -> None:
        self._navigate(1)

    def _navigate(self, step: int) -> None:
        self.imagelist.refresh(current=self.image_path)
        if step < 0:
            self.imagelist.prev()
        else:
            self.imagelist.next()
        self._load_current()

    def _load_current(self) -> None:
        self.image_path = self.imagelist.current
        self.image = self.imagelist.load()
        self.source = NumpySource(self.image)
        self.view.set_title(self.image_path)
        self.state.reset_view(self.state.zoom)
        self._render()

    def _render(self) -> None:
        self.view.render(self.source, self.state)

    def _schedule_render(self) -> None:
        if getattr(self, "_render_pending", False):
            return
        self._render_pending = True
        self.view.root.after_idle(self._render_scheduled)

    def _render_scheduled(self) -> None:
        self._render_pending = False
        self._render()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", default=TEST_IMAGE)
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    log.setLevel(logging.DEBUG if args.debug else logging.INFO)
    root = tk.Tk()
    App(root, fpath=args.image)
    root.mainloop()


if __name__ == "__main__":
    main()
