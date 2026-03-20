#!/usr/bin/env python3
import logging
import tkinter as tk
from pathlib import Path
import argparse
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageTk

# ---------------- LOGGING ----------------
# Structured logging helps trace coordinate issues.
# DEBUG → geometry, INFO → events, WARNING → anomalies.
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("npsee")


# ---------------- POINT ----------------

@dataclass
class Point:
    y: int
    x: int

    def __add__(self, o: "Point") -> "Point":
        return Point(self.y + o.y, self.x + o.x)

    def __sub__(self, o: "Point") -> "Point":
        return Point(self.y - o.y, self.x - o.x)

    def __floordiv__(self, k: int) -> "Point":
        return Point(self.y // k, self.x // k)

    def __mul__(self, k: int) -> "Point":
        return Point(self.y * k, self.x * k)


# ---------------- STATE ----------------

@dataclass
class AppState:
    mouse: Point
    selected: Point
    img_origin: Point
    zoom: int


# ---------------- CONFIG ----------------

CANVAS_SIZE = Point(600, 1000)
TEST_IMAGE = Path(__file__).parent.parent.parent / "data" / "test.jpg"


# ---------------- IMAGE ----------------

def load_image(path: Path):
    return np.asarray(Image.open(path))


# ---------------- APP ----------------

class App:
    def __init__(self, root, fpath=TEST_IMAGE):
        self.root = root
        self.canvas_size = CANVAS_SIZE

        self.canvas = tk.Canvas(
            root,
            width=self.canvas_size.x,
            height=self.canvas_size.y,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack()
        self._init_statusbar()

        self.image = load_image(fpath)

        self.state = AppState(
            mouse=Point(0, 0),
            selected=Point(0, 0),
            img_origin=Point(0, 0),
            zoom=2,
        )

        self._img_id = None
        self._tk_img = None

        self._bind_mouse()
        self._bind_keys()

        log.info("App initialized (image=%s, zoom=%d)", fpath, self.state.zoom)
        self.render()


    def _init_statusbar(self):
        self.status = tk.Label(
            self.root,
            anchor="w",
            relief="sunken",
            padx=6,
            pady=2,
            font=("TkDefaultFont", 9),
        )
        self.status.pack(side="bottom", fill="x")
            
    def _update_statusbar(self):
        s = self.state

        img_px = self._image_pixel()
        h, w = self.image.shape[:2]

        text = (
            f"mouse=({s.mouse.y},{s.mouse.x}) | "
            f"img_px=({img_px.y},{img_px.x}) | "
            f"selected=({s.selected.y},{s.selected.x}) | "
            f"origin=({s.img_origin.y},{s.img_origin.x}) | "
            f"zoom={s.zoom}"
        )

        self.status.config(text=text)
    # ---------------- INPUT ----------------

    def _bind_mouse(self):
        self.canvas.bind("<Button-1>", self._on_down)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_up)
        self.canvas.bind("<Motion>", self._on_move)

    def _bind_keys(self):
        self.root.bind("<MouseWheel>", self._on_wheel)
        self.root.bind("<Button-4>", self._on_wheel)
        self.root.bind("<Button-5>", self._on_wheel)

    def _update_mouse(self, event):
        # Convert Tk event coords -> canvas coords (important for accuracy)
        self.state.mouse = Point(
            int(self.canvas.canvasy(event.y)),
            int(self.canvas.canvasx(event.x)),
        )

    def _on_down(self, event):
        self._update_mouse(event)
        self.select_anchor()

    def _on_drag(self, event):
        self._update_mouse(event)

    def _on_up(self, event):
        self._update_mouse(event)
        self.render()

    def _on_move(self, event):
        self._update_mouse(event)
        self._update_statusbar()

    def _on_wheel(self, event):
        self.select_anchor()

        if event.num == 4 or event.delta > 0:
            self.state.zoom = max(1, self.state.zoom - 1)
        else:
            self.state.zoom = min(100, self.state.zoom + 1)

        log.info("zoom changed → %d", self.state.zoom)
        self.render()

    # ---------------- LOGIC ----------------

    def select_anchor(self):
        """
        Select pixel under mouse.
        Keeps selected pixel stable across zoom changes.
        """
        s = self.state
        z = s.zoom

        # Crop adjustment (when image is partially outside canvas)
        crop = Point(
            max(-s.img_origin.y, 0),
            max(-s.img_origin.x, 0),
        )

        base = Point(
            max(s.img_origin.y, 0),
            max(s.img_origin.x, 0),
        )

        pos = (s.mouse - base + crop) * z

        h, w = self.image.shape[:2]

        s.selected = Point(
            min(max(pos.y, 0), h - 1),
            min(max(pos.x, 0), w - 1),
        )

        log.debug("selected = %s", s.selected)

    # ---------------- RENDER ----------------

    def _to_photo(self, arr):
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 1)
            arr = (arr * 255).astype(np.uint8)
        return ImageTk.PhotoImage(Image.fromarray(arr))

    def render(self):
        s = self.state
        z = s.zoom

        # Compute where the image should start on canvas
        s.img_origin = s.mouse - (s.selected // z)
        origin = s.img_origin

        log.debug("origin = %s", origin)

        # Clamp drawing region to visible canvas
        draw = Point(
            max(origin.y, 0),
            max(origin.x, 0),
        )

        # Determine crop offset when image goes outside canvas
        crop = Point(
            max(-origin.y, 0),
            max(-origin.x, 0),
        )

        # Visible region size
        view = Point(
            self.canvas_size.y - draw.y,
            self.canvas_size.x - draw.x,
        )

        # Convert canvas region → image region
        y0 = crop.y * z
        x0 = crop.x * z
        y1 = (crop.y + view.y) * z
        x1 = (crop.x + view.x) * z

        log.debug("crop = y[%d:%d], x[%d:%d], zoom=%d", y0, y1, x0, x1, z)

        cropped = self.image[y0:y1:z, x0:x1:z]

        if cropped.size == 0:
            log.warning("empty crop (check bounds or zoom)")
            return

        self._tk_img = self._to_photo(cropped)

        if self._img_id is None:
            self._img_id = self.canvas.create_image(
                draw.x,
                draw.y,
                anchor="nw",
                image=self._tk_img,
            )
            log.debug("image created at %s", draw)
        else:
            self.canvas.coords(self._img_id, draw.x, draw.y)
            self.canvas.itemconfig(self._img_id, image=self._tk_img)
            log.debug("image moved to %s", draw)


    def _image_pixel(self) -> Point:
        """
        Map current mouse position → image pixel coordinates.

        This is the inverse of rendering:
        canvas → image space, accounting for:
        - origin (pan)
        - zoom
        """
        s = self.state
        z = s.zoom

        # reverse of render(): image origin on canvas
        origin = s.img_origin

        # convert mouse → image coordinate system
        y = (s.mouse.y - origin.y) * z
        x = (s.mouse.x - origin.x) * z

        return Point(y, x)

# ---------------- MAIN ----------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("image", nargs="?", default=TEST_IMAGE)
    args = p.parse_args()

    root = tk.Tk()
    root.title("npsee")

    App(root, fpath=args.image)

    root.mainloop()


if __name__ == "__main__":
    main()