#!/usr/bin/env python3
import logging
import tkinter as tk
from pathlib import Path
import argparse
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image, ImageTk

# ---------------- LOGGING ----------------
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
@lru_cache(maxsize=32)
def load_image(path: str):
    return np.asarray(Image.open(path))


def load_image_context(path: Path):
    image_path = path.resolve()
    image_dir = image_path.parent

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    image_list = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in exts
    )

    if not image_list:
        raise ValueError(f"No images in {image_dir}")

    image_idx = next(
        (i for i, p in enumerate(image_list) if p.resolve() == image_path),
        None,
    )

    if image_idx is None:
        raise ValueError(f"{image_path} not found in directory")

    image = load_image(str(image_path))

    return image, image_list, image_idx, image_path


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
        self.canvas.pack(fill="both", expand=True)

        self._init_statusbar()

        self.image, self.image_list, self.image_idx, self.image_path = load_image_context(Path(fpath))

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

        self._update_title()

        log.info("Initialized: %s", self.image_path)
        self.render()

    # ---------------- UI ----------------
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
        px = self._image_pixel()

        self.status.config(
            text=(
                f"mouse=({s.mouse.y},{s.mouse.x}) | "
                f"img_px=({px.y},{px.x}) | "
                f"selected=({s.selected.y},{s.selected.x}) | "
                f"origin=({s.img_origin.y},{s.img_origin.x}) | "
                f"zoom={s.zoom}"
            )
        )

    def _update_title(self):
        self.root.title(f"{self.image_path.name} — {self.image_path.parent}")

    # ---------------- INPUT ----------------
    def _bind_mouse(self):
        self.canvas.bind("<Button-1>", self._on_down)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_up)
        self.canvas.bind("<Motion>", self._on_move)
        self.canvas.bind("<Configure>", self._on_resize)

    def _bind_keys(self):
        self.root.bind("<MouseWheel>", self._on_wheel)
        self.root.bind("<Button-4>", self._on_wheel)
        self.root.bind("<Button-5>", self._on_wheel)

        self.root.bind("<Left>", self._on_prev_image)
        self.root.bind("<Right>", self._on_next_image)

    def _update_mouse(self, event):
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

        self.render()
        
    def _on_resize(self, event):
        self.canvas_size = Point(event.height, event.width)
        self.render()

    # ---------------- NAVIGATION ----------------
    def _load_image_at_index(self, idx: int):
        idx = idx % len(self.image_list)

        self.image_idx = idx
        self.image_path = self.image_list[idx]
        self.image = load_image(str(self.image_path))

        self._update_title()
        self.render()

    def _on_prev_image(self, event=None):
        self._load_image_at_index(self.image_idx - 1)

    def _on_next_image(self, event=None):
        self._load_image_at_index(self.image_idx + 1)

    # ---------------- LOGIC ----------------
    def select_anchor(self):
        s = self.state
        z = s.zoom

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

    def _image_pixel(self) -> Point:
        s = self.state
        z = s.zoom

        crop = Point(
            max(-s.img_origin.y, 0),
            max(-s.img_origin.x, 0),
        )

        base = Point(
            max(s.img_origin.y, 0),
            max(s.img_origin.x, 0),
        )

        return (s.mouse - base + crop) * z

    # ---------------- RENDER ----------------
    def _to_photo(self, arr):
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 1)
            arr = (arr * 255).astype(np.uint8)
        return ImageTk.PhotoImage(Image.fromarray(arr))

    def render(self):
        s = self.state
        z = s.zoom

        s.img_origin = s.mouse - (s.selected // z)
        origin = s.img_origin

        draw = Point(max(origin.y, 0), max(origin.x, 0))
        crop = Point(max(-origin.y, 0), max(-origin.x, 0))

        view = Point(
            self.canvas_size.y - draw.y,
            self.canvas_size.x - draw.x,
        )

        y0 = crop.y * z
        x0 = crop.x * z
        y1 = (crop.y + view.y) * z
        x1 = (crop.x + view.x) * z

        cropped = self.image[y0:y1:z, x0:x1:z]

        if cropped.size == 0:
            return

        self._tk_img = self._to_photo(cropped)

        if self._img_id is None:
            self._img_id = self.canvas.create_image(
                draw.x, draw.y, anchor="nw", image=self._tk_img
            )
        else:
            self.canvas.coords(self._img_id, draw.x, draw.y)
            self.canvas.itemconfig(self._img_id, image=self._tk_img)

        self._update_statusbar()


# ---------------- MAIN ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("image", nargs="?", default=TEST_IMAGE)
    args = p.parse_args()

    root = tk.Tk()
    App(root, fpath=args.image)
    root.mainloop()


if __name__ == "__main__":
    main()
