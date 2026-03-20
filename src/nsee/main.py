#!/usr/bin/env python3
import logging
import tkinter as tk
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk

CANVAS_SIZE = (600, 1000)
TEST_IMAGE = Path(__file__).parent.parent.parent / "data" / "test.jpg"

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


# ---------------- IMAGE ----------------

def load_image(path: Path) -> np.ndarray:
    img = np.asarray(Image.open(path))
    log.debug(f"Loaded image {img.shape} {img.dtype}")
    return img


# ---------------- APP ----------------

class App:
    def __init__(self, root, fpath=TEST_IMAGE):
        self.root = root

        self.canvas_size = CANVAS_SIZE

        self.canvas = tk.Canvas(
            root,
            width=self.canvas_size[1],
            height=self.canvas_size[0],
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack()

        self._create_status_label()

        # image
        self.image = load_image(fpath)

        # state
        self.mouse_y = 0
        self.mouse_x = 0

        self.z_image_on_canvas_y = 0
        self.z_image_on_canvas_x = 0

        self.selected_y = 0
        self.selected_x = 0

        self.zoom = 2

        self._bind_mouse()
        self._bind_keys()

        self.render()

    # ---------------- INPUT ----------------

    def _bind_mouse(self):
        self.canvas.bind("<Button-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas.bind("<Motion>", self._on_mouse_move)

    def _bind_keys(self):
        self.root.bind("<MouseWheel>", self._on_mouse_wheel)
        self.root.bind("<Button-4>", self._on_mouse_wheel)
        self.root.bind("<Button-5>", self._on_mouse_wheel)

    def _update_mouse(self, event):
        self.mouse_x = int(self.canvas.canvasx(event.x))
        self.mouse_y = int(self.canvas.canvasy(event.y))

    def _on_mouse_down(self, event):
        self._update_mouse(event)
        self.select_anchor_pixel()
        self._update_status()

    def _on_mouse_drag(self, event):
        self._update_mouse(event)
        self._update_status()

    def _on_mouse_up(self, event):
        self._update_mouse(event)
        self.render()
        self._update_status()

    def _on_mouse_move(self, event):
        self._update_mouse(event)
        self._update_status()

    def _on_mouse_wheel(self, event):
        self.select_anchor_pixel()

        if event.num == 4 or event.delta > 0:
            self.zoom = max(1, self.zoom - 1)
        else:
            self.zoom = min(100, self.zoom + 1)

        self.render()
        self._update_status()

    # ---------------- STATUS ----------------

    def _create_status_label(self):
        self.status_label = tk.Label(self.root, text="")
        self.status_label.pack()

    def _update_status(self):
        self.status_label.config(
            text=(
                f"zoom:{self.zoom} "
                f"mouse:({self.mouse_y},{self.mouse_x}) "
                f"selected:({self.selected_y},{self.selected_x}) "
                f"img_origin:({self.z_image_on_canvas_y},{self.z_image_on_canvas_x})"
            )
        )

    # ---------------- COORDINATES ----------------

    def select_anchor_pixel(self):
        z = self.zoom

        vy, vx = self.z_image_on_canvas_y, self.z_image_on_canvas_x

        crop_y = max(-vy, 0)
        crop_x = max(-vx, 0)

        pos_y = self.mouse_y - max(vy, 0)
        pos_x = self.mouse_x - max(vx, 0)

        pos_y = (pos_y + crop_y) * z
        pos_x = (pos_x + crop_x) * z

        h, w = self.image.shape[:2]

        self.selected_y = min(max(pos_y, 0), h - 1)
        self.selected_x = min(max(pos_x, 0), w - 1)

    def mouse_pos_on_image(self):
        z = self.zoom

        vy, vx = self.z_image_on_canvas_y, self.z_image_on_canvas_x

        crop_y = max(-vy, 0)
        crop_x = max(-vx, 0)

        pos_y = self.mouse_y - max(vy, 0)
        pos_x = self.mouse_x - max(vx, 0)

        pos_y = (pos_y + crop_y) * z
        pos_x = (pos_x + crop_x) * z

        h, w = self.image.shape[:2]

        return (
            min(max(pos_y, 0), h - 1),
            min(max(pos_x, 0), w - 1),
        )

    # ---------------- RENDER ----------------

    def photoimage_from_array(self, arr):
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 1)
            arr = (arr * 255).astype(np.uint8)

        return ImageTk.PhotoImage(Image.fromarray(arr))

    def render(self):
        z = self.zoom

        vy = self.mouse_y - self.selected_y // z
        vx = self.mouse_x - self.selected_x // z

        self.z_image_on_canvas_y = vy
        self.z_image_on_canvas_x = vx

        draw_y = max(vy, 0)
        draw_x = max(vx, 0)

        crop_y = max(-vy, 0)
        crop_x = max(-vx, 0)

        view_y = self.canvas_size[0] - draw_y
        view_x = self.canvas_size[1] - draw_x

        y0 = crop_y * z
        x0 = crop_x * z
        y1 = (crop_y + view_y) * z
        x1 = (crop_x + view_x) * z

        cropped = self.image[y0:y1:z, x0:x1:z]

        if cropped.size == 0:
            return

        self._tk_img = self.photoimage_from_array(cropped)

        if not hasattr(self, "_img_id"):
            self._img_id = self.canvas.create_image(
                draw_x,
                draw_y,
                anchor="nw",
                image=self._tk_img,
            )
        else:
            self.canvas.coords(self._img_id, draw_x, draw_y)
            self.canvas.itemconfig(self._img_id, image=self._tk_img)


# ---------------- MAIN ----------------

if __name__ == "__main__":
    root = tk.Tk()
    root.title("npsee")
    App(root)
    root.mainloop()