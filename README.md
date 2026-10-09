# nsee

A small Tkinter image viewer for browsing and cropping image files.

## Run

Install with `uv sync`, then launch an image or a directory:

```sh
uv run nsee path/to/image.jpg
uv run nsee path/to/image-directory
```

The viewer supports PNG, JPEG, BMP, and TIFF images. Use the arrow keys to move
through images in the current directory, the mouse wheel to change the integer
zoom level, and the right mouse button to select a crop. Press `c` to crop, `s`
to save as, or `Ctrl+S` to save over the current image. Press `r` to rotate a
JPEG losslessly; this requires `jpegtran` on `PATH`.

Pass `--debug` to enable debug logging.

## Viewer core

The GUI uses the GUI-independent `nseecore` package for integer-stride zoom,
pointer anchoring, pan state, half-open selections, and active-view NumPy
slicing. `src/nsee/view.py` shows how to adapt its `ViewportFrame` to Tk without
creating a full-image display copy.
