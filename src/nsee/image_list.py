from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .io import load_image


@dataclass
class ImageList:
    directory: Path
    exts: set[str] = field(
        default_factory=lambda: {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )

    paths: list[Path] = field(default_factory=list)
    index: int = 0

    def refresh(self, current: Path | None = None) -> None:
        self.paths = sorted(p for p in self.directory.iterdir() if p.suffix.lower() in self.exts)

        if not self.paths:
            raise ValueError(f"No images in {self.directory}")

        if current:
            current = current.resolve()
            for i, p in enumerate(self.paths):
                if p.resolve() == current:
                    self.index = i
                    return

        self.index = min(self.index, len(self.paths) - 1)

    @property
    def current(self) -> Path:
        return self.paths[self.index]

    def load(self) -> np.ndarray:
        return load_image(str(self.current))

    def next(self) -> Path:
        self.index = (self.index + 1) % len(self.paths)
        return self.current

    def prev(self) -> Path:
        self.index = (self.index - 1) % len(self.paths)
        return self.current