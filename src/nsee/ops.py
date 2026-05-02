from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

JPEG_EXTS = {".jpg", ".jpeg"}


def rotate_jpeg_lossless(path: Path) -> bool:
    """
    Rotate JPEG 90° clockwise losslessly using jpegtran.

    Returns:
        True if rotation succeeded.
        False if file unsupported.
    """
    if path.suffix.lower() not in JPEG_EXTS:
        return False

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(tmp_fd)

    try:
        cmds = [
            "jpegtran",
            "-rotate",
            "90",
            "-copy",
            "all",
            "-outfile",
            tmp_path,
            str(path),
        ]
        # print(cmds)
        subprocess.run(
            cmds,
            check=True,
        )

        os.replace(tmp_path, path)
        # print(f"Rotated {path} losslessly")
        return True

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
