from __future__ import annotations

import os
import shutil
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

    executable = shutil.which("jpegtran")
    if executable is None:
        raise RuntimeError("jpegtran is required for lossless JPEG rotation")

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(tmp_fd)

    try:
        cmds = [
            executable,
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
            capture_output=True,
            text=True,
        )

        os.replace(tmp_path, path)
        # print(f"Rotated {path} losslessly")
        return True

    except subprocess.CalledProcessError as exc:
        details = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError(f"jpegtran failed: {details}") from exc
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
