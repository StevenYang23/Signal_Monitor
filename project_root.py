"""Locate Signal_Monitor repo root from notebook cwd or file path."""
from __future__ import annotations

import sys
from pathlib import Path

MARKER = "vol_surface.py"


def find_project_root(*starts: Path | str | None) -> Path:
    """Walk up from notebook path and/or cwd until vol_surface.py is found."""
    candidates: list[Path] = []
    seen: set[Path] = set()

    for start in starts:
        if start is None:
            continue
        path = Path(start).resolve()
        if path.suffix == ".ipynb":
            path = path.parent
        if path not in seen:
            seen.add(path)
            candidates.append(path)

    cwd = Path.cwd().resolve()
    if cwd not in seen:
        candidates.append(cwd)

    for start in candidates:
        for path in (start, *start.parents):
            if (path / MARKER).is_file():
                return path

    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Project root not found (no {MARKER} above: {tried}). "
        "Open the Signal_Monitor folder in Cursor/Jupyter."
    )


def setup_path(*starts: Path | str | None) -> Path:
    """Return repo root and ensure it is on sys.path."""
    root = find_project_root(*starts)
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root
