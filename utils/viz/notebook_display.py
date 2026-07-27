"""Notebook helpers for displaying SVG figures in a fixed-height grid."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Iterable, Optional, Union

PathLike = Union[str, Path]


def display_svg_grid(
    paths: Iterable[PathLike],
    n_cols: Optional[int] = None,
    max_width: int = 280,
    height: Optional[int] = None,
) -> None:
    """Display SVG files in a notebook table grid with consistent row height.

    Parameters
    ----------
    paths
        SVG file paths (missing / empty lists print a short notice).
    n_cols
        Number of columns. Defaults to ``min(3, n_paths)``.
    max_width
        Reserved for API compatibility with notebook call sites; layout uses
        full cell width with fixed ``height``.
    height
        Display height in pixels for every image. Defaults to ``max_width``.
    """
    from IPython.display import HTML, display

    paths = [Path(p) for p in paths]
    if not paths:
        print("No SVG files found to display.")
        return
    if n_cols is None:
        n_cols = min(3, len(paths))
    if height is None:
        height = max_width

    rows = []
    for i in range(0, len(paths), n_cols):
        cells = []
        for path in paths[i : i + n_cols]:
            if not path.is_file():
                cells.append(
                    f'<td style="width:{100 / n_cols:.2f}%;padding:4px;text-align:center;'
                    f'vertical-align:middle;color:#888">Missing: {path.name}</td>'
                )
                continue
            uri = "data:image/svg+xml;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
            cells.append(
                f'<td style="width:{100 / n_cols:.2f}%;padding:4px;text-align:center;vertical-align:middle">'
                f'<div style="height:{height}px;width:100%;overflow-x:auto;overflow-y:hidden;'
                f'display:flex;align-items:center;justify-content:center">'
                f'<img src="{uri}" style="height:{height}px;width:auto;max-width:none;display:block;"/>'
                f"</div></td>"
            )
        cells.extend(f'<td style="width:{100 / n_cols:.2f}%"></td>' for _ in range(n_cols - len(cells)))
        rows.append("<tr>" + "".join(cells) + "</tr>")
    display(HTML(f'<table style="width:100%;table-layout:fixed;border-collapse:collapse"><tbody>{"".join(rows)}</tbody></table>'))
