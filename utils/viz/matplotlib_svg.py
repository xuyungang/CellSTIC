"""Matplotlib SVG export options for external vector editors (e.g. Adobe Illustrator)."""

from __future__ import annotations

from pathlib import Path
from typing import Union


def path_wants_svg(path: Union[str, Path]) -> bool:
    return str(path).lower().endswith(".svg")


def savefig(fig, path: Union[str, Path], *, dpi: int = 300, **kwargs) -> None:
    """
    Save a matplotlib Figure; if ``path`` ends with ``.svg``, use ``svg.fonttype = "none"``
    in a temporary rc_context so text stays editable in vector tools.
    """
    import matplotlib as mpl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path_wants_svg(path):
        with mpl.rc_context({"svg.fonttype": "none"}):
            fig.savefig(path, dpi=dpi, **kwargs)
    else:
        fig.savefig(path, dpi=dpi, **kwargs)


def configure_matplotlib_svg_for_illustrator() -> None:
    """Prefer real ``<text>`` nodes over glyph paths + ``<use xlink:href>`` refs.

    Matplotlib defaults to ``svg.fonttype = "path"``, which embeds font outlines under
    ``<defs>`` and references them with ``<use>``. Illustrator often cannot treat those
    as editable text. Setting ``svg.fonttype`` to ``"none"`` emits UTF-8 text in
    ``<text>`` / ``<tspan>``, which Illustrator can usually select and edit (local font
    substitution applies).

    Note: ``matplotlib.patheffects`` on ``Text`` (e.g. white stroke halos) still force
    outline paths in the SVG even when ``svg.fonttype`` is ``"none"``. For AI-editable
    SVG, avoid text path effects when saving (see alluvial/icicle plots).
    """
    import matplotlib as mpl

    mpl.rcParams["svg.fonttype"] = "none"
