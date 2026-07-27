"""Viz: matplotlib SVG export helpers and notebook display utilities."""

from .matplotlib_svg import configure_matplotlib_svg_for_illustrator, path_wants_svg
from .notebook_display import display_svg_grid

__all__ = [
    "configure_matplotlib_svg_for_illustrator",
    "display_svg_grid",
    "path_wants_svg",
]
