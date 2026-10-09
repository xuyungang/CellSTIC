"""Export editable figures and embed compact notebook previews."""
from __future__ import annotations

import base64
from html import escape
from io import BytesIO

from pathlib import Path

import matplotlib as mpl

import matplotlib.pyplot as plt


from PIL import Image


def show_image_row(paths, labels=None):
    """Embed existing figures in one row, with no links to local image files.

    SVG artwork retains its original drawing style. Each image has its own
    document, so repeated SVG element identifiers cannot affect another panel.
    """
    from IPython.display import HTML, display

    paths = [Path(path) for path in paths]
    labels = [""] * len(paths) if labels is None else list(labels)
    if not paths or len(labels) != len(paths):
        raise ValueError("Supply at least one image and one label per image")
    panels = []
    for path, label in zip(paths, labels):
        if path.suffix.lower() == ".svg":
            mime, content = "image/svg+xml", path.read_bytes()
        else:
            with Image.open(path) as source:
                preview = source.convert("RGB")
                preview.thumbnail((1800, 1800))
                buffer = BytesIO()
                preview.save(buffer, format="PNG")
            mime, content = "image/png", buffer.getvalue()
        encoded = base64.b64encode(content).decode("ascii")
        caption = f'<figcaption style="text-align:center">{escape(str(label))}</figcaption>' if label else ""
        panels.append(
            '<figure style="flex:1 1 0;min-width:0;margin:0">'
            f'<img src="data:{mime};base64,{encoded}" alt="{escape(str(label), quote=True)}" '
            'style="display:block;width:100%;height:auto">'
            f'{caption}</figure>'
        )
    display(HTML('<div style="display:flex;flex-wrap:nowrap;align-items:flex-start;gap:12px;width:100%">'
                 + "".join(panels) + "</div>"))


def show_png(path):
    """Embed a readable preview; full-resolution PNG/PDF/SVG exports remain on disk."""
    from IPython.display import Image as NotebookImage, display
    with Image.open(path) as source:
        preview = source.convert('RGB')
        preview.thumbnail((1800, 1800))
        buffer = BytesIO()
        preview.save(buffer, format='PNG')
    display(NotebookImage(data=buffer.getvalue()))

def save_show(fig, output, name, *, exact_size=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context({'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42}):
        for suffix in ('png', 'pdf', 'svg'):
            fig.savefig(output / f'{name}.{suffix}', dpi=600,
                        bbox_inches=None if exact_size else 'tight',
                        pad_inches=0 if exact_size else .03, facecolor='white')
    plt.close(fig)
    show_png(output / f'{name}.png')
