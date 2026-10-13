"""Regional communication maps and shared cell-type heatmaps for mouse brain.

The regional display and ordering follow the manuscript's main figure.
Only the saved leaf graphs, coordinates and annotations are required.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FuncFormatter
from scipy import sparse
from scipy.spatial import cKDTree
from .result_io import lr_matrices
from .figure_export import save_show

TEXT = "#263A46"

FRAME = "#8492A1"

DOMAIN_PALETTE = (
    "#4477AA",
    "#D87352",
    "#228833",
    "#CCBB44",
    "#087F8C",
    "#AA5C73",
    "#7C3AED",
    "#D99528",
    "#66727F",
    "#88AACC",
    "#E6A18A",
    "#77B58A",
    "#DED28C",
    "#73B5BD",
    "#C994A4",
    "#AF8DEA",
    "#E9BD78",
    "#A8B0B8",
    "#533B61",
)

DOMAIN_CMAP = LinearSegmentedColormap.from_list(
    "mouse_brain_coral", ["#F8F4F2", "#F4BE99", "#D97465", "#8D3056"]
)

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
    "font.size": 6,
    "axes.titlesize": 6.5,
    "axes.labelsize": 6,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    "legend.fontsize": 5,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "xtick.major.size": 1.5,
    "ytick.major.size": 1.5,
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "axes.titlecolor": TEXT,
    "axes.edgecolor": FRAME,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.edgecolor": "none",
}

LR_PAIRS = (
    ("Cd200", "Cd200r4", "CD200–CD200R4"),
    ("Col6a3", "Sdc4", "COL6A3–SDC4"),
    ("Penk", "Oprk1", "PENK–OPRK1"),
)

SELECTED_REGIONS = ("5", "4", "0", "1", "3")

def natural_labels(values):
    return sorted(
        set(map(str, values)), key=lambda v: (0, int(v)) if v.isdigit() else (1, v)
    )

def domain_colors(labels):
    return {
        label: (
            DOMAIN_PALETTE[int(label) % len(DOMAIN_PALETTE)]
            if label.isdigit()
            else DOMAIN_PALETTE[i % len(DOMAIN_PALETTE)]
        )
        for i, label in enumerate(natural_labels(labels))
    }

def group_strength(matrix, labels, order):
    codes = pd.Categorical(labels, categories=order).codes
    if np.any(codes < 0):
        raise ValueError("Group order omits observations")
    membership = sparse.csr_matrix(
        (np.ones(len(codes)), (np.arange(len(codes)), codes)),
        shape=(len(codes), len(order)),
    )
    return (membership.T @ matrix @ membership).toarray()

def strength_table(matrix, order, row_name, col_name):
    return pd.DataFrame(
        {
            row_name: np.repeat(order, len(order)),
            col_name: np.tile(order, len(order)),
            "strength": matrix.ravel(),
            "log1p_strength": np.log1p(matrix).ravel(),
        }
    )

def display_edges(matrix, max_outgoing=3, max_incoming=3):
    """Limit directed edges for display only; summaries use all calibrated edges."""
    coo = matrix.tocoo()
    outgoing = np.zeros(matrix.shape[0], dtype=int)
    incoming = np.zeros(matrix.shape[0], dtype=int)
    kept = []
    for k in np.argsort(-coo.data, kind="stable"):
        src, dst = coo.row[k], coo.col[k]
        if src == dst or coo.data[k] <= 0:
            continue
        if outgoing[src] < max_outgoing and incoming[dst] < max_incoming:
            kept.append(k)
            outgoing[src] += 1
            incoming[dst] += 1
    kept = np.asarray(kept, dtype=int)
    return coo.row[kept], coo.col[kept], coo.data[kept]

def spatial_grid(axis, cells, *, values=None, colors=None, cmap=None, norm=None):
    coords = cells[["x", "y"]].to_numpy(float)
    nearest = cKDTree(coords).query(coords, k=2)[0][:, 1]
    side = float(np.median(nearest[nearest > 0])) * 1.02
    offsets = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * side / 2
    collection = PolyCollection(
        coords[:, None, :] + offsets[None, :, :],
        edgecolors="none",
        linewidths=0,
        antialiased=False,
        cmap=cmap,
        norm=norm,
    )
    if values is not None:
        collection.set_array(np.asarray(values))
    else:
        collection.set_facecolors(colors)
    axis.add_collection(collection)
    axis.set(
        xlim=(coords[:, 0].min() - side / 2, coords[:, 0].max() + side / 2),
        ylim=(coords[:, 1].min() - side / 2, coords[:, 1].max() + side / 2),
        aspect="equal",
        xticks=[],
        yticks=[],
    )
    for spine in axis.spines.values():
        spine.set_visible(False)
    return collection

def heatmap(axis, matrix, *, cmap=DOMAIN_CMAP, vmin=0, vmax=None):
    mesh = axis.pcolormesh(
        matrix,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="none",
        linewidth=0,
        rasterized=False,
    )
    axis.set_xlim(0, matrix.shape[1])
    axis.set_ylim(matrix.shape[0], 0)
    axis.set_aspect("auto")
    for spine in axis.spines.values():
        spine.set_linewidth(0.35)
    return mesh

def axes_mm(fig, x, y, width, height):
    fw, fh = fig.get_size_inches() * 25.4
    return fig.add_axes([x / fw, 1 - (y + height) / fh, width / fw, height / fh])

def text_mm(fig, x, y, label, **kwargs):
    fw, fh = fig.get_size_inches() * 25.4
    return fig.text(x / fw, 1 - y / fh, label, ha="left", va="top", **kwargs)


def rasterize_spatial_layer(axis):
    """Rasterize spatial geometry at export DPI, not text/frames/annotations.

    Use only on a spatial axis after its data artists have been added. Do not
    rasterize the whole axes: ROI boxes, labels and spines remain editable.
    This does not alter geometry, scores, styles, clipping or PNG rendering.
    """
    for artist in (*axis.collections, *axis.lines):
        artist.set_rasterized(True)

def legend_grid(fig, x, y, labels, colors, *, ncols=5, colwidth=7, rowheight=2.8):
    for i, label in enumerate(labels):
        row, col = divmod(i, ncols)
        ax = axes_mm(fig, x + col * colwidth, y + row * rowheight, 1.3, 1.3)
        ax.set_facecolor(colors[label])
        ax.set(xticks=[], yticks=[])
        for spine in ax.spines.values():
            spine.set_visible(False)
        text_mm(
            fig,
            x + col * colwidth + 1.8,
            y + row * rowheight - 0.1,
            str(label),
            fontsize=5,
        )

def colorbar_mm(
    fig, x, y, w, h, cmap, norm, *, label=None, ticks=None,
    orientation="horizontal",
):
    axis = axes_mm(fig, x, y, w, h)
    bar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=axis,
        orientation=orientation,
        ticks=ticks,
    )
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=5, length=1.2, pad=0.6)
    tick_axis = bar.ax.xaxis if orientation == "horizontal" else bar.ax.yaxis
    tick_axis.set_major_formatter(
        FuncFormatter(
            lambda value, _: (
                f"{value:.1f}".rstrip("0").rstrip(".")
                if abs(value) < 10
                else f"{value:.0f}"
            )
        )
    )
    if label:
        text_mm(fig, x, y - 2.6, label, fontsize=5)
    # Matplotlib otherwise rasterizes long colorbar gradients in vector export.
    bar.solids.set_rasterized(False)
    return bar
def draw_regional_network(axis, data, pair):
    cells, matrix = data["cells"], data["matrices"][pair]
    keep = data["selection"]
    idx = np.flatnonzero(keep)
    src, dst, values = display_edges(matrix[keep][:, keep], 2, 2)
    src, dst = idx[src], idx[dst]
    palette = domain_colors(cells.domain)
    colors = [
        palette[d] if use else "#F3F4F5" for d, use in zip(cells.domain, keep)
    ]
    spatial_grid(axis, cells, colors=colors)
    coords = cells[["x", "y"]].to_numpy(float)
    segments = np.stack([coords[src], coords[dst]], axis=1)
    lines = LineCollection(
        segments, cmap=DOMAIN_CMAP, norm=Normalize(0, 1), linewidths=0.28, alpha=0.9
    )
    lines.set_array(values)
    axis.add_collection(lines)
    delta = coords[dst] - coords[src]
    axis.quiver(
        coords[src, 0],
        coords[src, 1],
        delta[:, 0],
        delta[:, 1],
        values,
        cmap=DOMAIN_CMAP,
        norm=Normalize(0, 1),
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.0018,
        headwidth=3.2,
        headlength=4,
        headaxislength=3,
        linewidth=0,
        alpha=0.8,
    )
    return src, dst, values
def cell_type_heatmap_order(table):
    """Shared row/column seriation; values and self-communication stay unchanged.

    Minimize LR-balanced, score-weighted distance from the diagonal using
    deterministic pair swaps from alphabetical, degree and spectral starts.
    This is visualization ordering, not clustering or statistical evidence.
    """
    labels = sorted(table.sender_cell_type.unique())
    combined = np.zeros((len(labels), len(labels)))
    pair_matrices = {}
    for pair, frame in table.groupby("lr_pair", sort=True):
        matrix = frame.pivot(index="sender_cell_type", columns="receiver_cell_type", values="strength").reindex(index=labels, columns=labels).to_numpy()
        scores = np.nan_to_num(matrix, nan=0.0)
        pair_matrices[pair] = scores
        if scores.sum() > 0:
            combined += scores / scores.sum()
    affinity = combined + combined.T
    np.fill_diagonal(affinity, 0)
    degree = affinity.sum(axis=1)
    laplacian = np.diag(degree) - affinity
    _, vectors = np.linalg.eigh(laplacian)
    n = len(labels)
    distances = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    def cost(order):
        return float(np.sum(combined[np.ix_(order, order)] * distances))
    starts = [np.arange(n), np.argsort(degree, kind="stable")]
    if n > 1:
        starts.append(np.argsort(vectors[:, 1], kind="stable"))
    best = np.arange(n)
    for start in starts:
        order, current = start.copy(), cost(start)
        for _ in range(200):
            improved, minimum = None, current
            for i in range(n):
                for j in range(i + 1, n):
                    candidate = order.copy()
                    candidate[i], candidate[j] = candidate[j], candidate[i]
                    value = cost(candidate)
                    if value < minimum - 1e-12:
                        improved, minimum = candidate, value
            if improved is None:
                break
            order, current = improved, minimum
        if current < cost(best) - 1e-12:
            best = order
    # The objective is reversal invariant; fix a reproducible orientation.
    if tuple(labels[i] for i in best[::-1]) < tuple(labels[i] for i in best):
        best = best[::-1]
    metadata = {
        "method": "shared deterministic minimum weighted diagonal-distance seriation",
        "values_changed": False, "same_row_column_order": True,
        "ordering_metric": "LR-balanced total retained communication score",
        "objective_alphabetical": cost(np.arange(n)), "objective_reordered": cost(best),
        "order": [labels[i] for i in best],
        "per_pair_mean_distance": {
            pair: {
                "alphabetical": float(np.sum(matrix * distances) / matrix.sum()),
                "reordered": float(np.sum(matrix[np.ix_(best, best)] * distances) / matrix.sum()),
            }
            for pair, matrix in pair_matrices.items() if matrix.sum() > 0
        },
    }
    return metadata["order"], metadata

def draw_regional_interactions(fig, rect, data):
    x, y, w, h = rect
    # Both rows share the same column bounds; reserve space for the y labels.
    column_lefts = (x + 13, x + 45)
    column_width = 26
    for i, (l, r, title) in enumerate(LR_PAIRS[1:]):
        left = column_lefts[i]
        axis = axes_mm(fig, left, y + 3, column_width, 26)
        draw_regional_network(axis, data, f"{l}:{r}")
        rasterize_spatial_layer(axis)
        text_mm(fig, left, y, title, fontsize=5.8)
    colors = domain_colors(data["domains"])
    legend_grid(
        fig, column_lefts[0], y + 29.5, list(SELECTED_REGIONS), colors,
        ncols=5, colwidth=8.5,
    )
    # Former panels e and f now share one letter and are stacked vertically.
    table = data["cell_type_strength"]
    if "cell_type_order" not in data:
        data["cell_type_order"], data["cell_type_order_metadata"] = cell_type_heatmap_order(table)
    types = data["cell_type_order"]
    vmax = float(table.log1p_strength.max())
    for i, (l, r, title) in enumerate(LR_PAIRS[1:]):
        pair = f"{l}:{r}"
        frame = table.query("lr_pair == @pair")
        matrix = (
            frame.pivot(
                index="sender_cell_type",
                columns="receiver_cell_type",
                values="log1p_strength",
            )
            .reindex(index=types, columns=types)
            .to_numpy()
        )
        left = column_lefts[i]
        axis = axes_mm(fig, left, y + 32.5, column_width, column_width)
        cmap = DOMAIN_CMAP.copy()
        cmap.set_bad("#DDE1E5")
        heatmap(axis, matrix, cmap=cmap, vmax=vmax)
        # Both matrices use identical row order and share the complete y key.
        # Show every cell type by name, rather than alternating numeric IDs.
        tick_positions = np.arange(len(types))
        axis.set_yticks(tick_positions + 0.5)
        axis.set_yticklabels(types if i == 0 else [])
        axis.set_xticks(tick_positions + 0.5)
        axis.set_xticklabels(types, rotation=90, ha="center")
        axis.tick_params(labelsize=5, length=0, pad=0.7)
        if i == 0:
            axis.set_ylabel("Sender cell type", fontsize=5, labelpad=1)
    text_mm(fig, column_lefts[0], y + 71, "Receiver cell type", fontsize=5)
    text_mm(fig, x + 1, y + 73.3, "log(1+sum)", fontsize=5)
    colorbar_mm(
        fig,
        x + 29,
        y + 74,
        30,
        1.1,
        DOMAIN_CMAP,
        Normalize(0, vmax),
        ticks=[0, vmax],
    )

def regional_communication(result, output):
    """Draw COL6A3–SDC4 and PENK–OPRK1 within domains 5, 4, 0, 1 and 3.

    Heatmaps sum all retained directed weights in the selected regions,
    including self weights. Spatial links alone are thinned for readability.
    The same deterministic cell-type order and color limits serve both pairs.
    """
    xy = np.asarray(result.obsm["spatial"], dtype=float)
    cells = pd.DataFrame({"cell_id": result.obs_names.astype(str), "x": xy[:, 0], "y": xy[:, 1],
                          "domain": result.obs["domain"].astype(str).to_numpy(),
                          "cell_type": result.obs["cell_type"].astype(str).to_numpy()})
    selection = cells.domain.isin(SELECTED_REGIONS).to_numpy()
    if not selection.any():
        raise ValueError("None of the selected spatial domains is present")
    matrices = lr_matrices(result)
    selected_types = sorted(cells.loc[selection, "cell_type"].unique())
    parts = []
    for ligand, receptor, _ in LR_PAIRS[1:]:
        pair = f"{ligand}:{receptor}"
        grouped = group_strength(matrices[pair][selection][:, selection],
                                 cells.loc[selection, "cell_type"], selected_types)
        parts.append(strength_table(grouped, selected_types, "sender_cell_type", "receiver_cell_type")
                     .assign(lr_pair=pair))
    data = dict(cells=cells, matrices=matrices, selection=selection,
                domains=natural_labels(cells.domain), cell_type_strength=pd.concat(parts, ignore_index=True))
    with mpl.rc_context(RC):
        fig = plt.figure(figsize=(76/25.4, 86/25.4))
        draw_regional_interactions(fig, (2, 2, 72, 77), data)
        save_show(fig, output, "regional_communication")
    data["cell_type_strength"].to_csv(Path(output)/"regional_celltype_strength.csv", index=False)
    return data["cell_type_order"]
