"""Square L2/L3/LR hierarchy strips for four sender–receiver cell-type pairs.

Rendering and leaf aggregation are extracted from human_lymph_node/plot_panel_e.py.
The utility prepares segment data from a packed CellSTIC out.h5ad or redraws an
exported table. It has no runtime dependency on the paper scripts.
"""
from __future__ import annotations

from pathlib import Path
from itertools import combinations
import json
import re
from typing import Mapping, Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import jensenshannon


PathLike = Union[str, Path]
SIDES = ("top", "right", "bottom", "left")
TEXT_COLOR, FRAME_COLOR, GRID_COLOR, SLATE = "#263A46", "#8492A1", "#DCE2E8", "#66727F"
BRANCH_COLORS = ("#4477AA", "#087F8C", "#D87352", "#D99528", "#AA5C73",
                 "#228833", "#7C3AED", "#CCBB44", "#66727F")
_NAMED_BRANCH_COLORS = {
    "Adhesion and Cell-Cell Communication Mediators": "#4477AA",
    "Chemokine Signaling Axis": "#087F8C",
    "Immunomodulatory Cytokine and Co-stimulatory Pathways": "#D87352",
}
INNER_HALF, CORNER_GAP = .42, .075
RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
    "font.size": 5.2, "axes.linewidth": .55,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "text.color": TEXT_COLOR, "axes.labelcolor": TEXT_COLOR,
    "axes.titlecolor": TEXT_COLOR, "axes.edgecolor": FRAME_COLOR,
    "xtick.color": TEXT_COLOR, "ytick.color": TEXT_COLOR, "grid.color": GRID_COLOR,
    "axes.facecolor": "#FFFFFF", "figure.facecolor": "#FFFFFF",
    "savefig.facecolor": "#FFFFFF", "savefig.bbox": None,
}
_REQUIRED = {
    "side", "source", "target", "leaf", "leaf_order", "l2_code", "l3_code",
    "fraction", "segment_start", "segment_end",
}


def _metadata(value):
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(value) if isinstance(value, str) else value


def _hierarchy_paths(hierarchy, leaves):
    keys = sorted((key for key in hierarchy if re.fullmatch(r"level_\d+", str(key))),
                  key=lambda key: int(key.split("_")[1]))
    if keys[:3] != ["level_1", "level_2", "level_3"]:
        raise ValueError("The square plot requires hierarchy levels 1, 2 and 3 above the LR leaves")
    groups = {}
    for key in keys:
        mapping = {}
        for name, members in hierarchy[key].items():
            members = [members] if isinstance(members, Mapping) else members
            for member in members:
                leaf = str(member.get("edge_type_name", ""))
                if leaf in leaves:
                    if leaf in mapping and mapping[leaf] != str(name):
                        raise ValueError(f"Leaf {leaf!r} has multiple parents at {key}")
                    mapping[leaf] = str(name)
        if key in keys[:3] and set(leaves)-set(mapping):
            raise ValueError(f"{key} does not map every exported LR leaf")
        groups[key] = {leaf: mapping.get(leaf, leaf) for leaf in leaves}
    paths = {leaf: tuple(groups[key][leaf] for key in keys) for leaf in leaves}
    ordered = sorted(leaves, key=lambda leaf: paths[leaf]+(-1.0, leaf))
    codes, colors = {}, {}
    for depth in (2, 3):
        nodes = list(dict.fromkeys(paths[leaf][:depth] for leaf in ordered))
        for index, node in enumerate(nodes, 1):
            codes[node] = f"L{depth}-{index}"
            if depth == 2:
                colors[node] = _NAMED_BRANCH_COLORS.get(node[-1], BRANCH_COLORS[(index-1) % len(BRANCH_COLORS)])
    return paths, codes, colors


def _resolve_type(name, available):
    def key(value):
        return re.sub(r"[ /_-]+", " ", str(value).strip()).casefold()
    if name in available:
        return name
    matches = [candidate for candidate in available if key(candidate) == key(name)]
    if len(matches) != 1:
        raise ValueError(f"Cell type {name!r} not uniquely found; available: {available}")
    return matches[0]


def _choose_pairs(names, strengths, cell_counts, pairs, sender, min_receiver_cells):
    if pairs is not None:
        if sender is not None:
            raise ValueError("Choose pairs or sender, not both")
        if len(pairs) != 4 or any(isinstance(pair, str) or len(pair) != 2 for pair in pairs):
            raise ValueError("pairs must contain four (sender, receiver) pairs in top/right/bottom/left order")
        chosen = [tuple(_resolve_type(name, names) for name in pair) for pair in pairs]
        if len(set(chosen)) != 4:
            raise ValueError("The four requested cell-type pairs must be distinct")
        return chosen
    eligible = [i for i, name in enumerate(names)
                if name.casefold() != "others" and cell_counts[i] >= min_receiver_cells]
    totals = strengths.sum(axis=2)
    if sender is None:
        candidates = [i for i in range(len(names)) if np.count_nonzero(totals[i, eligible] > 0) >= 4]
        if not candidates:
            raise ValueError("No sender has four nonempty eligible receivers; lower min_receiver_cells or supply pairs")
        sender_index = max(candidates, key=lambda i: float(totals[i, eligible].sum()))
    else:
        sender_index = names.index(_resolve_type(sender, names))
    receivers = [i for i in eligible if totals[sender_index, i] > 0]
    if len(receivers) < 4:
        raise ValueError(f"{names[sender_index]!r} has fewer than four eligible receivers; lower min_receiver_cells or supply pairs")
    profiles = {i: strengths[sender_index, i]/totals[sender_index, i] for i in receivers}
    distances = {(i, j): float(jensenshannon(profiles[i], profiles[j], base=2))
                 for i, j in combinations(receivers, 2)}
    # Match panel E: maximize minimum pairwise distance, then mean distance.
    # Precompute each distance once; sorted names give deterministic tie breaks.
    def score(indices):
        values = [distances[pair] for pair in combinations(indices, 2)]
        return min(values), float(np.mean(values))
    selected = max(combinations(receivers, 4), key=score)
    return [(names[sender_index], names[i]) for i in selected]


def _read_source(value):
    frame = (value.copy(deep=True) if isinstance(value, pd.DataFrame)
             else pd.read_csv(value, float_precision="round_trip"))
    missing = _REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Source data are missing columns: {', '.join(sorted(missing))}")
    if set(frame.side) != set(SIDES):
        raise ValueError("Source data must contain the top, right, bottom and left sides")
    for column in ("source", "target", "leaf", "l2_code", "l3_code"):
        if not frame[column].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError(f"{column} must contain nonempty strings")
    for column in ("leaf_order", "fraction", "segment_start", "segment_end"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must contain finite numbers")
    if not ((frame.leaf_order > 0) & (frame.leaf_order == np.floor(frame.leaf_order))).all():
        raise ValueError("leaf_order must contain positive integers")
    for column, prefix in (("l2_code", "L2"), ("l3_code", "L3")):
        if not frame[column].str.fullmatch(rf"{prefix}-[1-9]\d*").all():
            raise ValueError(f"{column} must use {prefix}-1, {prefix}-2, ... codes")
    if (frame.fraction <= 0).any() or (frame.segment_end <= frame.segment_start).any():
        raise ValueError("Every plotted leaf must have a positive segment width")
    if not np.allclose(frame.fraction, frame.segment_end-frame.segment_start, atol=1e-10, rtol=1e-8):
        raise ValueError("fraction must equal segment_end minus segment_start")
    if (frame.groupby("l3_code").l2_code.nunique() > 1).any():
        raise ValueError("An L3 code must have the same L2 parent on all sides")
    if (frame.groupby("leaf")[["l2_code", "l3_code"]].nunique() > 1).any().any():
        raise ValueError("Each LR leaf must have the same hierarchy codes on all sides")
    for side in SIDES:
        part = frame[frame.side == side].sort_values("leaf_order", kind="stable")
        if len(part[["source", "target"]].drop_duplicates()) != 1:
            raise ValueError(f"{side}: expected one sender–receiver pair")
        if part.leaf.duplicated().any() or part.leaf_order.duplicated().any():
            raise ValueError(f"{side}: LR leaves and leaf_order must be unique")
        if not (np.isclose(part.segment_start.iloc[0], 0, atol=1e-10, rtol=0)
                and np.isclose(part.segment_end.iloc[-1], 1, atol=1e-10, rtol=0)
                and np.allclose(part.segment_start.to_numpy()[1:],
                                part.segment_end.to_numpy()[:-1], atol=1e-10, rtol=0)):
            raise ValueError(f"{side}: ordered segments must cover [0, 1] without gaps or overlaps")
        for column in ("l2_code", "l3_code"):
            runs = part.loc[part[column].ne(part[column].shift()), column]
            if runs.duplicated().any():
                raise ValueError(f"{side}: {column} groups must be contiguous")
    return frame


def _mix_with_white(color, amount):
    rgba = np.asarray(mcolors.to_rgba(color), dtype=float)
    rgba[:3] = rgba[:3]*(1-amount)+amount
    return tuple(rgba)


def _text_color(fill):
    rgb = np.asarray(mcolors.to_rgba(fill), dtype=float)[:3]
    linear = np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)
    return "#FFFFFF" if float(np.dot(linear, [.2126, .7152, .0722])) < .38 else TEXT_COLOR


def _point(side, t, offset):
    outer_t = CORNER_GAP+float(t)*(1-2*CORNER_GAP)
    h = INNER_HALF
    outer = {"top": (-1+2*outer_t, 1), "right": (1, 1-2*outer_t),
             "bottom": (1-2*outer_t, -1), "left": (-1, -1+2*outer_t)}[side]
    inner = {"top": (-h+2*h*t, h), "right": (h, h-2*h*t),
             "bottom": (h-2*h*t, -h), "left": (-h, -h+2*h*t)}[side]
    return tuple((1-offset)*a+offset*b for a, b in zip(outer, inner))


def _segment(ax, side, start, end, offsets, color, linewidth):
    low, high = offsets
    ax.add_patch(Polygon([
        _point(side, start, low), _point(side, end, low),
        _point(side, end, high), _point(side, start, high),
    ], closed=True, facecolor=color, edgecolor="#FFFFFF", linewidth=linewidth,
        antialiased=True, joinstyle="miter", zorder=2))


def _groups(rows, column):
    groups = []
    for row in rows:
        code = getattr(row, column)
        if groups and groups[-1]["code"] == code:
            groups[-1]["end"] = row.segment_end
        else:
            groups.append(dict(code=code, parent=row.l2_code,
                               start=row.segment_start, end=row.segment_end))
    return groups


def _spread(desired, min_gap, low=.04, high=.96):
    placed = [max(low, min(high, desired[0]))]
    for value in desired[1:]:
        placed.append(max(value, placed[-1]+min_gap))
    overflow = placed[-1]-high
    if overflow > 0:
        placed = [value-overflow for value in placed]
    if placed[0] < low:
        placed = [value+low-placed[0] for value in placed]
    if placed[-1] > high:
        placed[-1] = high
        for index in range(len(placed)-2, -1, -1):
            placed[index] = min(placed[index], placed[index+1]-min_gap)
    return [max(low, min(high, value)) for value in placed]


def _draw_side(ax, side, part, colors, pair_label, lr_label_size=3.0, min_group_label_fraction=0.0):
    rows = list(part.sort_values("leaf_order", kind="stable").itertuples(index=False))
    lr_track, l3_track, l2_track = (.02, .66), (.68, .82), (.84, .97)
    tangent = -90 if side == "right" else 90 if side == "left" else 0
    radial = 90 if side in {"top", "bottom"} else 0
    for segment in _groups(rows, "l2_code"):
        start, end = segment["start"], segment["end"]
        fill = _mix_with_white(colors[segment["parent"]], .12)
        _segment(ax, side, start, end, l2_track, fill, .28)
        if end-start >= max(.060, min_group_label_fraction):
            ax.text(*_point(side, .5*(start+end), np.mean(l2_track)), segment["code"],
                    ha="center", va="center", rotation=tangent, rotation_mode="anchor",
                    fontsize=5.0, fontweight="semibold", color=_text_color(fill), zorder=4)
    groups = _groups(rows, "l3_code")
    for segment in groups:
        fill = _mix_with_white(colors[segment["parent"]], .36)
        _segment(ax, side, segment["start"], segment["end"], l3_track, fill, .24)
    last = -10.0
    for segment in groups:
        position = .5*(segment["start"]+segment["end"])
        if segment["end"]-segment["start"] < max(.072, min_group_label_fraction) or position-last < .095:
            continue
        ax.text(*_point(side, position, .75), segment["code"], ha="center", va="center",
                rotation=tangent, rotation_mode="anchor", fontsize=5.0, color=TEXT_COLOR, zorder=4)
        last = position
    for row in rows:
        fill = _mix_with_white(colors[row.l2_code], .62)
        _segment(ax, side, row.segment_start, row.segment_end, lr_track, fill, .18)
    desired = [.5*(row.segment_start+row.segment_end) for row in rows]
    positions = _spread(desired, min(.036, .94/max(len(desired)-1, 1)))
    horizontal = "right" if side in {"top", "right"} else "left"
    for row, label_t, source_t in zip(rows, positions, desired):
        if abs(label_t-source_t) > .006:
            ax.plot(*zip(_point(side, source_t, .025), _point(side, label_t, .045)),
                    color=FRAME_COLOR, linewidth=.22, alpha=.65, zorder=3)
        ax.text(*_point(side, label_t, .045), row.leaf.replace(":", "–"),
                ha=horizontal, va="center", rotation=radial, rotation_mode="anchor",
                fontsize=lr_label_size, fontweight="normal", fontstretch="condensed",
                color=TEXT_COLOR, zorder=4, clip_on=True)
    x, y = {"top": (0, 1.125), "right": (1.125, 0),
            "bottom": (0, -1.125), "left": (-1.125, 0)}[side]
    horizontal, vertical = {"top": ("center", "bottom"), "right": ("left", "center"),
                            "bottom": ("center", "top"), "left": ("right", "center")}[side]
    if pair_label is None:
        pair_label = f"{rows[0].source}_to_{rows[0].target}"
        pair_label = pair_label if len(pair_label) <= 72 else pair_label[:71]+"."
    ax.text(x, y, pair_label, ha=horizontal, va=vertical, rotation=tangent,
            rotation_mode="anchor", fontsize=5.0, fontweight="semibold", color=TEXT_COLOR, zorder=5)


def _drawing_source(source_data, branch_colors, pair_labels):
    source = _read_source(source_data)
    codes = set(source.l2_code)
    colors = {code: BRANCH_COLORS[(int(code.split("-")[1])-1) % len(BRANCH_COLORS)] for code in codes}
    if "l2_color" in source:
        for code, group in source.groupby("l2_code"):
            supplied = group.l2_color.dropna().unique()
            if len(supplied) != 1 or not mcolors.is_color_like(supplied[0]):
                raise ValueError(f"{code} must have one valid l2_color")
            colors[code] = supplied[0]
    if branch_colors is not None:
        if set(branch_colors)-codes or not all(mcolors.is_color_like(c) for c in branch_colors.values()):
            raise ValueError("branch_colors must map present L2 codes to valid colors")
        colors.update(branch_colors)
    source["l2_color"] = source.l2_code.map(colors)
    labels = dict(pair_labels or {})
    if set(labels)-set(SIDES) or not all(isinstance(label, str) for label in labels.values()):
        raise ValueError("pair_labels must map square sides to strings")
    return source, colors, labels


class HierarchySquareVisualizer:
    """Four cell-type pairs arranged around a square with L2, L3 and LR bands."""

    @staticmethod
    def prepare_source(
        h5ad,
        *,
        cell_type_key="cell_type",
        pairs=None,
        sender=None,
        threshold=0.0,
        min_count_threshold=1,
        min_receiver_cells=200,
    ) -> pd.DataFrame:
        """Build plotting data from a CellSTIC out.h5ad path or packed AnnData.

        Read ``uns['cellstic']`` hierarchy/channel metadata and the final leaf
        matrices in ``obsp``. Remove self edges, apply ``threshold``, and sum
        LR scores within each directed cell-type pair. Set a pair/leaf strength
        to zero if its nonzero edge count is below ``min_count_threshold``.
        Sparse cell-by-cell matrices are never densified or changed in place.

        ``pairs`` fixes the four sides in top/right/bottom/left order. Otherwise
        select four receivers with >= ``min_receiver_cells`` cells (excluding
        Others) by maximum minimum Jensen–Shannon distance between LR profiles,
        breaking ties by mean distance. ``sender`` fixes the sender; when absent,
        use the sender with greatest total strength to eligible receivers and
        at least four nonempty receiver profiles. Names resolve case/spacing
        variants. Explicit pairs bypass the automatic receiver-size filter.

        Codes use all exported leaves, so their meanings stay constant across
        chosen pairs. The returned table includes strengths, nonzero edge counts,
        original L2/L3 names and colors alongside the drawing coordinates.
        """
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold must be finite and nonnegative")
        for name, value in (("min_count_threshold", min_count_threshold),
                            ("min_receiver_cells", min_receiver_cells)):
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(h5ad, (str, Path)):
            from anndata import read_h5ad
            adata = read_h5ad(h5ad)
        else:
            adata = h5ad
        if not hasattr(adata, "obs") or not hasattr(adata, "obsp"):
            raise TypeError("Supply an out.h5ad path or a packed AnnData object")
        if cell_type_key not in adata.obs:
            raise KeyError(f"adata.obs[{cell_type_key!r}] is required")
        raw_labels = adata.obs[cell_type_key]
        if raw_labels.isna().any():
            raise ValueError("Cell-type annotations must not contain missing values")
        labels = raw_labels.astype(str).to_numpy()
        names, inverse, cell_counts = np.unique(labels, return_inverse=True, return_counts=True)
        names = names.tolist()
        if any(not name.strip() for name in names):
            raise ValueError("Cell-type annotations must not be empty")
        meta = adata.uns.get("cellstic", {})
        if not {"hierarchy_dict_json", "tree_levels_json"}.issubset(meta):
            raise ValueError("Expected CellSTIC out.h5ad with hierarchy_dict_json and tree_levels_json metadata")
        hierarchy = _metadata(meta["hierarchy_dict_json"])
        levels = _metadata(meta["tree_levels_json"])
        if not levels:
            raise ValueError("No exported hierarchy levels found")
        level = max(levels, key=lambda item: int(item["level_num"]))
        number = int(level["level_num"])
        if number < 4:
            raise ValueError("The square plot needs L2/L3 levels above a final LR leaf level")
        leaves = [str(name) for name, _ in sorted(level["edge_type_map"].items(), key=lambda item: int(item[1]))]
        if not leaves or any(name.lower() in {"root", "total"} for name in leaves):
            raise ValueError("The final hierarchy level must contain LR leaves")
        paths, codes, colors = _hierarchy_paths(hierarchy, leaves)
        onehot = sparse.csr_matrix((np.ones(len(labels)), (np.arange(len(labels)), inverse)),
                                  shape=(len(labels), len(names)))
        strengths = np.zeros((len(names), len(names), len(leaves)), dtype=float)
        counts = np.zeros_like(strengths, dtype=np.int64)
        for index, leaf in enumerate(leaves):
            safe = leaf.replace(":", "_").replace("/", "_").replace(" ", "_").strip() or "channel"
            key = f"cellstic_L{number}_lr_{safe}"
            if key not in adata.obsp:
                raise KeyError(f"Missing adata.obsp[{key!r}]")
            matrix = sparse.csr_matrix(adata.obsp[key], copy=True)
            matrix.sum_duplicates()
            if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
                raise ValueError(f"{key}: scores must be finite and nonnegative")
            diagonal = matrix.diagonal()
            if np.any(diagonal):
                matrix = matrix-sparse.diags(diagonal, format="csr")
            matrix.data[matrix.data < threshold] = 0
            matrix.eliminate_zeros()
            edge_counts = (onehot.T @ matrix.astype(bool).astype(np.float32) @ onehot).toarray()
            values = (onehot.T @ matrix @ onehot).toarray()
            counts[:, :, index] = edge_counts.astype(np.int64)
            strengths[:, :, index] = np.where(edge_counts >= min_count_threshold, values, 0.0)
        chosen = _choose_pairs(names, strengths, cell_counts, pairs, sender, min_receiver_cells)
        rows = []
        for side, (source, target) in zip(SIDES, chosen):
            si, ti = names.index(source), names.index(target)
            values = {leaf: float(value) for leaf, value in zip(leaves, strengths[si, ti])}
            active = sorted((leaf for leaf in leaves if values[leaf] > 0),
                            key=lambda leaf: paths[leaf]+(-float(values[leaf]), leaf))
            if not active:
                raise ValueError(f"No positive LR strengths for {source!r} → {target!r} after filtering")
            total, cursor = float(sum(value for value in values.values() if value > 0)), 0.0
            for order, leaf in enumerate(active, 1):
                end = min(1.0, cursor+values[leaf]/total) if order < len(active) else 1.0
                path = paths[leaf]
                rows.append(dict(side=side, source=source, target=target, leaf=leaf, leaf_order=order,
                                 l2_code=codes[path[:2]], l3_code=codes[path[:3]],
                                 fraction=end-cursor, segment_start=cursor, segment_end=end,
                                 strength=values[leaf], n_edges=int(counts[si, ti, leaves.index(leaf)]),
                                 pair_total_strength=total, l2_name=path[1], l3_name=path[2],
                                 l2_color=colors[path[:2]], source_cell_count=int(cell_counts[si]),
                                 target_cell_count=int(cell_counts[ti])))
                cursor = end
        return _read_source(pd.DataFrame(rows))

    def draw(
        self,
        *,
        ax: plt.Axes,
        source_data: Union[PathLike, pd.DataFrame],
        branch_colors: Optional[Mapping[str, str]] = None,
        pair_labels: Optional[Mapping[str, str]] = None,
        lr_label_size: float = 3.0,
        min_group_label_fraction: float = 0.0,
    ) -> plt.Axes:
        """Draw into an existing axes without saving or closing its figure.

        Accept the same source table, colors and labels as ``plot``. Increase
        ``lr_label_size`` for schematic examples with fewer LR leaves. Geometry
        and default text sizes match the standalone 84-mm square plot. Set
        ``min_group_label_fraction`` to hide labels on narrow L2/L3 segments;
        all underlying segments remain visible with their original widths.
        """
        if not np.isfinite(lr_label_size) or lr_label_size <= 0:
            raise ValueError("lr_label_size must be positive and finite")
        if not np.isfinite(min_group_label_fraction) or not 0 <= min_group_label_fraction <= 1:
            raise ValueError("min_group_label_fraction must be between zero and one")
        source, colors, labels = _drawing_source(source_data, branch_colors, pair_labels)
        with plt.style.context("default"), mpl.rc_context(RC):
            ax.set_aspect("equal")
            ax.set_facecolor("white")
            ax.axis("off")
            for side in SIDES:
                _draw_side(ax, side, source[source.side == side], colors, labels.get(side),
                           lr_label_size, min_group_label_fraction)
            ax.plot([-1, 1, 1, -1, -1], [1, 1, -1, -1, 1], color=FRAME_COLOR,
                    linewidth=.58, solid_capstyle="butt", solid_joinstyle="miter", zorder=5)
            h = INNER_HALF
            ax.add_patch(Polygon([(-h, -h), (h, -h), (h, h), (-h, h)], closed=True,
                                 facecolor="#FFFFFF", edgecolor=GRID_COLOR, linewidth=.42, zorder=1))
            ax.text(0, .035, "Higher levels", ha="center", va="center", fontsize=6.0,
                    fontweight="semibold", color=TEXT_COLOR, zorder=6)
            ax.text(0, -.060, "L2  →  L3  →  LR", ha="center", va="center",
                    fontsize=6.0, color=SLATE, zorder=6)
            ax.set_xlim(-1.18, 1.18)
            ax.set_ylim(-1.18, 1.18)
        return ax

    def plot(
        self,
        *,
        source_data: Union[PathLike, pd.DataFrame],
        save_path: PathLike = "cell_type_hierarchy_square",
        size_mm: float = 84.0,
        dpi: int = 600,
        branch_colors: Optional[Mapping[str, str]] = None,
        pair_labels: Optional[Mapping[str, str]] = None,
    ) -> Path:
        """Export SVG/PDF/PNG and one direct-data CSV; return the SVG path.

        ``source_data`` is the panel E source CSV or an equivalent DataFrame.
        Each row describes a positive LR segment and has these columns:

        * ``side``: top, right, bottom or left; one cell-type pair per side.
        * ``source``, ``target``, ``leaf``: sender, receiver and LR names.
        * ``leaf_order``: positive integer ordering leaves along each side.
        * ``l2_code``, ``l3_code``: stable L2-n and L3-n hierarchy codes.
        * ``fraction``, ``segment_start``, ``segment_end``: within-pair widths
          and contiguous bounds covering [0, 1], in ``leaf_order`` order.

        Use ``prepare_source(out_h5ad_path, sender=...)`` to generate this table
        directly from pipeline output. Segment widths are used as supplied.
        L2 and L3 bands merge adjacent
        leaves sharing a code. Fractions describe composition within each
        pair, not absolute strength differences between pairs. Extra columns
        are preserved in the output CSV; input tables are not modified.

        ``branch_colors`` optionally maps L2 codes to colors. Defaults match
        panel E (L2-1 blue, L2-2 teal, L2-3 coral). ``pair_labels`` optionally
        overrides outer labels by side. ``save_path`` accepts a directory or
        a .svg/.pdf/.png filename selecting the shared artifact stem. Default
        size is 84 × 84 mm, with editable vector text and a 600-dpi PNG. No
        display, backend change or global plotting-style change is needed.
        """
        if not np.isfinite(size_mm) or size_mm <= 0:
            raise ValueError("size_mm must be positive and finite")
        if not isinstance(dpi, (int, np.integer)) or dpi < 300:
            raise ValueError("dpi must be an integer of at least 300")
        source, colors, labels = _drawing_source(source_data, branch_colors, pair_labels)
        target = Path(save_path)
        if target.suffix.lower() in {".svg", ".pdf", ".png"}:
            directory, stem = target.parent, target.stem
        else:
            directory, stem = target, f"cell_type_hierarchy_square_{size_mm:g}mm"
        directory.mkdir(parents=True, exist_ok=True)
        with plt.style.context("default"), mpl.rc_context(RC):
            fig, ax = plt.subplots(figsize=(size_mm/25.4, size_mm/25.4), dpi=dpi, facecolor="white")
            try:
                self.draw(ax=ax, source_data=source, branch_colors=colors, pair_labels=labels)
                fig.subplots_adjust(left=.025, right=.975, top=.975, bottom=.025)
                for suffix in ("svg", "pdf", "png"):
                    fig.savefig(directory/f"{stem}.{suffix}", dpi=dpi,
                                facecolor="white", edgecolor="none")
            finally:
                plt.close(fig)
        source.to_csv(directory/f"{stem}_source_data.csv", index=False, float_format="%.17g")
        return directory/f"{stem}.svg"


__all__ = ["HierarchySquareVisualizer"]
