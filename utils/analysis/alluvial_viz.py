from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.patches import FancyBboxPatch, Polygon

from utils.viz.matplotlib_svg import configure_matplotlib_svg_for_illustrator, path_wants_svg

from .aggregated_heatmap_viz import AggregatedHeatmapVisualizer
from .cell_type_communication import CellTypeCommunicationComputer


_NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
    "font.size": 7.5,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#2B2B2B",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "none",
}


class AlluvialVisualizer:
    """Alluvial plots over LR hierarchy tree (Nature-style, matplotlib only)."""

    def __init__(self) -> None:
        self._agg = AggregatedHeatmapVisualizer()
        self._ccc = CellTypeCommunicationComputer()


    def _plot_alluvial_core(
        self,
        leaf_strengths: Mapping[str, float],
        leaf_names: List[str],
        level_keys: List[str],
        leaf_to_group: Dict[str, Dict[str, str]],
        title: str,
        save_path: Path,
        min_width_fraction: float,
        figsize: Tuple[float, float],
        dpi: int,
        high_contrast: bool = False,
        stable_order_level_keys: Optional[Iterable[str]] = ("level_2", "level_3"),
        show_legend: bool = True,
    ) -> None:
        """Draw one alluvial diagram for given leaf-level strengths.

        Notes
        -----
        - The vertical order of groups within each level is always derived from the
          leaf order (via `first_pos`) so that parent–child branches do not cross
          between levels.
        - `stable_order_level_keys` controls *color assignment* only (via the main
          level sorting), not the geometric ordering of groups. This ensures that
          the same node keeps the same base colour across different plots, while
          preserving the non‑crossing tree structure.
        """
        plt.rcParams.update(_NATURE_RC)

        # path_effects on text force outline paths in SVG; Illustrator cannot select as text.
        svg_plain_text = path_wants_svg(save_path)

        _stable_set: Set[str] = set(stable_order_level_keys) if stable_order_level_keys else set()

        leaf_names = [ln for ln in leaf_names if ln in leaf_strengths]
        if not leaf_names:
            return

        total_strength = float(sum(float(leaf_strengths[ln]) for ln in leaf_names))
        if total_strength <= 0:
            total_strength = 1.0

        leaf_order = sorted(
            leaf_names,
            key=lambda ln: _path_key_for_leaf(ln, level_keys, leaf_to_group, leaf_strengths),
        )
        leaf_pos = {leaf: i for i, leaf in enumerate(leaf_order)}

        group_strengths_per_level: List[Dict[str, float]] = []
        groups_per_level: List[List[str]] = []
        group_codes_per_level: List[Dict[str, str]] = []

        for lvl_idx, level_key in enumerate(level_keys):
            gs: Dict[str, float] = defaultdict(float)
            first_pos: Dict[str, int] = {}

            mapping = leaf_to_group[level_key]
            for leaf in leaf_order:
                g = mapping.get(leaf, leaf)
                gs[g] += float(leaf_strengths[leaf])
                if g not in first_pos:
                    first_pos[g] = leaf_pos[leaf]

            # Always use path-based order so that branches remain vertically
            # aligned between levels and do not cross, independent of colour
            # assignment or level-specific settings.
            groups = sorted(gs.keys(), key=lambda g: (first_pos[g], -gs[g], str(g)))
            group_strengths_per_level.append(gs)
            groups_per_level.append(groups)

            codes: Dict[str, str] = {}
            if 0 < lvl_idx < len(level_keys) - 1:
                for gi, g in enumerate(groups):
                    codes[g] = f"L{lvl_idx + 1}-{gi + 1}"
            group_codes_per_level.append(codes)

        flows_per_level: List[Dict[Tuple[str, str], float]] = []
        for l in range(len(level_keys) - 1):
            left_map = leaf_to_group[level_keys[l]]
            right_map = leaf_to_group[level_keys[l + 1]]
            fd: Dict[Tuple[str, str], float] = defaultdict(float)

            for leaf in leaf_order:
                g_left = left_map.get(leaf, leaf)
                g_right = right_map.get(leaf, leaf)
                fd[(g_left, g_right)] += float(leaf_strengths[leaf])

            flows_per_level.append(fd)

        n_levels = len(level_keys)
        x_gap = 1.28
        x_positions = np.arange(n_levels, dtype=float) * x_gap
        bar_half_width = 0.055

        span_ymin = 0.07
        span_ymax = 0.89
        usable_span = span_ymax - span_ymin

        max_groups = max((len(g) for g in groups_per_level), default=1)
        vertical_gap = min(0.018, max(0.006, 0.16 / (max_groups + 1)))
        scale = max((usable_span - vertical_gap * max(max_groups - 1, 0)) / total_strength, 1e-9)
        min_band = max(min_width_fraction * usable_span, 0.004)

        level_group_y: List[Dict[str, Tuple[float, float]]] = []
        level_heights: List[float] = []

        for lvl_idx, groups in enumerate(groups_per_level):
            y = 0.0
            positions: Dict[str, Tuple[float, float]] = {}
            for g in groups:
                h_raw = group_strengths_per_level[lvl_idx].get(g, 0.0) * scale
                h = max(h_raw, min_band)
                positions[g] = (y, y + h)
                y += h + vertical_gap

            height = max((v[1] for v in positions.values()), default=0.0)
            level_heights.append(height)
            level_group_y.append(positions)

        for lvl_idx, positions in enumerate(level_group_y):
            height = level_heights[lvl_idx]
            offset = span_ymin + 0.5 * (usable_span - height)
            for g in list(positions.keys()):
                y0, y1 = positions[g]
                positions[g] = (y0 + offset, y1 + offset)

        out_offset: List[Dict[str, float]] = []
        in_offset: List[Dict[str, float]] = []
        for lvl_idx, groups in enumerate(groups_per_level):
            out_offset.append({g: level_group_y[lvl_idx][g][0] for g in groups})
            in_offset.append({g: level_group_y[lvl_idx][g][0] for g in groups})

        branch_palette = [
            "#3C5488",
            "#00A087",
            "#E64B35",
            "#7E6148",
            "#4DBBD5",
            "#6F4C9B",
            "#F39B7F",
            "#8491B4",
        ]
        if high_contrast:
            # More saturated, higher-contrast palette (for light backgrounds)
            branch_palette = [
                "#1B5E9B",
                "#00897B",
                "#C62828",
                "#5D4037",
                "#0277BD",
                "#6A1B9A",
                "#E65100",
                "#37474F",
            ]

        main_level_idx = 1 if n_levels >= 2 and len(groups_per_level[1]) > 0 else 0
        main_level_key = level_keys[main_level_idx]
        main_groups = groups_per_level[main_level_idx]

        if main_level_key in _stable_set:
            sorted_main = sorted(main_groups, key=str)
        else:
            sorted_main = sorted(
                main_groups,
                key=lambda g: (-group_strengths_per_level[main_level_idx].get(g, 0.0), str(g)),
            )

        base_colors: Dict[str, str] = {}
        for i, g in enumerate(sorted_main):
            base_colors[g] = branch_palette[i % len(branch_palette)]

        group_branch_score: Dict[Tuple[int, str], Dict[str, float]] = {}
        for leaf in leaf_order:
            branch = leaf_to_group[main_level_key].get(leaf, leaf)
            w = float(leaf_strengths.get(leaf, 0.0))
            for lvl_idx, level_key in enumerate(level_keys):
                g = leaf_to_group[level_key].get(leaf, leaf)
                d = group_branch_score.setdefault((lvl_idx, g), defaultdict(float))
                d[branch] += w

        color_map: Dict[Tuple[int, str], Tuple[float, float, float, float]] = {}
        for lvl_idx, groups in enumerate(groups_per_level):
            for g in groups:
                if lvl_idx == main_level_idx and g in base_colors:
                    base = base_colors[g]
                else:
                    scores = group_branch_score.get((lvl_idx, g), {})
                    if scores:
                        dominant_branch = max(scores.items(), key=lambda kv: kv[1])[0]
                        base = base_colors.get(dominant_branch, branch_palette[0])
                    else:
                        base = branch_palette[0]

                if lvl_idx < main_level_idx:
                    light_amt = 0.12
                elif lvl_idx == main_level_idx:
                    light_amt = 0.00
                elif lvl_idx == main_level_idx + 1:
                    light_amt = 0.18
                else:
                    light_amt = 0.32
                if high_contrast:
                    if lvl_idx < main_level_idx:
                        light_amt = 0.06
                    elif lvl_idx == main_level_idx:
                        light_amt = 0.00
                    elif lvl_idx == main_level_idx + 1:
                        light_amt = 0.10
                    else:
                        light_amt = 0.18

                color_map[(lvl_idx, g)] = _mix_with_white(base, light_amt)

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor="white")
        ax.set_facecolor("white")

        for i in range(n_levels - 1):
            x_mid = 0.5 * (x_positions[i] + x_positions[i + 1])
            ax.axvline(x_mid, ymin=0.06, ymax=0.94, color="#EAEAEA", lw=0.7, zorder=0)

        for lvl_idx in range(n_levels - 1):
            flows = flows_per_level[lvl_idx]
            left_order = {g: i for i, g in enumerate(groups_per_level[lvl_idx])}
            right_order = {g: i for i, g in enumerate(groups_per_level[lvl_idx + 1])}

            ordered_flows = sorted(
                flows.items(),
                key=lambda kv: (
                    left_order.get(kv[0][0], 10**9),
                    right_order.get(kv[0][1], 10**9),
                    -kv[1],
                ),
            )

            for (g_left, g_right), w in ordered_flows:
                if g_left not in level_group_y[lvl_idx] or g_right not in level_group_y[lvl_idx + 1]:
                    continue

                h_raw = w * scale
                if h_raw <= 0:
                    continue
                h = max(h_raw, min_band * 0.55)

                y0_left = out_offset[lvl_idx][g_left]
                y1_left = y0_left + h
                out_offset[lvl_idx][g_left] = y1_left

                y0_right = in_offset[lvl_idx + 1][g_right]
                y1_right = y0_right + h
                in_offset[lvl_idx + 1][g_right] = y1_right

                x_left = x_positions[lvl_idx] + bar_half_width
                x_right = x_positions[lvl_idx + 1] - bar_half_width

                t = np.linspace(0.0, 1.0, 60, dtype=float)
                s = _smoothstep(t)
                x_curve = x_left + (x_right - x_left) * t
                y_top = y1_left + (y1_right - y1_left) * s
                y_bottom = y0_left + (y0_right - y0_left) * s

                xs = np.concatenate([x_curve, x_curve[::-1]])
                ys = np.concatenate([y_top, y_bottom[::-1]])
                verts = np.column_stack([xs, ys])

                flow_color = color_map.get((lvl_idx, g_left), (0.6, 0.6, 0.6, 1.0))
                poly = Polygon(
                    verts,
                    closed=True,
                    facecolor=flow_color,
                    edgecolor=(1, 1, 1, 0.18),
                    linewidth=0.25,
                    alpha=0.52,
                    antialiased=True,
                    zorder=1,
                )
                ax.add_patch(poly)

        for lvl_idx, groups in enumerate(groups_per_level):
            x_center = x_positions[lvl_idx]
            for g in groups:
                y0, y1 = level_group_y[lvl_idx][g]
                h = y1 - y0
                color = color_map.get((lvl_idx, g), (0.75, 0.75, 0.75, 1.0))
                patch = FancyBboxPatch(
                    (x_center - bar_half_width, y0),
                    2 * bar_half_width,
                    h,
                    boxstyle="round,pad=0,rounding_size=0.010",
                    linewidth=0.35,
                    edgecolor=(1, 1, 1, 0.75),
                    facecolor=color,
                    mutation_aspect=1.0,
                    zorder=3,
                )
                ax.add_patch(patch)

        if svg_plain_text:
            outer_text_effects = []
            inner_text_effects = []
        else:
            outer_text_effects = [pe.withStroke(linewidth=2.4, foreground="white", alpha=0.95)]
            inner_text_effects = [pe.withStroke(linewidth=2.0, foreground="white", alpha=0.9)]
        text_color_outer = "#222222"
        text_color_inner = "#1F1F1F"

        label_min_h_outer = 0.018
        label_min_h_inner = 0.028
        label_min_h_last = 0.010

        if n_levels >= 1:
            lvl_idx = 0
            x_center = x_positions[lvl_idx]
            for g in groups_per_level[lvl_idx]:
                y0, y1 = level_group_y[lvl_idx][g]
                h = y1 - y0
                if h < label_min_h_outer:
                    continue
                ax.text(
                    x_center - bar_half_width - 0.07,
                    0.5 * (y0 + y1),
                    _ellipsize(g, 24),
                    ha="right",
                    va="center",
                    fontsize=7,
                    color=text_color_outer,
                    path_effects=outer_text_effects,
                    zorder=5,
                )

        for lvl_idx in range(1, max(1, n_levels - 1)):
            if lvl_idx >= n_levels - 1:
                break
            x_center = x_positions[lvl_idx]
            codes = group_codes_per_level[lvl_idx]
            for g in groups_per_level[lvl_idx]:
                y0, y1 = level_group_y[lvl_idx][g]
                h = y1 - y0
                if h < label_min_h_inner:
                    continue
                code = codes.get(g, "")
                if not code:
                    continue
                ax.text(
                    x_center,
                    0.5 * (y0 + y1),
                    code,
                    ha="center",
                    va="center",
                    fontsize=6.4,
                    color=text_color_inner,
                    path_effects=inner_text_effects,
                    zorder=5,
                )

        if n_levels >= 2:
            last_idx = n_levels - 1
            x_center = x_positions[last_idx]
            x_text = x_center + bar_half_width + 0.16
            x_line_end = x_center + bar_half_width + 0.11

            candidates: List[Dict[str, Any]] = []
            for g in groups_per_level[last_idx]:
                y0, y1 = level_group_y[last_idx][g]
                h = y1 - y0
                if h < label_min_h_last:
                    continue
                candidates.append(
                    {
                        "group": g,
                        "anchor_y": 0.5 * (y0 + y1),
                        "y0": y0,
                        "y1": y1,
                        "height": h,
                    }
                )

            if candidates:
                candidates.sort(key=lambda d: d["anchor_y"])
                desired = [d["anchor_y"] for d in candidates]
                min_gap = 0.018
                low = span_ymin + 0.005
                high = span_ymax - 0.005

                placed = _spread_positions(
                    desired_positions=desired,
                    low=low,
                    high=high,
                    min_gap=min_gap,
                )

                for d, y_lab in zip(candidates, placed):
                    g = d["group"]
                    y_anchor = d["anchor_y"]

                    ax.plot(
                        [x_center + bar_half_width + 0.01, x_line_end, x_text - 0.01],
                        [y_anchor, y_lab, y_lab],
                        color="#9A9A9A",
                        lw=0.55,
                        solid_capstyle="round",
                        zorder=4,
                    )
                    ax.text(
                        x_text,
                        y_lab,
                        _ellipsize(g, 28),
                        ha="left",
                        va="center",
                        fontsize=6.6,
                        color=text_color_outer,
                        path_effects=outer_text_effects,
                        zorder=5,
                    )

        for lvl_idx, x in enumerate(x_positions):
            ax.text(
                x,
                0.94,
                f"Level {lvl_idx + 1}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#4A4A4A",
                zorder=6,
            )

        if title:
            ax.text(
                x_positions[0] - 0.45,
                0.985,
                title,
                ha="left",
                va="top",
                fontsize=8,
                color=text_color_outer,
                zorder=6,
            )

        legend_lines: List[str] = []
        for lvl_idx in range(1, n_levels - 1):
            codes = group_codes_per_level[lvl_idx]
            if not codes:
                continue
            legend_lines.append(f"Level {lvl_idx + 1}")
            for g in groups_per_level[lvl_idx]:
                code = codes.get(g, "")
                if not code:
                    continue
                legend_lines.append(f"  {code}: {g}")
            legend_lines.append("")

        legend_text = "\n".join(legend_lines).rstrip()

        extra_right_for_last_labels = 0.95
        ax.set_xlim(x_positions[0] - 0.55, x_positions[-1] + extra_right_for_last_labels)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

        if show_legend and legend_text:
            fig.subplots_adjust(right=0.80)
            fig.text(
                0.82,
                0.50,
                legend_text,
                ha="left",
                va="center",
                fontsize=6.1,
                color="#2A2A2A",
                linespacing=1.08,
            )
            plt.tight_layout(rect=[0.0, 0.0, 0.80, 1.0], pad=0.4)
        else:
            plt.tight_layout(pad=0.4)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        if path_wants_svg(save_path):
            configure_matplotlib_svg_for_illustrator()
        plt.savefig(
            save_path,
            dpi=max(dpi, 300),
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
            pad_inches=0.03,
        )
        plt.close(fig)


def _ellipsize(text: str, max_chars: int = 18) -> str:
    text = str(text)
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


def _smoothstep(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


def _mix_with_white(
    color: Union[str, Tuple[float, float, float, float]],
    amount: float,
) -> Tuple[float, float, float, float]:
    rgba = np.array(mcolors.to_rgba(color), dtype=float)
    rgba[:3] = rgba[:3] * (1.0 - amount) + amount
    return tuple(rgba)


def _path_key_for_leaf(
    leaf: str,
    level_keys: List[str],
    leaf_to_group: Dict[str, Dict[str, str]],
    leaf_strengths: Mapping[str, float],
) -> Tuple[Any, ...]:
    path = tuple(str(leaf_to_group[level_key].get(leaf, leaf)) for level_key in level_keys)
    return path + (-float(leaf_strengths.get(leaf, 0.0)), str(leaf))


def _spread_positions(
    desired_positions: List[float],
    low: float,
    high: float,
    min_gap: float,
) -> List[float]:
    if not desired_positions:
        return []

    placed = [0.0] * len(desired_positions)

    # forward pass
    placed[0] = max(low, min(high, desired_positions[0]))
    for i in range(1, len(desired_positions)):
        placed[i] = max(desired_positions[i], placed[i - 1] + min_gap)

    # if overflow, shift downward
    overflow = placed[-1] - high
    if overflow > 0:
        for i in range(len(placed)):
            placed[i] -= overflow

    # backward pass to respect low bound and gaps
    if placed[0] < low:
        placed[0] = low
        for i in range(1, len(placed)):
            placed[i] = max(placed[i], placed[i - 1] + min_gap)

    # final clamp by backward correction
    if placed[-1] > high:
        placed[-1] = high
        for i in range(len(placed) - 2, -1, -1):
            placed[i] = min(placed[i], placed[i + 1] - min_gap)

    return placed


def _sanitize_filename(name: str) -> str:
    """Sanitize string for safe file names."""
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(name))
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe or "item"


def _sorted_level_keys(hierarchy_dict: Mapping[str, Any]) -> List[str]:
    level_keys = [k for k in hierarchy_dict.keys() if k.startswith("level_")]
    level_keys.sort(key=lambda x: int(x.split("_")[1]))
    return level_keys


def _build_leaf_to_group_map(
    hierarchy_dict: Mapping[str, Any],
    leaf_names: Iterable[str],
    level_keys: List[str],
) -> Dict[str, Dict[str, str]]:
    leaf_set = set(leaf_names)
    out: Dict[str, Dict[str, str]] = {}

    for level_key in level_keys:
        mapping: Dict[str, str] = {}
        level_data = hierarchy_dict.get(level_key, {})

        for key, value in level_data.items():
            if isinstance(value, dict) and "edge_type_name" in value:
                leaf_name = value["edge_type_name"]
                if leaf_name in leaf_set:
                    mapping[leaf_name] = key
            else:
                if not isinstance(value, list):
                    continue
                for item in value:
                    leaf_name = item.get("edge_type_name")
                    if leaf_name in leaf_set:
                        mapping[leaf_name] = key

        for leaf in leaf_set:
            mapping.setdefault(leaf, leaf)
        out[level_key] = mapping

    return out
