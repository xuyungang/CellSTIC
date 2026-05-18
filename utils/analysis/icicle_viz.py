from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.patches import Circle, Wedge

from utils.viz.matplotlib_svg import configure_matplotlib_svg_for_illustrator, path_wants_svg

from .aggregated_heatmap_viz import AggregatedHeatmapVisualizer
from .cell_type_communication import CellTypeCommunicationComputer


# Nature / Springer Nature figure guidance emphasizes editable vector text,
# sans-serif fonts (preferably Arial/Helvetica), small but readable figure text,
# and thin clean strokes.
_NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
    "font.size": 6.5,
    "axes.linewidth": 0.5,
    "axes.edgecolor": "#2B2B2B",
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 1.8,
    "ytick.major.size": 1.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "none",
    "patch.antialiased": True,
    "lines.antialiased": True,
    "text.antialiased": True,
}


@dataclass
class _RadialNode:
    name: str
    level_idx: int
    node_id: Tuple[str, ...]
    weight: float = 0.0
    first_pos: int = 10**9
    children: Dict[str, "_RadialNode"] = field(default_factory=dict)
    branch_scores: Dict[Tuple[str, ...], float] = field(default_factory=dict)
    angle_span: Tuple[float, float] = (0.0, 0.0)


@dataclass
class _OuterLabelSpec:
    text: str
    theta_mid: float
    fill: Tuple[float, float, float, float]
    fontsize: float
    anchor_xy: Tuple[float, float]
    elbow_xy: Tuple[float, float]
    side: str


class IcicleVisualizer:
    """Icicle plot (sunburst-style) over LR hierarchy tree.

    Public API is intentionally unchanged, although the internal plot is circular
    rather than icicle.
    """

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
    ) -> None:
        """Draw one circular hierarchical plot for given leaf-level strengths."""
        plt.rcParams.update(_NATURE_RC)

        # path_effects on text become SVG paths; disable for .svg so AI can select text.
        svg_plain_text = path_wants_svg(save_path)

        if not level_keys:
            return

        _stable_set: Set[str] = set(stable_order_level_keys) if stable_order_level_keys else set()

        leaf_names = [ln for ln in leaf_names if ln in leaf_strengths]
        if not leaf_names:
            return

        safe_strengths = {ln: max(float(leaf_strengths.get(ln, 0.0)), 0.0) for ln in leaf_names}
        total_strength = float(sum(safe_strengths.values()))
        if total_strength <= 0:
            safe_strengths = {ln: 1.0 for ln in leaf_names}
            total_strength = float(len(leaf_names))

        leaf_order = sorted(
            leaf_names,
            key=lambda ln: _path_key_for_leaf(ln, level_keys, leaf_to_group, safe_strengths),
        )
        leaf_pos = {leaf: i for i, leaf in enumerate(leaf_order)}
        n_levels = len(level_keys)

        main_level_idx = 1 if n_levels >= 2 else 0

        root, nodes_by_level = _build_radial_tree(
            level_keys=level_keys,
            leaf_order=leaf_order,
            leaf_pos=leaf_pos,
            leaf_to_group=leaf_to_group,
            leaf_strengths=safe_strengths,
            main_level_idx=main_level_idx,
        )
        if not any(nodes_by_level):
            return

        level_codes = _build_level_codes(nodes_by_level)
        _layout_radial_tree(root, start_angle=90.0, end_angle=450.0)

        branch_palette = _get_branch_palette(high_contrast=high_contrast)

        main_level_key = level_keys[main_level_idx]
        main_nodes = nodes_by_level[main_level_idx] if 0 <= main_level_idx < len(nodes_by_level) else []

        if main_level_key in _stable_set:
            sorted_main_nodes = sorted(main_nodes, key=lambda n: (str(n.name), n.first_pos, n.node_id))
        else:
            sorted_main_nodes = sorted(main_nodes, key=lambda n: (-n.weight, n.first_pos, str(n.name), n.node_id))

        base_colors: Dict[Tuple[str, ...], str] = {}
        for i, node in enumerate(sorted_main_nodes):
            base_colors[node.node_id] = branch_palette[i % len(branch_palette)]

        def _level_light_amt(lvl_idx: int, main_level_idx: int, n_levels: int) -> float:
            if lvl_idx < main_level_idx:
                return 0.10
            if lvl_idx == main_level_idx:
                return 0.00
            if lvl_idx == main_level_idx + 1:
                return 0.07
            if lvl_idx == n_levels - 1:
                return 0.12
            return 0.09

        color_map: Dict[Tuple[str, ...], Tuple[float, float, float, float]] = {}
        for lvl_idx, level_nodes in enumerate(nodes_by_level):
            for node in level_nodes:
                if lvl_idx == main_level_idx and node.node_id in base_colors:
                    base = base_colors[node.node_id]
                else:
                    if node.branch_scores:
                        dominant_branch = max(node.branch_scores.items(), key=lambda kv: kv[1])[0]
                        base = base_colors.get(dominant_branch, branch_palette[0])
                    else:
                        base = branch_palette[0]

                color_map[node.node_id] = _mix_with_white(
                    base,
                    _level_light_amt(lvl_idx, main_level_idx, n_levels),
                )

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor="white")
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.set_aspect("equal")

        # Compact layout: slightly larger center hole, tighter ring gaps, shorter outside label offset.
        inner_hole = 0.34
        outer_radius = 0.985
        ring_gap = 0.010 if n_levels >= 4 else 0.012
        usable_radial = outer_radius - inner_hole - ring_gap * max(n_levels - 1, 0)
        ring_width = max(usable_radial / max(n_levels, 1), 0.075)

        outer_frame = Circle(
            (0.0, 0.0),
            radius=outer_radius + 0.0015,
            facecolor="none",
            edgecolor="#ECECEC",
            linewidth=0.42,
            zorder=0,
        )
        ax.add_patch(outer_frame)

        for lvl_idx in range(n_levels):
            if lvl_idx == 0:
                continue
            r_in = inner_hole + lvl_idx * (ring_width + ring_gap)
            r_out = r_in + ring_width
            guide = Wedge(
                center=(0.0, 0.0),
                r=r_out,
                theta1=0.0,
                theta2=360.0,
                width=ring_width,
                facecolor="#FDFDFD",
                edgecolor="#F2F2F2",
                linewidth=0.28,
                zorder=0,
            )
            ax.add_patch(guide)

        outer_label_specs: List[_OuterLabelSpec] = []

        for lvl_idx, level_nodes in enumerate(nodes_by_level):
            if lvl_idx == 0:
                continue

            r_in = inner_hole + lvl_idx * (ring_width + ring_gap)
            r_out = r_in + ring_width
            r_mid = 0.5 * (r_in + r_out)

            level_nodes_sorted = sorted(
                level_nodes,
                key=lambda n: ((n.angle_span[1] - n.angle_span[0]), n.first_pos),
                reverse=True,
            )

            for node in level_nodes_sorted:
                theta1, theta2 = node.angle_span
                span = theta2 - theta1
                if span <= 0:
                    continue

                fill = color_map.get(node.node_id, mcolors.to_rgba("#C9CED6"))
                edge_rgba = _mix_with_white(fill, 0.58)

                wedge = Wedge(
                    center=(0.0, 0.0),
                    r=r_out,
                    theta1=theta1,
                    theta2=theta2,
                    width=ring_width,
                    facecolor=fill,
                    edgecolor=edge_rgba,
                    linewidth=0.44,
                    joinstyle="round",
                    antialiased=True,
                    zorder=2,
                )
                ax.add_patch(wedge)

                arc_len = r_mid * np.deg2rad(span)

                if 0 < lvl_idx < n_levels - 1 and node.node_id in level_codes[lvl_idx]:
                    display_text = level_codes[lvl_idx][node.node_id]
                    min_span = 12.5
                    min_arc = 0.13
                    fontweight = "semibold"
                else:
                    if lvl_idx == n_levels - 1:
                        frac = float(node.weight) / total_strength if total_strength > 0 else 0.0
                        display_text = f"{node.name} ({frac * 100:.1f}%)"
                        min_span = 8.5
                        min_arc = 0.10
                    else:
                        display_text = _ellipsize(node.name, 18)
                        min_span = 13.5
                        min_arc = 0.16
                    fontweight = "normal"

                if span < min_span or arc_len < min_arc:
                    continue

                mid = 0.5 * (theta1 + theta2)
                fontsize = _font_size_for_arc(span, arc_len, is_last_level=(lvl_idx == n_levels - 1))

                if lvl_idx == n_levels - 1:
                    # Outer-most labels are moved just outside the ring so their font
                    # color can exactly match the corresponding wedge color without
                    # sacrificing readability.
                    side = "right" if np.cos(np.deg2rad(mid)) >= 0 else "left"
                    x_anchor, y_anchor = _polar_to_xy(r_out + 0.006, mid)
                    x_elbow, y_elbow = _polar_to_xy(outer_radius + 0.020, mid)
                    outer_label_specs.append(
                        _OuterLabelSpec(
                            text=display_text,
                            theta_mid=mid,
                            fill=fill,
                            fontsize=max(6.2, min(fontsize, 7.5)),
                            anchor_xy=(x_anchor, y_anchor),
                            elbow_xy=(x_elbow, y_elbow),
                            side=side,
                        )
                    )
                    continue

                txt_color, txt_effects = _text_style_for_fill(fill)
                if svg_plain_text:
                    txt_effects = []

                x, y = _polar_to_xy(r_mid, mid)
                rotation = _tangent_rotation(mid)

                txt = ax.text(
                    x,
                    y,
                    display_text,
                    ha="center",
                    va="center",
                    rotation=rotation,
                    rotation_mode="anchor",
                    fontsize=fontsize,
                    fontweight=fontweight,
                    color=txt_color,
                    path_effects=txt_effects,
                    zorder=4,
                )
                txt.set_clip_path(wedge)

        _draw_outer_labels(
            ax=ax,
            label_specs=outer_label_specs,
            outer_radius=outer_radius,
            svg_plain_text=svg_plain_text,
        )

        center_radius = inner_hole + 0.72 * ring_width
        center = Circle(
            (0.0, 0.0),
            radius=center_radius,
            facecolor="white",
            edgecolor="#EAEAEA",
            linewidth=0.55,
            zorder=5,
        )
        ax.add_patch(center)

        center_inner = Circle(
            (0.0, 0.0),
            radius=max(center_radius - 0.013, 0.0),
            facecolor="white",
            edgecolor="none",
            zorder=5,
        )
        ax.add_patch(center_inner)

        center_title = _format_center_title(title)
        if center_title:
            ax.text(
                0.0,
                0.01,
                center_title,
                ha="center",
                va="center",
                fontsize=6.8,
                fontweight="semibold",
                color="#111111",
                linespacing=1.10,
                path_effects=(
                    []
                    if svg_plain_text
                    else [pe.withStroke(linewidth=1.0, foreground="white", alpha=0.94)]
                ),
                zorder=6,
            )

        # Tighter canvas while still leaving room for short outside labels.
        limit = 1.16
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-1.09, 1.09)
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.tight_layout(pad=0.10)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        if path_wants_svg(save_path):
            configure_matplotlib_svg_for_illustrator()
        plt.savefig(
            save_path,
            dpi=max(dpi, 400),
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
            pad_inches=0.018,
        )
        plt.close(fig)


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



def _build_radial_tree(
    level_keys: List[str],
    leaf_order: List[str],
    leaf_pos: Mapping[str, int],
    leaf_to_group: Dict[str, Dict[str, str]],
    leaf_strengths: Mapping[str, float],
    main_level_idx: int,
) -> Tuple[_RadialNode, List[List[_RadialNode]]]:
    root = _RadialNode(name="root", level_idx=-1, node_id=())
    nodes_by_level: List[List[_RadialNode]] = [[] for _ in level_keys]

    for leaf in leaf_order:
        weight = max(float(leaf_strengths.get(leaf, 0.0)), 0.0)
        if weight <= 0:
            continue

        path_groups = [str(leaf_to_group[level_key].get(leaf, leaf)) for level_key in level_keys]
        if not path_groups:
            continue

        node = root
        prefix: List[str] = []
        nodes_on_path: List[_RadialNode] = []

        for lvl_idx, group_name in enumerate(path_groups):
            prefix.append(group_name)
            child = node.children.get(group_name)
            if child is None:
                child = _RadialNode(
                    name=group_name,
                    level_idx=lvl_idx,
                    node_id=tuple(prefix),
                )
                node.children[group_name] = child
                nodes_by_level[lvl_idx].append(child)

            child.weight += weight
            child.first_pos = min(child.first_pos, int(leaf_pos[leaf]))
            nodes_on_path.append(child)
            node = child

        branch_idx = min(max(main_level_idx, 0), len(nodes_on_path) - 1)
        main_branch_id = nodes_on_path[branch_idx].node_id
        for nd in nodes_on_path:
            nd.branch_scores[main_branch_id] = nd.branch_scores.get(main_branch_id, 0.0) + weight

    return root, nodes_by_level



def _build_level_codes(nodes_by_level: List[List[_RadialNode]]) -> List[Dict[Tuple[str, ...], str]]:
    out: List[Dict[Tuple[str, ...], str]] = []
    n_levels = len(nodes_by_level)

    for lvl_idx, nodes in enumerate(nodes_by_level):
        codes: Dict[Tuple[str, ...], str] = {}
        if 0 < lvl_idx < n_levels - 1:
            ordered = sorted(nodes, key=lambda n: (n.first_pos, -n.weight, str(n.name), n.node_id))
            for j, node in enumerate(ordered, start=1):
                codes[node.node_id] = f"L{lvl_idx + 1}-{j}"
        out.append(codes)

    return out



def _layout_radial_tree(
    root: _RadialNode,
    start_angle: float,
    end_angle: float,
) -> None:
    def _recurse(parent: _RadialNode, a0: float, a1: float, depth: int) -> None:
        children = sorted(
            parent.children.values(),
            key=lambda n: (n.first_pos, -n.weight, str(n.name), n.node_id),
        )
        if not children:
            return

        span = max(a1 - a0, 0.0)
        n = len(children)
        if n <= 1:
            gap = 0.0
        else:
            target_gap = 1.10 if depth <= 1 else 0.72
            gap = min(target_gap, 0.14 * span / (n + 0.5))
            gap = max(gap, 0.08 if span > 22 else 0.0)

        total = sum(max(c.weight, 0.0) for c in children)

        cursor = a0
        remaining_weight = total
        remaining_children = n

        for idx, child in enumerate(children):
            if idx == n - 1:
                child_a0 = cursor
                child_a1 = a1
            else:
                remaining_span = max(a1 - cursor - gap * (remaining_children - 1), 0.0)
                if remaining_weight <= 0:
                    frac = 1.0 / max(remaining_children, 1)
                else:
                    frac = max(child.weight, 0.0) / remaining_weight

                child_span = remaining_span * frac
                child_a0 = cursor
                child_a1 = cursor + child_span

            child.angle_span = (child_a0, child_a1)
            _recurse(child, child_a0, child_a1, depth + 1)

            cursor = child_a1 + gap
            remaining_weight -= max(child.weight, 0.0)
            remaining_children -= 1

    _recurse(root, start_angle, end_angle, depth=0)



def _ellipsize(text: str, max_chars: int = 18) -> str:
    text = str(text)
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"



def _get_branch_palette(high_contrast: bool = True) -> List[str]:
    if high_contrast:
        return [
            "#2F5D8A",
            "#2E8B7F",
            "#C65D4B",
            "#7A5C99",
            "#8D9A52",
            "#A06A45",
            "#5E7E99",
            "#B36A8C",
        ]
    return [
        "#4C78A8",
        "#54A24B",
        "#E17C05",
        "#B279A2",
        "#72B7B2",
        "#9D755D",
        "#BAB0AC",
        "#D65F5F",
    ]



def _mix_with_white(
    color: Union[str, Tuple[float, float, float, float]],
    amount: float,
) -> Tuple[float, float, float, float]:
    rgba = np.array(mcolors.to_rgba(color), dtype=float)
    rgba[:3] = rgba[:3] * (1.0 - amount) + amount
    return tuple(rgba)



def _relative_luminance(color: Union[str, Tuple[float, float, float, float]]) -> float:
    rgba = np.array(mcolors.to_rgba(color), dtype=float)
    rgb = rgba[:3]

    def _linearize(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (_linearize(float(c)) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b



def _text_style_for_fill(
    fill: Union[str, Tuple[float, float, float, float]]
) -> Tuple[str, List[Any]]:
    lum = _relative_luminance(fill)
    if lum < 0.22:
        return "#FFFFFF", [pe.withStroke(linewidth=1.2, foreground=(0, 0, 0, 0.18))]
    if lum < 0.42:
        return "#F8F8F8", [pe.withStroke(linewidth=1.2, foreground=(0, 0, 0, 0.15))]
    return "#1A1A1A", [pe.withStroke(linewidth=1.2, foreground=(1, 1, 1, 0.34))]



def _font_size_for_arc(span_deg: float, arc_len: float, is_last_level: bool) -> float:
    if is_last_level:
        if span_deg >= 40 and arc_len >= 0.34:
            return 7.2
        if span_deg >= 24 and arc_len >= 0.21:
            return 6.8
        return 6.4

    if span_deg >= 40 and arc_len >= 0.34:
        return 6.6
    if span_deg >= 26 and arc_len >= 0.24:
        return 6.2
    if span_deg >= 16 and arc_len >= 0.17:
        return 5.8
    return 5.4



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

    placed[0] = max(low, min(high, desired_positions[0]))
    for i in range(1, len(desired_positions)):
        placed[i] = max(desired_positions[i], placed[i - 1] + min_gap)

    overflow = placed[-1] - high
    if overflow > 0:
        for i in range(len(placed)):
            placed[i] -= overflow

    if placed[0] < low:
        placed[0] = low
        for i in range(1, len(placed)):
            placed[i] = max(placed[i], placed[i - 1] + min_gap)

    if placed[-1] > high:
        placed[-1] = high
        for i in range(len(placed) - 2, -1, -1):
            placed[i] = min(placed[i], placed[i + 1] - min_gap)

    return placed



def _polar_to_xy(r: float, angle_deg: float) -> Tuple[float, float]:
    rad = np.deg2rad(angle_deg % 360.0)
    return float(r * np.cos(rad)), float(r * np.sin(rad))



def _tangent_rotation(angle_deg: float) -> float:
    a = angle_deg % 360.0
    rot = a - 90.0
    if 90.0 < a < 270.0:
        rot += 180.0
    return rot



def _radial_text_rotation_and_alignment(angle_deg: float) -> Tuple[float, str]:
    a = angle_deg % 360.0
    if 90.0 < a < 270.0:
        return a - 180.0, "right"
    return a, "left"



def _format_center_title(title: str, max_chars: int = 20) -> str:
    """Format center text as 'source → target' without truncation."""
    if not title:
        return ""

    title = str(title).strip()

    if "→" in title:
        left, right = [s.strip() for s in title.split("→", 1)]
    elif "->" in title:
        left, right = [s.strip() for s in title.split("->", 1)]
    else:
        return title

    return f"{left}\n→\n{right}"



def _draw_outer_labels(
    ax: plt.Axes,
    label_specs: List[_OuterLabelSpec],
    outer_radius: float,
    svg_plain_text: bool,
) -> None:
    if not label_specs:
        return

    # Compact, bilateral outside labels.
    x_text_offset = outer_radius + 0.085
    x_line_end = outer_radius + 0.048
    y_low, y_high = -0.98, 0.98

    for side in ("left", "right"):
        subset = [lab for lab in label_specs if lab.side == side]
        if not subset:
            continue

        subset = sorted(subset, key=lambda lab: lab.elbow_xy[1])
        desired_y = [lab.elbow_xy[1] for lab in subset]
        min_gap = max(0.056, 0.050 + 0.0025 * max(len(subset) - 1, 0))
        spread_y = _spread_positions(desired_y, low=y_low, high=y_high, min_gap=min_gap)

        sign = 1.0 if side == "right" else -1.0
        ha = "left" if side == "right" else "right"
        x_text = sign * x_text_offset
        x_line = sign * x_line_end

        for lab, y_text in zip(subset, spread_y):
            x_anchor, y_anchor = lab.anchor_xy
            x_elbow, y_elbow = lab.elbow_xy
            line_color = lab.fill

            ax.plot(
                [x_anchor, x_elbow, x_line],
                [y_anchor, y_elbow, y_text],
                color=line_color,
                linewidth=0.48,
                solid_capstyle="round",
                solid_joinstyle="round",
                alpha=0.95,
                zorder=3,
            )

            text_effects: List[Any] = []
            if not svg_plain_text:
                text_effects = [pe.withStroke(linewidth=1.1, foreground="white", alpha=0.96)]

            ax.text(
                x_text,
                y_text,
                lab.text,
                ha=ha,
                va="center",
                fontsize=lab.fontsize,
                fontweight="normal",
                color=line_color,
                path_effects=text_effects,
                zorder=4,
            )
