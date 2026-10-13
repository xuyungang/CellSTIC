"""Temporal summaries drawn with the publication layouts and colors.

These renderers consume TimeSeriesAnalysis tables and do not load training code.
Spatial networks and domain plots use the existing visualization package.
"""
from __future__ import annotations

from pathlib import Path
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_hex
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, LogLocator, MaxNLocator
from scipy.optimize import linear_sum_assignment
from .result_io import load_result, lr_matrices
from .dataset_analysis import group_communication
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

SPATIAL_CMAP = LinearSegmentedColormap.from_list(
    "mouse_brain_blue", ["#EFF3F7", "#B5C9DE", "#5D88B2", "#244E78"]
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

def cell_type_colors(labels):
    return {
        label: (
            "#B8B8B8"
            if label in {"Other", "Others"}
            else DOMAIN_PALETTE[i % len(DOMAIN_PALETTE)]
        )
        for i, label in enumerate(sorted(set(map(str, labels))))
    }

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
MOUSE_BRAIN_RC = RC

RC = {
    **MOUSE_BRAIN_RC,
    "font.size": 6,
    "axes.titlesize": 6,
    "axes.labelsize": 5.5,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    "legend.fontsize": 5,
}

BLUE, CORAL, GREEN, GOLD, TEAL, MAUVE = DOMAIN_PALETTE[:6]

ORGAN_COLORS = {"Brain": BLUE, "Liver": CORAL}

PROCESS_COLORS = {"Development": BLUE, "Regeneration": CORAL}

MOUSE_STAGES = ["9.5", "10.5", "11.5", "12.5", "13.5", "14.5", "15.5", "16.5"]

DEVELOPMENT_STAGES = ["develop_44", "develop_54", "develop_57", "develop_Juvenile", "develop_Adult", "develop_Metamorphosed"]

REGENERATION_STAGES = [f"regeneration_{s}" for s in (2, 5, 10, 15, 20, 30, 60, "control")]

AXO_STAGES = DEVELOPMENT_STAGES + REGENERATION_STAGES

STAGE_LABELS = dict(zip(DEVELOPMENT_STAGES, ["E44", "E54", "E57", "Juv.", "Adult", "Meta."]))
STAGE_LABELS.update({s: ("Ctrl" if s.endswith("control") else "R" + s.split("_")[-1]) for s in REGENERATION_STAGES})
STAGE_LABELS.update({s: "E" + s for s in MOUSE_STAGES})

MOUSE_PAIRS = ["F2_F2r", "Lpar3_Adgre5", "Nts_Sort1", "Plg_Pard3", "Thbs4_Cd36"]

PAIR_COLORS = dict(zip(MOUSE_PAIRS, [BLUE, CORAL, TEAL, MAUVE, GOLD]))

SELECTED_PAIRS = {"Brain": "Nts_Sort1", "Liver": "F2_F2r", "telencephalon": "WNT7B_FZD5"}

LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (5, 1, 1, 1, 1, 1))]

METRICS = {"k_out": "Out-degree", "k_in": "In-degree", "s_out": "Out-strength", "s_in": "In-strength"}

def log_tick_label(value, _position):
    """Keep log tick text at 5 pt, including exponent glyphs."""
    if value <= 0:
        return ""
    exponent = int(round(np.log10(value)))
    if np.isclose(value, 10.**exponent, rtol=1e-8, atol=0):
        return "10" + str(exponent).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{value:g}"

def subaxis(fig, rect, left, top, width, height):
    x, y, w, h = rect
    return axes_mm(fig, x + left*w, y + top*h, width*w, height*h)

def polish(ax, *, grid=False):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(pad=1.2)
    if grid:
        ax.grid(axis="y", color="#E8EDF1", lw=.35, zorder=0)

def stage_ticks(ax, stages, *, rotation=40):
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels([STAGE_LABELS[s] for s in stages], rotation=rotation,
                       ha="right" if rotation else "center")
    ax.set_xlim(-.35, len(stages) - .65)

def _decode(values):
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in values])

def read_obs_column(handle, key):
    """Read modern and legacy AnnData categorical columns without loading X."""
    obs = handle["obs"]
    node = obs[key]
    if isinstance(node, h5py.Group):
        categories, codes = _decode(node["categories"][:]), node["codes"][:]
    elif "categories" in node.attrs:
        categories, codes = _decode(handle[node.attrs["categories"]][:]), node[:]
    elif "__categories" in obs and key in obs["__categories"]:
        categories, codes = _decode(obs["__categories"][key][:]), node[:]
    else:
        return _decode(node[:])
    labels = np.full(len(codes), "Unknown", dtype=object)
    valid = codes >= 0
    labels[valid] = categories[codes[valid]]
    return labels.astype(str)

def spatial_frame(path, stage, label_key):
    with h5py.File(path, "r") as handle:
        xy = np.asarray(handle["obsm/spatial"])[:, :2]
        labels = read_obs_column(handle, label_key)
        index_key = handle["obs"].attrs.get("_index", "_index")
        if isinstance(index_key, bytes):
            index_key = index_key.decode()
        if index_key not in handle["obs"]:
            index_key = "cell_name" if "cell_name" in handle["obs"] else "CellID"
        ids = read_obs_column(handle, index_key)
    valid = np.isfinite(xy).all(axis=1)
    if not valid.all():
        raise ValueError(f"Nonfinite spatial coordinates: {path}")
    return pd.DataFrame(dict(stage=stage, cell_id=ids, x=xy[:, 0], y=xy[:, 1], annotation=labels))

def draw_spatial_row(fig, rect, table, stages, title, color, *, common_scale=True):
    x, y, w, h = rect
    arrow_ax = axes_mm(fig, x, y, w, 4)
    arrow_ax.annotate("", xy=(1, .35), xytext=(0, .35), xycoords="axes fraction",
                      arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3))
    arrow_ax.text(.5, .9, title, ha="center", va="bottom", fontsize=6, transform=arrow_ax.transAxes)
    arrow_ax.set_axis_off()
    spans = table[table.stage.isin(stages)].groupby("stage")[["x", "y"]].agg(lambda v: v.max()-v.min())
    span_x, span_y = spans.max().to_numpy() * 1.06
    for i, stage in enumerate(stages):
        ax = axes_mm(fig, x + i*w/len(stages), y + 7, w/len(stages)-.7, h-7)
        sub = table[table.stage.eq(stage)]
        # Marker area scales with density while every measured spot is retained.
        size = max(.10, min(1.1, 5500 / max(len(sub), 1)))
        ax.scatter(sub.x, sub.y, c=sub.display_color, s=size, linewidths=0, rasterized=True)
        cx, cy = (sub.x.min()+sub.x.max())/2, (sub.y.min()+sub.y.max())/2
        if common_scale:
            ax.set_xlim(cx-span_x/2, cx+span_x/2)
            ax.set_ylim(cy-span_y/2, cy+span_y/2)
        ax.set_aspect("equal")
        label = text_mm(fig, x+(i+.5)*w/len(stages), y+4, STAGE_LABELS[stage], fontsize=5)
        label.set_ha("center")
        ax.set_axis_off()
        rasterize_spatial_layer(ax)

def categorical_legend(fig, rect, colors, *, ncols=8, fontsize=5):
    x, y, w, h = rect
    items = list(colors.items())
    nrows = int(np.ceil(len(items)/ncols))
    for i, (label, color) in enumerate(items):
        row, col = divmod(i, ncols)
        lx, ly = x + col*w/ncols, y + row*h/max(nrows, 1)
        text_mm(fig, lx, ly, "■", color=color, fontsize=fontsize+1)
        text_mm(fig, lx+1.8, ly+.1, label, fontsize=fontsize)

def draw_strong(ax, frame, stages, *, title=None):
    styles = {5: "-", 10: "--", 15: ":"}
    for (group, k), g in frame.groupby(["group", "k"], sort=False):
        g = g.sort_values("x")
        prefix = group + " " if group in ("Brain", "Liver") else ""
        ax.plot(g.x, g.value, color=g.display_color.iloc[0], ls=styles[k], marker="o", ms=1.8, lw=.85, label=f"{prefix}Top {k}")
    stage_ticks(ax, stages)
    ax.set_ylim(-.025, 1.04)
    ax.set_ylabel("Strong-node contribution", fontsize=5.3, labelpad=1)
    if title:
        ax.set_title(title, fontsize=5.5, loc="left", pad=2)
    ax.legend(fontsize=5, frameon=False, ncol=2 if len(frame.group.unique()) > 1 else 1, labelspacing=.2, handlelength=1.5)
    polish(ax)

def draw_ccdf(ax, frame, metric):
    for (_, stage), g in frame[frame.metric.eq(metric)].groupby(["group", "stage"], sort=False):
        g = g.sort_values("x")
        ax.plot(g.x, g.value, color=g.display_color.iloc[0], ls=LINESTYLES[int(g.line_style.iloc[0])],
                lw=.7, marker="o" if len(g) < 3 else None, ms=1.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(log_tick_label))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    if metric.startswith("k_"):
        low, high = ax.get_xlim()
        ticks = [m*10.**e for e in range(-1, 7) for m in (1, 2, 5) if low <= m*10.**e <= high]
        if len(ticks) > 5:
            ticks = ticks[::2]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    else:
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
        ax.xaxis.set_major_formatter(FuncFormatter(log_tick_label))
    ax.set_xlabel(METRICS[metric], fontsize=5, labelpad=1)
    ax.set_ylabel("CCDF P(X ≥ x)", fontsize=5, labelpad=1)
    ax.tick_params(labelsize=5)
    polish(ax)

def stage_legend(fig, rect, stages, *, colors=None, ncols=4, fontsize=5):
    axis = axes_mm(fig, *rect)
    handles = [Line2D([], [], color="#596D7B", ls=LINESTYLES[i], lw=.85, label=STAGE_LABELS[s]) for i, s in enumerate(stages)]
    if colors:
        handles = [Line2D([], [], color=color, lw=1, label=label) for label, color in colors.items()] + handles
    axis.legend(handles=handles, loc="center", frameon=False, ncol=ncols, fontsize=fontsize,
                handlelength=1.8, columnspacing=1.1, labelspacing=.4)
    axis.set_axis_off()

def draw_distance(ax, frame, *, title=None, legend=True):
    for stage, g in frame.groupby("stage", sort=False):
        g = g.sort_values("x")
        point = g["plot"].iloc[0] == "distance_point"
        ax.plot(g.x, g.value, color=g.display_color.iloc[0],
                ls="none" if point else LINESTYLES[int(g.line_style.iloc[0])],
                lw=.85, marker="o" if point else None, ms=2, label=STAGE_LABELS[stage])
    ax.set_xlabel("Spatial distance", fontsize=5, labelpad=1)
    ax.set_ylabel("Communication strength", fontsize=5, labelpad=1)
    if title:
        ax.set_title(title, loc="left", fontsize=5.5, pad=2)
    if legend:
        ax.legend(fontsize=5, frameon=False, ncol=2, handlelength=1.8, labelspacing=.25)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useMathText=False)
    ax.tick_params(labelsize=5)
    polish(ax)
def draw_mouse_lr_counts(fig, rect, frame):
    for i, organ in enumerate(("Brain", "Liver")):
        ax = subaxis(fig, rect, .10 + i*.49, .25, .38, .58)
        for pair in MOUSE_PAIRS:
            g = frame[frame.organ.eq(organ) & frame.lr_pair.eq(pair)].sort_values("x")
            ax.plot(g.x, g.value, color=PAIR_COLORS[pair], lw=.85, marker="o", ms=1.8, label=pair.replace("_", "–").upper())
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(log_tick_label))
        ax.set_title(organ, loc="left", pad=2)
        ax.set_ylabel("Number of edges" if i == 0 else "", labelpad=1)
        stage_ticks(ax, MOUSE_STAGES)
        polish(ax, grid=True)
    handles, labels = ax.get_legend_handles_labels()
    legend_ax = subaxis(fig, rect, .08, .055, .92, .10)
    legend_ax.legend(handles, labels, fontsize=5, frameon=False, ncol=5,
                     loc="center", handlelength=1.1, handletextpad=.35,
                     columnspacing=.7, borderpad=0)
    legend_ax.set_axis_off()
def draw_mouse_celltype_strength(fig, rect, frame):
    group_width = rect[2]/2
    for i, group in enumerate(frame.group.unique()):
        sub = frame[frame.group.eq(group)]
        group_rect = (rect[0] + i*group_width, rect[1], group_width, rect[3])
        ax = subaxis(fig, group_rect, .17, 3.4/rect[3], .81,
                     (rect[3]-8.4)/rect[3])
        for cell_type in sub.cell_type.unique():
            g = sub[sub.cell_type.eq(cell_type)]
            label = cell_type.replace("Immature ", "Imm. ")
            label = label.replace("Monocytes & Monocyte-derived cells", "Monocytes")
            label = label.replace("Macrophages & Monocyte-derived cells", "Mono./macro.")
            label = label.replace("Endothelial cells", "Endothelial").replace("Kupffer cells", "Kupffer")
            ax.bar(g.x, g.value, bottom=g.y_bottom, color=g.display_color.iloc[0], width=.76, linewidth=.2, edgecolor="white", label=label)
        stages = list(sub.stage.unique())
        stage_ticks(ax, stages)
        ax.set_ylabel("Total strength (out + in)", fontsize=5, labelpad=1)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=False)
        peak = float(sub.groupby("stage")["value"].sum().max())
        ax.set_ylim(0, peak*1.6 if peak > 0 else 1)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.legend(fontsize=5, frameon=False, loc="upper left",
                  ncol=2, labelspacing=.05, handlelength=.65,
                  handletextpad=.3, borderpad=0, columnspacing=.6,
                  borderaxespad=.5)
        ax.set_title("Mouse " + group.lower(), loc="right", pad=1, fontsize=5.5)
        polish(ax)
def draw_mouse_interactions(fig, rect, frame):
    norm = Normalize(0, frame.value.max())
    for i, stage in enumerate(("12.5", "14.5")):
        ax = subaxis(fig, rect, .16+i*.46, .11, .30, .57)
        sub = frame[frame.stage.eq(stage)]
        tab = sub.pivot(index="row", columns="column", values="value")
        im = ax.imshow(tab, cmap=SPATIAL_CMAP, norm=norm, aspect="auto", interpolation="nearest")
        ax.set_yticks(range(len(tab)))
        ax.set_yticklabels(sub.groupby("row").sender.first(), fontsize=5)
        ax.set_xticks(range(len(tab.columns)))
        ax.set_xticklabels(sub.groupby("column").receiver.first(), fontsize=5, rotation=90)
        ax.tick_params(length=0, pad=1)
        ax.set_title(f"E{stage} · NTS–SORT1", fontsize=5, pad=2)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cax = subaxis(fig, rect, .94, .11, .014, .57)
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=5, pad=1)
    cb.set_label("log(1 + strength)", fontsize=5, labelpad=1)
    cb.outline.set_linewidth(.4)
def draw_axolotl_celltype_strength(fig, rect, frame):
    gap = 18  # Keep the external legend clear of the next panel's y-axis.
    group_width = (rect[2] - gap)/2
    for i, group in enumerate(frame.group.unique()):
        sub = frame[frame.group.eq(group)]
        group_rect = (rect[0] + i*(group_width + gap), rect[1], group_width, rect[3])
        ax = subaxis(fig, group_rect, .085, 3.4/rect[3], .55*4/3,
                     (rect[3]-8.4)/rect[3])
        for cell_type in sub.cell_type.unique():
            g = sub[sub.cell_type.eq(cell_type)]
            label = cell_type.replace("Immature ", "Imm. ")
            label = label.replace("Monocytes & Monocyte-derived cells", "Monocytes")
            label = label.replace("Macrophages & Monocyte-derived cells", "Mono./macro.")
            label = label.replace("Endothelial cells", "Endothelial").replace("Kupffer cells", "Kupffer")
            ax.bar(g.x, g.value, bottom=g.y_bottom, color=g.display_color.iloc[0], width=.76, linewidth=.2, edgecolor="white", label=label)
        stages = list(sub.stage.unique())
        stage_ticks(ax, stages)
        ax.set_ylabel("Total strength (out + in)", fontsize=5, labelpad=1)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=False)
        peak = float(sub.groupby("stage")["value"].sum().max())
        ax.set_ylim(0, peak*1.25 if peak > 0 else 1)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.legend(fontsize=5, frameon=False, loc="upper left",
                  bbox_to_anchor=(1.01, 1), ncol=1, labelspacing=.1,
                  handlelength=.65, handletextpad=.3, borderpad=0,
                  borderaxespad=.3)
        ax.set_title(group, loc="right", pad=1, fontsize=5.5)
        polish(ax)
def draw_axolotl_lr_counts(fig, rect, frame):
    for i, (group, stages) in enumerate((("Development", DEVELOPMENT_STAGES), ("Regeneration", REGENERATION_STAGES))):
        group_rect = (rect[0] + i*rect[2]/2, rect[1], rect[2]/2, rect[3])
        ax = subaxis(fig, group_rect, .21, .12, .73, .74)
        sub = frame[frame.group.eq(group)]
        for row in sub.itertuples():
            ax.plot([0, row.value], [row.y, row.y], color="#D9E1E7", lw=.7)
            ax.scatter(row.value, row.y, s=13, color=row.display_color, edgecolor="white", linewidth=.4, zorder=3)
            ax.annotate(f"{int(row.value)}", (row.value, row.y), xytext=(3, 0), textcoords="offset points", color=row.display_color, fontsize=5, va="center")
        ax.set_yticks(range(len(stages)))
        ax.set_yticklabels([STAGE_LABELS[s] for s in stages], fontsize=5)
        ax.set_ylim(len(stages)-.5, -.5)
        ax.set_xlim(0, frame.value.max()*1.19)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.set_title(group, fontsize=5.3, loc="left", pad=1)
        ax.tick_params(labelsize=5)
        ax.set_xlabel("Number of edges", fontsize=5, labelpad=1)
        polish(ax)
def draw_density(ax, frame, organ):
    sub = frame[frame.organ.eq(organ)]
    for pair in PAIR_COLORS:
        g = sub[sub.lr_pair.eq(pair)].sort_values("x")
        ax.plot(g.x, g.value, color=PAIR_COLORS[pair], lw=.8, marker="o", ms=1.5, label=pair.replace("_", "–").upper())
    stage_ticks(ax, MOUSE_STAGES)
    ax.set_title(organ, loc="left", pad=2)
    ax.set_ylabel("Graph density", labelpad=1)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useMathText=False)
    ax.legend(fontsize=5, frameon=False, handlelength=1.2, labelspacing=.15, loc="upper right")
    polish(ax)

def draw_mouse_topology(spatial, colors, density, strong, ccdf, distance):
    fig = plt.figure(figsize=(180/25.4, 170/25.4))
    draw_spatial_row(fig, (7, 6, 170, 28), spatial, MOUSE_STAGES,
                     "Mouse embryonic development", CORAL, common_scale=False)
    categorical_legend(fig, (7, 36, 170, 15), colors, ncols=6, fontsize=5)

    for x, organ in ((12, "Brain"), (70, "Liver")):
        draw_density(axes_mm(fig, x, 61, 43, 25), density, organ)
    draw_strong(axes_mm(fig, 133, 61, 44, 25), strong, MOUSE_STAGES)

    stage_legend(fig, (12, 95, 103, 8), MOUSE_STAGES,
                 colors={"Brain · NTS–SORT1": BLUE, "Liver · F2–F2R": CORAL}, ncols=5, fontsize=5)
    for i, metric in enumerate(METRICS):
        draw_ccdf(axes_mm(fig, 12+(i%2)*58, 109+(i//2)*32, 43, 22), ccdf, metric)
    for i, organ in enumerate(("Brain", "Liver")):
        draw_distance(axes_mm(fig, 135, 105+i*35, 42, 23), distance[distance.organ.eq(organ)], title=organ)
    return fig
def draw_axolotl_topology(spatial, colors, strong, ccdf, distance):
    fig = plt.figure(figsize=(180/25.4, 170/25.4))
    for y, height, stages, title, color in (
        (6, 22, DEVELOPMENT_STAGES, "Axolotl telencephalon development", BLUE),
        (35, 16, REGENERATION_STAGES, "Axolotl telencephalon regeneration", CORAL),
    ):
        draw_spatial_row(fig, (8, y, 114, height), spatial, stages, title, color)
    categorical_legend(fig, (7, 54, 117, 17), colors, ncols=5, fontsize=5)
    for group, stages, x in (("Development", DEVELOPMENT_STAGES, 12), ("Regeneration", REGENERATION_STAGES, 75)):
        draw_strong(axes_mm(fig, x, 81, 48, 21), strong[strong.group.eq(group)], stages, title=group)
    for i, group in enumerate(("Development", "Regeneration")):
        draw_distance(axes_mm(fig, 140, 11+i*55, 37, 36), distance[distance.group.eq(group)], title=group)
    for group, x, stages in (("Development", 2, DEVELOPMENT_STAGES), ("Regeneration", 90, REGENERATION_STAGES)):
        stage_legend(fig, (x+9, 109, 75, 8), stages, ncols=4, fontsize=5)
        sub = ccdf[ccdf.group.eq(group)]
        for i, metric in enumerate(METRICS):
            draw_ccdf(axes_mm(fig, x+10+(i%2)*44, 121+(i//2)*25, 31, 17), sub, metric)
    return fig

def _time_frame(series, name):
    table = series.tables[name].copy()
    mouse = set(series.samples.group) == {'Brain', 'Liver'}
    if mouse:
        mapping = {row.sample: row.stage[1:] if row.stage.startswith('E') else row.stage for row in series.samples.itertuples()}
    else:
        mapping = {row.sample: row.sample.replace('dev_meta', 'develop_Metamorphosed').replace('dev_', 'develop_').replace('reg_', 'regeneration_') for row in series.samples.itertuples()}
    table['stage'] = table['sample'].map(mapping)
    table['organ'] = table['group'] if mouse else 'telencephalon'
    if 'lr_pair' in table:
        table['lr_pair'] = table.lr_pair.str.replace(':', '_', regex=False)
    stages = MOUSE_STAGES if mouse else AXO_STAGES
    table['x'] = [((MOUSE_STAGES if mouse else DEVELOPMENT_STAGES if s in DEVELOPMENT_STAGES else REGENERATION_STAGES).index(s)) for s in table.stage]
    return table

def _time_panel(draw, size_mm, frame, output, name):
    w, h = size_mm
    with mpl.rc_context(RC):
        fig = plt.figure(figsize=((w + 4) / 25.4, (h + 5) / 25.4))
        draw(fig, (2, 2, w, h), frame)
        save_show(fig, output, name)
    frame.to_csv(Path(output) / (name + '_source.csv'), index=False)

def time_lr_counts(series, output):
    frame = _time_frame(series, 'lr').rename(columns={'edge_count': 'value'})
    if set(series.samples.group) == {'Brain', 'Liver'}:
        frame = frame[frame.value.gt(0)].copy()
        frame['display_color'] = frame.lr_pair.map(PAIR_COLORS)
        _time_panel(draw_mouse_lr_counts, (108, 52), frame, output, 'mouse_lr_counts')
    else:
        frame = frame[frame.lr_pair.eq('WNT7B_FZD5')].copy()
        frame['display_color'] = frame.group.map(PROCESS_COLORS)
        frame['y'] = frame.x
        _time_panel(draw_axolotl_lr_counts, (96, 43), frame, output, 'wnt7b_counts')

def time_celltypes(series, output):
    raw = _time_frame(series, 'celltypes')
    rows = []
    mouse = set(series.samples.group) == {'Brain', 'Liver'}
    for group in series.samples.group.unique():
        sub = raw[raw.group.eq(group)].copy()
        stages = MOUSE_STAGES if mouse else DEVELOPMENT_STAGES if group == 'Development' else REGENERATION_STAGES
        order = list(sub.groupby('cell_type').strength.sum().nlargest(10).index)
        sub['cell_type'] = sub.cell_type.where(sub.cell_type.isin(order), 'Other')
        if sub.cell_type.eq('Other').any(): order.append('Other')
        pivot = sub.groupby(['stage', 'cell_type']).strength.sum().unstack(fill_value=0).reindex(index=stages, columns=order, fill_value=0)
        for x, stage in enumerate(stages):
            bottom = 0.
            for celltype in order:
                value = float(pivot.loc[stage, celltype])
                rows.append(dict(group=group, stage=stage, cell_type=celltype, x=x, y_bottom=bottom, value=value))
                bottom += value
    frame = pd.DataFrame(rows)
    colors = cell_type_colors(raw.cell_type)
    colors['Other'] = '#B8B8B8'
    frame['display_color'] = frame.cell_type.map(colors)
    _time_panel(draw_mouse_celltype_strength if mouse else draw_axolotl_celltype_strength, (128, 50) if mouse else (154, 46), frame, output, 'celltype_strength')

def time_topology(series, data_root, output):
    mouse = set(series.samples.group) == {'Brain', 'Liver'}
    density = _time_frame(series, 'lr').rename(columns={'density': 'value'})
    strong = _time_frame(series, 'hubs').rename(columns={'fraction': 'value'})
    ccdf = _time_frame(series, 'ccdf').rename(columns={'survival': 'value'})
    ccdf['x'] = series.tables['ccdf']['x'].to_numpy()
    ccdf['metric'] = ccdf.metric.map({'out_degree':'k_out','in_degree':'k_in','out_strength':'s_out','in_strength':'s_in'})
    distance = _time_frame(series, 'distance').rename(columns={'score': 'value'})
    distance['x'] = distance.distance
    distance['plot'] = distance.kind.map({'cubic fit':'distance_curve', 'observed edge':'distance_point'})
    for frame in [density, strong, ccdf, distance]:
        frame['line_style'] = [str((MOUSE_STAGES if mouse else DEVELOPMENT_STAGES if s in DEVELOPMENT_STAGES else REGENERATION_STAGES).index(s)) for s in frame.stage]
        frame['display_color'] = frame.group.map(ORGAN_COLORS if mouse else PROCESS_COLORS)
    density['display_color'] = density.lr_pair.map(PAIR_COLORS)
    data_root = Path(data_root)
    if mouse:
        frames = [spatial_frame(data_root / 'raw' / f'E{s}_E1S1.MOSTA.h5ad', s, 'annotation') for s in MOUSE_STAGES]
    else:
        frames = [spatial_frame(data_root / 'h5ad' / f'{row.sample}.h5ad',
                  row.sample.replace('dev_meta','develop_Metamorphosed').replace('dev_','develop_').replace('reg_','regeneration_'), 'Annotation')
                  for row in series.samples.itertuples()]
    spatial = pd.concat(frames, ignore_index=True)
    colors = cell_type_colors(spatial.annotation)
    spatial['display_color'] = spatial.annotation.map(colors)
    with mpl.rc_context(RC):
        fig = draw_mouse_topology(spatial, colors, density, strong, ccdf, distance) if mouse else draw_axolotl_topology(spatial, colors, strong, ccdf, distance)
        save_show(fig, output, 'temporal_topology', exact_size=True)
    for name, frame in [('spatial', spatial), ('density', density), ('strong', strong), ('ccdf', ccdf), ('distance', distance)]:
        frame.to_csv(Path(output) / f'topology_{name}.csv', index=False)

def mouse_stage_interactions(run_root, output):
    from matplotlib.colors import Normalize, to_hex
    order = ['Blood', 'Immune', 'Gastrulation', 'Vascular', 'Neuron', 'Ectoderm', 'Fibroblast', 'Glioblast', 'Radial glia']
    frames = []
    for stage in ['12.5','14.5']:
        result = load_result(Path(run_root) / f'{stage}_Brain/out/out.h5ad', leaf_only=True)
        table = group_communication(result, lr_matrices(result)['Nts:Sort1'], 'cell_type').reindex(index=order, columns=order, fill_value=0)
        table = np.log1p(table)
        rows, cols = linear_sum_assignment(-table.to_numpy())
        pairs = sorted(zip(rows, cols), key=lambda pair: table.to_numpy()[pair], reverse=True)
        table = table.iloc[[r for r, _ in pairs], [c for _, c in pairs]]
        frames.append(pd.DataFrame(dict(stage=stage, row=np.repeat(np.arange(len(table)), len(table.columns)),
            column=np.tile(np.arange(len(table.columns)), len(table)), sender=np.repeat(table.index, len(table.columns)),
            receiver=np.tile(table.columns, len(table)), value=table.to_numpy().ravel())))
    frame = pd.concat(frames, ignore_index=True)
    norm = Normalize(0, frame.value.max())
    frame['display_color'] = [to_hex(SPATIAL_CMAP(norm(v))) for v in frame.value]
    _time_panel(draw_mouse_interactions, (116, 60), frame, output, 'nts_celltype_comparison')


def time_network_metrics(series, output):
    """Draw recomputed efficiency/path length and modularity in the main-figure style.

    Each process occupies two adjacent axes, and the processes share one row.
    The regeneration control is a horizontal reference, never a later time point.
    """
    frame = _time_frame(series, "network_metrics")
    mouse = set(series.samples.group) == {"Brain", "Liver"}
    if mouse:
        frame["group"] = "Mouse"
    groups = [("Mouse", MOUSE_STAGES)] if mouse else [
        ("Development", DEVELOPMENT_STAGES), ("Regeneration", REGENERATION_STAGES[:-1])]
    with mpl.rc_context(RC):
        fig = plt.figure(figsize=((124 * len(groups) + 4)/25.4, 49/25.4))
        for i, (group, stages) in enumerate(groups):
            rect = (2 + i*124, 2, 124, 44)
            left = subaxis(fig, rect, .11, .18, .32, .66)
            right = left.twinx()
            mod = subaxis(fig, rect, .67, .18, .31, .66)
            subset = frame[frame.group.eq(group)]
            for organ, values in subset[~subset.control].groupby("organ", sort=False):
                values = values.sort_values("x")
                style = "--" if organ == "Liver" else "-"
                mod_color = ORGAN_COLORS[organ] if mouse else TEAL
                for axis, metric, color in [(left, "global_efficiency", BLUE),
                                             (right, "average_shortest_path", CORAL),
                                             (mod, "modularity", mod_color)]:
                    axis.plot(values.x, values[metric], color=color, ls=style,
                              lw=.85, marker="o", ms=1.7, label=organ if mouse else group)
            for row in subset[subset.control].itertuples():
                for axis, metric, color in [(left, "global_efficiency", BLUE),
                                             (right, "average_shortest_path", CORAL),
                                             (mod, "modularity", FRAME)]:
                    axis.axhline(getattr(row, metric), color=color, ls="--", lw=.65, label="Control")
            title = "Mouse" if mouse else "Axolotl " + group.lower()
            for axis in (left, mod):
                axis.set_title(title, fontsize=5.4, loc="left", pad=2)
                stage_ticks(axis, stages)
                polish(axis)
                axis.tick_params(labelsize=5)
            for axis, color, label in [(left, BLUE, "Global efficiency"),
                                       (right, CORAL, "Average shortest path")]:
                axis.set_ylabel(label, fontsize=5, color=color, labelpad=1)
                axis.tick_params(axis="y", colors=color, labelsize=5, pad=1)
                axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=False)
                axis.yaxis.set_major_locator(MaxNLocator(4))
            right.spines["top"].set_visible(False)
            right.spines["left"].set_visible(False)
            mod.set_ylabel("Modularity", fontsize=5, labelpad=1)
            mod.yaxis.set_major_locator(MaxNLocator(4))
            if mouse:
                left.legend(fontsize=5, frameon=False, loc="upper right", labelspacing=.15, handlelength=1)
                mod.legend(fontsize=5, frameon=False, loc="lower right")
            elif group == "Regeneration":
                mod.legend(fontsize=5, frameon=False, loc="lower right")
        save_show(fig, output, "network_metrics")
    frame.to_csv(Path(output)/"network_metrics_source.csv", index=False)
