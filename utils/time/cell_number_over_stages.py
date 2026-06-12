"""
Cell number over stages: count unique cells per stage per organ and plot.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.time import filter as time_filter
from utils.viz.matplotlib_svg import savefig as savefig_vector

_DATA_SUFFIX = "_data.csv"
_META_SUFFIX = "_meta.json"
_LOG_PREFIX = "[TimeSeriesMetrics]"


def _ensure_output_dir(output_dir: Union[str, Path]) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out

_NATURE_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
}


def _format_stage_labels(stages: List[str]) -> List[str]:
    return [f"E{s}" if not str(s).upper().startswith("E") else str(s) for s in stages]


def _plot_nature_line_chart_multi_series(
    ordered_stages: List[str],
    series: Dict[str, List[int]],
    output_path: Path,
    ylabel: str,
    log_scale: bool = False,
    font_size: Optional[float] = None,
) -> None:
    """Draw Nature-style line chart with multiple series (one per organ) and save."""
    from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter, AutoMinorLocator
    from utils.metrics.palette_utils import get_custom_palette

    style = dict(_NATURE_STYLE)
    style.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 1.0,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
    })
    if font_size is not None:
        style.update({
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "xtick.labelsize": max(font_size - 1, 1),
            "ytick.labelsize": max(font_size - 1, 1),
            "legend.fontsize": max(font_size - 1, 1),
        })

        with plt.rc_context(style):
            fig, ax = plt.subplots(
                figsize=(4.5, 3.0),
                dpi=600,
                facecolor="white",
            )
            ax.set_facecolor("white")

            x = np.arange(len(ordered_stages))
            colors = get_custom_palette(len(series))

            if log_scale and "(log10)" not in ylabel:
                ylabel = f"{ylabel} (log10)"

            all_y = []
            for (organ, counts), color in zip(series.items(), colors):
                y = np.array(counts, dtype=float)
                if log_scale:
                    y = np.maximum(y, 1)

                all_y.append(y)

                ax.plot(
                    x,
                    y,
                    color=color,
                    linewidth=1.8,
                    marker="o",
                    markersize=4.5,
                    markerfacecolor=color,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                    label=organ,
                    zorder=3,
                )

            ax.set_xlabel("Embryonic day", labelpad=5)
            ax.set_ylabel(ylabel, labelpad=5)

            ax.set_xticks(x)
            ax.set_xticklabels(
                _format_stage_labels(ordered_stages),
                rotation=45,
                ha="right",
                va="top",
                rotation_mode="anchor",
            )
            ax.set_xlim(-0.15, len(ordered_stages) - 0.85)

            if log_scale:
                ax.set_yscale("log")
                ax.yaxis.set_major_locator(LogLocator(base=10.0))
                ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
                ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
                ax.yaxis.set_minor_formatter(NullFormatter())

                if all_y:
                    y_all = np.concatenate(all_y)
                    y_min = max(1, np.nanmin(y_all) * 0.85)
                    y_max = np.nanmax(y_all) * 1.20
                    ax.set_ylim(y_min, y_max)
            else:
                ax.yaxis.set_minor_locator(AutoMinorLocator(2))

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(1.0)
            ax.spines["bottom"].set_linewidth(1.0)
            ax.spines["left"].set_color("#222222")
            ax.spines["bottom"].set_color("#222222")

            ax.tick_params(
                axis="both",
                which="major",
                direction="out",
                length=3.2,
                width=0.9,
                colors="#222222",
                pad=2.5,
            )
            ax.tick_params(
                axis="both",
                which="minor",
                direction="out",
                length=2.0,
                width=0.6,
                colors="#222222",
            )

            ax.grid(False)

            ax.legend(
                loc="upper left",
                frameon=False,
                handlelength=1.6,
                handletextpad=0.5,
                borderaxespad=0.2,
                markerscale=0.95,
            )

            fig.tight_layout(pad=0.6)
            savefig_vector(
                fig,
                output_path,
                dpi=600,
                bbox_inches="tight",
                pad_inches=0.02,
                facecolor="white",
                edgecolor="none",
            )
            plt.close(fig)

        print(f"{_LOG_PREFIX} Plot saved: {output_path}")


def _compute_cell_counts_by_organ(
    stages: List[str],
    cci_source: "CciSource",
    annotation_filter: Optional[List[str]] = None,
    lr_filter: Optional[List[str]] = None,
) -> Tuple[List[str], Dict[str, List[int]]]:
    all_organs: set = set()
    stage_organ_counts: Dict[str, Dict[str, int]] = {}
    for stage in stages:
        organ_counts: Dict[str, int] = {}
        for organ in cci_source.list_organs(stage, annotation_filter):
            if not cci_source.list_lr_stems(stage, organ, lr_filter):
                continue
            adata = cci_source.get_adata(stage, organ)
            if adata is None:
                continue
            organ_counts[organ] = int(adata.n_obs)
            all_organs.add(organ)
        if organ_counts:
            stage_organ_counts[stage] = organ_counts
            print(f"{_LOG_PREFIX} Stage {stage}: {organ_counts}")
    if not stage_organ_counts:
        return [], {}
    ordered_stages = [s for s in stages if s in stage_organ_counts]
    ordered_organs = sorted(all_organs)
    series = {
        o: [stage_organ_counts[s].get(o, 0) for s in ordered_stages]
        for o in ordered_organs
    }
    return ordered_stages, series


def count_cell_number_over_stages(
    stages: List[str],
    output_dir: Union[str, Path],
    *,
    cci_source,
    recompute: bool = True,
    annotation_filter: Optional[List[str]] = None,
    lr_filter: Optional[List[str]] = None,
    font_size: Optional[float] = None,
    fig_format: str = "png",
) -> None:
    output_dir = _ensure_output_dir(output_dir)
    fig_ext = fig_format.lstrip(".")
    fig_name = f"cell_number_over_stages.{fig_ext}"
    data_path = output_dir / f"cell_number_over_stages{_DATA_SUFFIX}"
    meta_path = output_dir / f"cell_number_over_stages{_META_SUFFIX}"

    if not recompute and data_path.exists() and meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            if (meta.get("stages") == stages and
                meta.get("annotation_filter") == annotation_filter and
                meta.get("lr_filter") == lr_filter):
                df = pd.read_csv(data_path)
                if "organ" not in df.columns:
                    raise ValueError("Cached CSV missing 'organ' column; recompute to refresh.")
                df["stage"] = df["stage"].astype(str)
                ordered_stages = [s for s in stages if s in df["stage"].values]
                series = {}
                for o in df["organ"].unique():
                    sub = df[df["organ"] == o].set_index("stage")
                    series[o] = [int(sub.loc[s, "count"]) if s in sub.index else 0 for s in ordered_stages]
                _plot_nature_line_chart_multi_series(
                    ordered_stages, series,
                    output_dir / fig_name,
                    ylabel="Number of cells",
                    log_scale=True,
                    font_size=font_size,
                )
                print(f"{_LOG_PREFIX} Loaded cell counts from cache.")
                return
        except Exception as e:
            print(f"{_LOG_PREFIX} Cache load failed: {e}; recomputing.")

    ordered_stages, series = _compute_cell_counts_by_organ(
        stages, cci_source, annotation_filter=annotation_filter, lr_filter=lr_filter,
    )
    if not ordered_stages:
        print(f"{_LOG_PREFIX} No cell counts computed; aborting plot.")
        return

    rows = [{"stage": s, "organ": o, "count": c} for o, cnts in series.items() for s, c in zip(ordered_stages, cnts)]
    pd.DataFrame(rows).to_csv(data_path, index=False)
    with open(meta_path, "w") as f:
        json.dump({"stages": stages, "annotation_filter": annotation_filter, "lr_filter": lr_filter}, f, indent=0)
    _plot_nature_line_chart_multi_series(
        ordered_stages, series,
        output_dir / fig_name,
        ylabel="Number of cells",
        log_scale=True,
        font_size=font_size,
    )
    print(f"{_LOG_PREFIX} Cell-number line plot saved to {output_dir / fig_name}")
