"""
Graph density over stages: density curves per LR pair and organ.

For each (stage, organ, LR pair) CCI matrix, we compute the graph density
of the corresponding weighted directed graph (after thresholding), and then
plot density-vs-stage curves.

One figure per LR pair (annotation); within each figure, each organ is a line.
"""

from pathlib import Path
from typing import Dict, List, Optional

import json
import numpy as np
import pandas as pd

from utils.time import filter as time_filter
from utils.time.stage_axis import stage_axis_from_present
from utils.viz.matplotlib_svg import savefig as savefig_vector

_LOG_PREFIX = "[TimeSeriesDensity]"

_DATA_SUFFIX = "_data.csv"
_META_SUFFIX = "_meta.json"


def _ensure_output_dir(output_dir: Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _compute_density_for_csv(csv_path: Path, threshold: float) -> Optional[float]:
    """
    Compute graph density for a single CCI matrix.

    We treat the matrix as a directed graph with possible self-loops removed:
        density = E / (N * (N - 1))
    where E is the number of entries > threshold and N is the number of nodes.
    """
    try:
        df = pd.read_csv(csv_path, index_col=0, low_memory=False)
    except Exception as e:
        print(f"{_LOG_PREFIX} Warning: failed to read {csv_path}: {e}")
        return None
    if df.empty:
        return None

    try:
        values = df.to_numpy(dtype=np.float32, copy=False)
    except (TypeError, ValueError):
        values = df.to_numpy()

    n = values.shape[0]
    if n <= 1:
        return None

    # Remove diagonal (self-loops) before counting edges.
    np.fill_diagonal(values, 0.0)
    e = int(np.count_nonzero(values > threshold))
    possible = n * (n - 1)
    if possible == 0:
        return None
    return float(e) / float(possible)


def _compute_density_table(
    stages: List[str],
    raw_root: Path,
    threshold: float,
    annotation_filter: Optional[List[str]] = None,
    lr_filter: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute graph density per (stage, organ, lr_pair).

    Returns a tidy DataFrame with columns:
        stage, organ, lr_pair, density
    """
    raw_root = Path(raw_root)
    rows: List[Dict[str, object]] = []

    for i, stage in enumerate(stages, start=1):
        print(f"{_LOG_PREFIX} Processing stage {stage} ({i}/{len(stages)})...")
        stage_dir = raw_root / str(stage)
        if not stage_dir.exists() or not stage_dir.is_dir():
            continue

        for organ_dir in stage_dir.iterdir():
            if not organ_dir.is_dir():
                continue
            organ = organ_dir.name
            if not time_filter.annotation_pass(organ, annotation_filter):
                continue

            for csv_path in organ_dir.glob("*.csv"):
                if not time_filter.lr_match(csv_path.stem, lr_filter):
                    continue
                lr_pair = csv_path.stem
                density = _compute_density_for_csv(csv_path, threshold=threshold)
                if density is None:
                    continue
                rows.append(
                    {
                        "stage": str(stage),
                        "organ": organ,
                        "lr_pair": lr_pair,
                        "density": density,
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["stage", "organ", "lr_pair", "density"])

    return pd.DataFrame(rows)


def _plot_density_curves(
    df: pd.DataFrame,
    output_dir: Path,
    font_size: Optional[float] = None,
    fig_format: str = "png",
    stage_order: Optional[List[str]] = None,
) -> None:
    """
    One figure per annotation (here: organ), with density-vs-stage curves.

    Within each figure:
        - x-axis: stage
        - y-axis: graph density
        - one line per LR pair (lr_pair)
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_ext = fig_format.lstrip(".")

    if df.empty:
        return

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["font.size"] = 7 if font_size is None else font_size
    plt.rcParams["axes.linewidth"] = 0.5

    # Colormap for LR pairs
    cmap = plt.get_cmap("tab20")

    organs = sorted(df["organ"].dropna().unique().tolist())
    stage_to_x, _ord_st, tick_positions, tick_labels, xlab = stage_axis_from_present(
        df["stage"].unique(),
        full_order=stage_order,
    )

    for organ in organs:
        sub = df[df["organ"] == organ].copy()
        if sub.empty:
            continue
        sub = sub.copy()
        sub["_stage_x"] = sub["stage"].astype(str).map(stage_to_x)

        fig, ax = plt.subplots(figsize=(4.0, 3.0), dpi=300)
        tick_size = 7 if font_size is None else max(font_size - 1, 1)
        legend_size = 6 if font_size is None else max(font_size - 1, 1)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("none")

        lr_pairs = sorted(sub["lr_pair"].dropna().unique().tolist())
        for i, lr in enumerate(lr_pairs):
            g = sub[sub["lr_pair"] == lr].sort_values("_stage_x")
            x = g["_stage_x"].to_numpy(dtype=float)
            y = g["density"].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            if x.size == 0:
                continue

            ax.plot(
                x,
                y,
                marker="o",
                markersize=3.0,
                linewidth=1.2,
                color=cmap(i % cmap.N),
                label=lr,
            )

        ax.set_xlabel(xlab)
        ax.set_ylabel("Graph density")

        # Clean but visible axes
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.spines["left"].set_color("black")
        ax.spines["bottom"].set_color("black")
        ax.tick_params(axis="both", labelsize=tick_size)

        if ax.get_lines():
            ax.legend(
                loc="upper left",
                fontsize=legend_size,
                frameon=False,
                borderaxespad=0.2,
                ncol=1,
            )

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")

        fig.tight_layout()
        out_path = output_dir / f"density_over_stages_{organ}.{fig_ext}"
        savefig_vector(fig, out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"{_LOG_PREFIX} Density figure saved for {organ} -> {out_path}")


def _load_density_from_cache(
    data_path: Path,
    meta_path: Path,
    stages: List[str],
    threshold: float,
    annotation_filter: Optional[List[str]],
    lr_filter: Optional[List[str]],
) -> Optional[pd.DataFrame]:
    """Try to load density table from cache."""
    if not data_path.exists() or not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        if (
            meta.get("schema_version") == 1
            and meta.get("stages") == stages
            and meta.get("threshold") == threshold
            and meta.get("annotation_filter") == annotation_filter
            and meta.get("lr_filter") == lr_filter
        ):
            return pd.read_csv(data_path)
    except Exception as e:
        print(f"{_LOG_PREFIX} Failed to load cache from {meta_path}: {e}; recomputing.")
    return None


def compute_save_and_plot_density_over_stages(
    stages: List[str],
    raw_root: Path,
    output_dir: Path,
    threshold: float = 0.0,
    annotation_filter: Optional[List[str]] = None,
    lr_filter: Optional[List[str]] = None,
    recompute: bool = True,
    font_size: Optional[float] = None,
    fig_format: str = "png",
) -> bool:
    """
    High-level helper: compute and plot graph density curves over stages.

    - stages: list of embryonic stages (strings or numbers, e.g. ["9.5", "10.5", ...]).
    - raw_root: root directory containing CCI CSVs: raw_root/<stage>/<organ>/*.csv
    - Each figure corresponds to one LR pair (lr_filter defines which to include).
    - Within a figure, each organ is a line over stages.

    Caching:
        - Density values are cached to CSV + JSON meta in `output_dir`.
        - When recompute=False and the meta matches (stages, threshold, filters),
          plots will be regenerated from cached density values.
    """
    output_dir = _ensure_output_dir(output_dir)
    data_path = output_dir / f"density_over_stages{_DATA_SUFFIX}"
    meta_path = output_dir / f"density_over_stages{_META_SUFFIX}"

    # Try cache
    if not recompute:
        df_cached = _load_density_from_cache(
            data_path=data_path,
            meta_path=meta_path,
            stages=stages,
            threshold=threshold,
            annotation_filter=annotation_filter,
            lr_filter=lr_filter,
        )
        if df_cached is not None and not df_cached.empty:
            _plot_density_curves(
                df_cached,
                output_dir=output_dir,
                font_size=font_size,
                fig_format=fig_format,
                stage_order=stages,
            )
            print(f"{_LOG_PREFIX} Loaded density from cache.")
            return True

    # Recompute
    df = _compute_density_table(
        stages=stages,
        raw_root=raw_root,
        threshold=threshold,
        annotation_filter=annotation_filter,
        lr_filter=lr_filter,
    )
    if df.empty:
        print(f"{_LOG_PREFIX} No density data computed; aborting plot.")
        return False

    df.to_csv(data_path, index=False)
    meta = {
        "schema_version": 1,
        "stages": stages,
        "threshold": threshold,
        "annotation_filter": annotation_filter,
        "lr_filter": lr_filter,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    _plot_density_curves(
        df, output_dir=output_dir, font_size=font_size, fig_format=fig_format, stage_order=stages
    )
    print(f"{_LOG_PREFIX} Density curves saved under {output_dir}.")
    return True

