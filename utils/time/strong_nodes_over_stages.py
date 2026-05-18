"""
Strong-node contribution curves over developmental stages.

For each (stage, organ) CCI network (selected via region_lr_map),
we compute node strengths (sum of incident edge weights after thresholding),
and then, for K in {5, 10, 15}, compute:

    contribution_K = (sum of strengths of top-K nodes) / (sum of strengths of all nodes).

These contributions are then plotted as line charts over stages, with styling
similar to other time-series utilities in this package.
"""

from pathlib import Path
from typing import Dict, List, Optional

import json
import numpy as np
import pandas as pd

from utils.time.stage_axis import stage_axis_from_present
from utils.viz.matplotlib_svg import savefig as savefig_vector


_LOG_PREFIX = "[StrongNodesOverStages]"


def _find_cci_csv_for_organ_lr(
    cci_root: Path,
    stage: str,
    organ: str,
    lr_pair: str,
) -> Optional[Path]:
    """
    Resolve the CCI CSV path for (stage, organ, lr_pair), tolerant to '-' vs '_' in LR names.
    """
    stage_dir = Path(cci_root) / str(stage) / organ
    if not stage_dir.exists():
        return None

    candidates = [
        lr_pair,
        lr_pair.replace("-", "_"),
        lr_pair.replace("_", "-"),
    ]
    for stem in candidates:
        path = stage_dir / f"{stem}.csv"
        if path.exists():
            return path
    return None


def _compute_node_strengths(csv_path: Path, threshold: float) -> Optional[np.ndarray]:
    """
    Compute per-node strength from a CCI matrix:

        strength(node) = sum of outgoing + incoming weights (after thresholding).

    Returns:
        1D numpy array of strengths, or None if not computable.
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

    # Threshold
    values = np.where(values > threshold, values, 0.0)

    # Outgoing strength per source node (row index)
    s_out = pd.Series(values.sum(axis=1), index=df.index.astype(str))
    # Incoming strength per target node (column index)
    s_in = pd.Series(values.sum(axis=0), index=df.columns.astype(str))

    # Union of all node labels from rows and columns
    all_nodes = sorted(set(s_out.index) | set(s_in.index))
    strengths = []
    for node in all_nodes:
        total = float(s_out.get(node, 0.0) + s_in.get(node, 0.0))
        strengths.append(total)

    strengths_arr = np.asarray(strengths, dtype=float)
    if strengths_arr.size == 0 or not np.any(strengths_arr > 0):
        return None
    return strengths_arr


def _compute_strong_node_contributions(
    strengths: np.ndarray,
    ks: List[int],
) -> Dict[int, float]:
    """
    Given an array of node strengths, compute contribution of top-K nodes
    for each K in ks.
    """
    strengths = np.asarray(strengths, dtype=float)
    strengths = strengths[np.isfinite(strengths)]
    strengths = strengths[strengths > 0]
    if strengths.size == 0:
        return {k: float("nan") for k in ks}

    strengths_sorted = np.sort(strengths)[::-1]
    total = float(strengths_sorted.sum())
    if total <= 0:
        return {k: float("nan") for k in ks}

    contributions: Dict[int, float] = {}
    for k in ks:
        k_eff = min(k, strengths_sorted.size)
        top_sum = float(strengths_sorted[:k_eff].sum())
        contributions[k] = top_sum / total
    return contributions


def compute_strong_nodes_table(
    cci_root: Path,
    stages: List[str],
    region_lr_map: Dict[str, str],
    threshold: float = 0.0,
    ks: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Compute strong-node contribution table over stages.

    Returns a tidy DataFrame with columns:
        organ, stage, k, contribution
    where contribution is the fraction of total node strength contributed
    by the top-k strongest nodes.
    """
    cci_root = Path(cci_root)
    if not cci_root.exists():
        raise FileNotFoundError(f"CCI root directory not found: {cci_root}")

    if ks is None:
        ks = [5, 10, 15]
    ks_sorted = sorted(int(k) for k in ks)

    rows: List[Dict[str, object]] = []
    organs = list(region_lr_map.keys())

    for stage in stages:
        for organ in organs:
            lr_pair = region_lr_map.get(organ)
            if not lr_pair:
                continue
            csv_path = _find_cci_csv_for_organ_lr(
                cci_root=cci_root,
                stage=str(stage),
                organ=organ,
                lr_pair=lr_pair,
            )
            if csv_path is None:
                print(
                    f"{_LOG_PREFIX} Skip (missing CCI): {stage} / {organ} / {lr_pair}.csv",
                    flush=True,
                )
                continue

            strengths = _compute_node_strengths(csv_path, threshold=threshold)
            if strengths is None:
                continue

            contribs = _compute_strong_node_contributions(strengths, ks_sorted)
            for k_val, contrib in contribs.items():
                rows.append(
                    {
                        "organ": organ,
                        "stage": str(stage),
                        "k": int(k_val),
                        "contribution": float(contrib),
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["organ", "stage", "k", "contribution"])

    df = pd.DataFrame(rows)
    df["k"] = df["k"].astype(int)
    return df


def _plot_strong_node_curves(
    df: pd.DataFrame,
    out_path: Path,
    font_size: Optional[float] = None,
    stage_order: Optional[List[str]] = None,
) -> None:
    """
    Plot strong-node contribution curves in a single panel, with styling
    similar to the provided reference figure:

        - x-axis: embryonic day (E9.5, E10.5, ...)
        - y-axis: strong-edge contribution (0..1)
        - curves: Brain Top 5/10/15 and Liver Top 5/10/15
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if df.empty:
        return

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["font.size"] = 7 if font_size is None else font_size
    plt.rcParams["axes.linewidth"] = 0.6

    fig, ax = plt.subplots(figsize=(4.0, 2.8), dpi=300)
    tick_size = 7 if font_size is None else max(font_size - 1, 1)
    legend_size = 6 if font_size is None else max(font_size - 1, 1)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("none")

    # Styling: line styles and markers per (organ, k)
    organ_colors = {
        "Brain": "#2166ac",
        "Liver": "#b2182b",
    }
    organ_markers = {
        "Brain": "o",
        "Liver": "s",
    }
    auto_markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    k_linestyles = {
        5: "-",
        10: "--",
        15: ":",
    }

    df = df.copy()
    stage_to_x, _ordered, tick_positions, tick_labels, xlab = stage_axis_from_present(
        df["stage"].unique(),
        full_order=stage_order,
    )
    df["_stage_x"] = df["stage"].astype(str).map(stage_to_x)

    cmap = plt.get_cmap("tab10")
    present_organs = sorted({str(x) for x in df["organ"].dropna().unique().tolist()})
    auto_organs = [o for o in present_organs if o not in organ_colors]
    for i, organ in enumerate(auto_organs):
        organ_colors[organ] = cmap(i % cmap.N)
        organ_markers[organ] = auto_markers[i % len(auto_markers)]

    for organ in sorted(df["organ"].unique()):
        for k_val in sorted(df["k"].unique()):
            sub = df[(df["organ"] == organ) & (df["k"] == k_val)].copy()
            if sub.empty:
                continue
            sub = sub.sort_values("_stage_x")
            x = sub["_stage_x"].to_numpy(dtype=float)
            y = sub["contribution"].to_numpy(dtype=float)
            mask = np.isfinite(y)
            x, y = x[mask], y[mask]
            if x.size == 0:
                continue

            label = f"{organ} Top {k_val}"
            ax.plot(
                x,
                y,
                color=organ_colors.get(organ, "#333333"),
                marker=organ_markers.get(organ, "o"),
                markersize=3.0,
                linewidth=1.2,
                linestyle=k_linestyles.get(int(k_val), "-"),
                label=label,
            )

    ax.set_xlabel(xlab)
    ax.set_ylabel("Strong-edge contribution")

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    ax.set_ylim(0.0, 1.0)

    # Clean but visible axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.tick_params(axis="both", labelsize=tick_size, direction="out", length=3, width=0.6)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # Legend below the plot, in multiple columns, similar to the reference style.
        ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            fontsize=legend_size,
            frameon=False,
            ncol=3,
        )

    fig.tight_layout(rect=[0.0, 0.05, 1.0, 1.0])
    out_path = Path(out_path)
    savefig_vector(fig, out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _load_strong_nodes_from_cache(
    data_path: Path,
    meta_path: Path,
    stages: List[str],
    region_lr_map: Dict[str, str],
    threshold: float,
    ks: List[int],
) -> Optional[pd.DataFrame]:
    """Try to load strong-node contribution table from cache."""
    if not data_path.exists() or not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        if (
            meta.get("schema_version") == 1
            and meta.get("stages") == stages
            and meta.get("region_lr_map") == region_lr_map
            and meta.get("threshold") == threshold
            and meta.get("ks") == ks
        ):
            return pd.read_csv(data_path)
    except Exception as e:
        print(f"{_LOG_PREFIX} Failed to load cache from {meta_path}: {e}; recomputing.")
    return None


def compute_save_and_plot_strong_nodes_over_stages(
    cci_root: Path,
    stages: List[str],
    region_lr_map: Dict[str, str],
    threshold: float,
    output_dir: Path,
    ks: Optional[List[int]] = None,
    recompute: bool = False,
    verbose: bool = True,
    font_size: Optional[float] = None,
    fig_format: str = "png",
) -> bool:
    """
    End-to-end helper:

    - For each (stage, organ), read the CCI matrix selected via region_lr_map.
    - Compute node strengths and strong-node contributions for top-K nodes.
    - Cache the long-format table and re-plot from cache if possible.
    - Plot a single figure summarizing Brain/Liver Top 5/10/15 over stages.
    """
    cci_root = Path(cci_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if ks is None:
        ks = [5, 10, 15]
    ks_sorted = sorted(int(k) for k in ks)

    data_path = output_dir / "strong_nodes_over_stages_data.csv"
    meta_path = output_dir / "strong_nodes_over_stages_meta.json"
    fig_ext = fig_format.lstrip(".")
    fig_path = output_dir / f"strong_nodes_over_stages.{fig_ext}"

    if not recompute:
        df_cached = _load_strong_nodes_from_cache(
            data_path=data_path,
            meta_path=meta_path,
            stages=stages,
            region_lr_map=region_lr_map,
            threshold=threshold,
            ks=ks_sorted,
        )
        if df_cached is not None and not df_cached.empty:
            _plot_strong_node_curves(
                df_cached, out_path=fig_path, font_size=font_size, stage_order=stages
            )
            if verbose:
                print(f"{_LOG_PREFIX} Plotted strong-node curves from cached data.")
            return True

    df = compute_strong_nodes_table(
        cci_root=cci_root,
        stages=stages,
        region_lr_map=region_lr_map,
        threshold=threshold,
        ks=ks_sorted,
    )
    if df.empty:
        print(f"{_LOG_PREFIX} No strong-node data computed; skipping plot.")
        return False

    df.to_csv(data_path, index=False)
    meta = {
        "schema_version": 1,
        "stages": stages,
        "region_lr_map": region_lr_map,
        "threshold": threshold,
        "ks": ks_sorted,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    _plot_strong_node_curves(df, out_path=fig_path, font_size=font_size, stage_order=stages)
    if verbose:
        print(f"{_LOG_PREFIX} Strong-node curves figure saved to {fig_path}.")
    return True

