"""
Efficiency and modularity utilities for CCI networks over developmental stages.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import numpy as np
import pandas as pd

from utils.time.stage_axis import stage_axis_from_present
from utils.viz.matplotlib_svg import savefig as savefig_vector


def collect_edges_from_dataframe(
    df: pd.DataFrame,
    threshold: float = 0.0,
) -> List[Tuple[str, str, float]]:
    """Collect edge list from an in-memory CCI matrix."""
    values = df.to_numpy()
    mask = values > threshold
    if not mask.any():
        return []
    row_idx, col_idx = np.where(mask)
    src_labels = df.index.to_numpy()[row_idx].astype(str)
    dst_labels = df.columns.to_numpy()[col_idx].astype(str)
    weights = values[row_idx, col_idx].astype(float)
    return list(zip(src_labels, dst_labels, weights))


def _build_graphs(edge_list: List[Tuple[str, str, float]]):
    """Build directed and undirected graphs. Edge weight as strength; for path use 1/weight as distance."""
    import networkx as nx

    G_dir = nx.DiGraph()
    G_undir = nx.Graph()
    for a, b, w in edge_list:
        if w <= 0:
            continue
        dist = 1.0 / max(w, 1e-10)
        G_dir.add_edge(a, b, weight=dist)
        if G_undir.has_edge(a, b):
            G_undir[a][b]["weight"] += w
        else:
            G_undir.add_edge(a, b, weight=float(w))
    for u, v in G_undir.edges():
        G_undir[u][v]["distance"] = 1.0 / max(G_undir[u][v]["weight"], 1e-10)
    return G_dir, G_undir


def modularity_undir(G_undir) -> float:
    """Modularity Q using greedy modularity communities. Weight = connection strength."""
    import networkx as nx

    if G_undir.number_of_nodes() == 0 or G_undir.number_of_edges() == 0:
        return float("nan")
    try:
        communities = nx.community.greedy_modularity_communities(G_undir, weight="weight")
        return float(nx.community.modularity(G_undir, communities, weight="weight"))
    except Exception:
        return float("nan")


def run_efficiency_table(
    stages: List[str],
    region_lr_map: Dict[str, str],
    cci_source,
    threshold: float = 0.0,
    verbose: bool = True,
) -> pd.DataFrame:
    from scipy.spatial.distance import pdist

    rows: List[Dict[str, object]] = []
    for stage in stages:
        for region in region_lr_map:
            adata = cci_source.get_adata(stage, region)
            if adata is None or "spatial" not in adata.obsm_keys():
                if verbose:
                    print(f"Skip {stage}/{region}: missing AnnData or spatial coordinates", flush=True)
                rows.append(
                    {
                        "organ": region,
                        "stage": str(stage),
                        "global_efficiency": np.nan,
                        "average_shortest_path": np.nan,
                        "modularity": np.nan,
                    }
                )
                continue

            coords = adata.obsm.get("spatial")
            if coords is None or coords.shape[0] != adata.n_obs or adata.n_obs < 2:
                rows.append(
                    {
                        "organ": region,
                        "stage": str(stage),
                        "global_efficiency": np.nan,
                        "average_shortest_path": np.nan,
                        "modularity": np.nan,
                    }
                )
                continue

            if verbose:
                print(f"  Spatial metrics: {stage} / {region} (n_cells={adata.n_obs}) ...", flush=True)

            dists = pdist(coords, metric="euclidean")
            finite_mask = np.isfinite(dists) & (dists > 0)
            if not np.any(finite_mask):
                ge = 0.0
                asp = float("nan")
            else:
                d_valid = dists[finite_mask]
                ge = float(np.mean(1.0 / d_valid))
                asp = float(np.mean(d_valid))

            mod = np.nan
            lr = region_lr_map.get(region)
            if lr:
                df = cci_source.load_cci_for_lr_pair(stage, region, lr)
                if df is not None:
                    edge_list = collect_edges_from_dataframe(df, threshold=threshold)
                    if edge_list:
                        _, G_undir = _build_graphs(edge_list)
                        mod = modularity_undir(G_undir)

            rows.append(
                {
                    "organ": region,
                    "stage": str(stage),
                    "global_efficiency": ge,
                    "average_shortest_path": asp,
                    "modularity": mod,
                }
            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def load_efficiency_from_cache(
    data_path: Path,
    meta_path: Path,
    threshold: float,
    stages: List[str],
    region_lr_map: Dict[str, str],
) -> Optional[pd.DataFrame]:
    """Try to load efficiency table from cache. Return DataFrame if valid, else None."""
    if not data_path.exists() or not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        if (
            meta.get("threshold") == threshold
            and meta.get("stages") == stages
            and meta.get("region_lr_map") == region_lr_map
        ):
            return pd.read_csv(data_path)
    except Exception as e:
        print(f"[Efficiency] Failed to load cache from {meta_path}: {e}; recomputing.")
    return None


def plot_efficiency_lines(
    df: pd.DataFrame,
    out_path: Path,
    font_size: Optional[float] = None,
    stage_order: Optional[List[str]] = None,
) -> None:
    """Plot global efficiency, average shortest path, and modularity as line plots (1x3).

    Embryonic stages are numeric strings (e.g. 12.5) mapped to float x positions.
    Non-numeric stage labels (e.g. develop_44) use ordinal x positions; pass ``stage_order``
    to control left-to-right order (should match the analysis ``stages`` list).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mm = 1 / 25.4
    fig_w = 89 * mm * 2
    fig_h = 55 * mm
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    tick_size = 6 if font_size is None else max(font_size - 1, 1)
    label_size = 7 if font_size is None else font_size
    legend_size = 5 if font_size is None else max(font_size - 2, 1)
    for ax in axes:
        ax.set_facecolor("none")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_linewidth(0.6)
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_color("black")
        ax.spines["left"].set_color("black")
        ax.tick_params(
            axis="both",
            which="both",
            direction="out",
            labelsize=tick_size,
            width=0.5,
            length=3,
            color="black",
        )

    metrics = [
        ("global_efficiency", "Global efficiency"),
        ("average_shortest_path", "shortest length"),
        ("modularity", "Modularity"),
    ]
    organ_colors = {"Brain": "#2166ac", "Liver": "#b2182b"}
    organ_markers = {"Brain": "o", "Liver": "s"}
    auto_markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    df = df.copy()
    stage_to_x, _ordered, tick_positions, tick_labels, xlabel = stage_axis_from_present(
        df["stage"].unique(),
        full_order=stage_order,
    )
    df["stage_float"] = df["stage"].astype(str).map(stage_to_x)

    # Auto-assign colors/markers for unseen organs (e.g. telencephalon).
    cmap = plt.get_cmap("tab10")
    present_organs = [str(x) for x in df["organ"].dropna().unique().tolist()]
    auto_organs = [o for o in present_organs if o not in organ_colors]
    for i, organ in enumerate(sorted(auto_organs)):
        organ_colors[organ] = cmap(i % cmap.N)
        organ_markers[organ] = auto_markers[i % len(auto_markers)]

    for ax, (col, ylabel) in zip(axes, metrics):
        for organ in df["organ"].unique():
            sub = df[df["organ"] == organ].copy()
            sub = sub.sort_values("stage_float")
            x = sub["stage_float"].values
            y = sub[col].values
            mask = np.isfinite(y)
            if not np.any(mask):
                continue
            x, y = x[mask], y[mask]
            ax.plot(
                x,
                y,
                color=organ_colors.get(organ, "#333333"),
                marker=organ_markers.get(organ, "o"),
                markersize=3,
                linewidth=1.2,
                label=organ,
            )
        ax.set_xlabel(xlabel, fontsize=label_size)
        ax.set_ylabel(ylabel, fontsize=label_size)
        if ax.get_lines():
            ax.legend(fontsize=legend_size, frameon=False)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            tick_labels,
            rotation=45,
            ha="right",
            va="top",
            rotation_mode="anchor",
        )

    plt.tight_layout()
    out_path = Path(out_path)
    savefig_vector(fig, out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_save_and_plot_efficiency(
    stages: List[str],
    region_lr_map: Dict[str, str],
    threshold: float,
    output_dir: Path,
    *,
    cci_source,
    recompute: bool = False,
    verbose: bool = True,
    font_size: Optional[float] = None,
    fig_format: str = "png",
) -> bool:
    """
    End-to-end helper: compute global efficiency, average shortest path, and modularity
    per (organ, stage), with caching and plotting.
    """
    output_dir = Path(output_dir)
    data_path = output_dir / "efficiency_metrics_data.csv"
    meta_path = output_dir / "efficiency_metrics_meta.json"
    fig_ext = fig_format.lstrip(".")
    fig_path = output_dir / f"efficiency_metrics.{fig_ext}"

    output_dir.mkdir(parents=True, exist_ok=True)

    if not recompute:
        df_cached = load_efficiency_from_cache(
            data_path=data_path,
            meta_path=meta_path,
            threshold=threshold,
            stages=stages,
            region_lr_map=region_lr_map,
        )
        if df_cached is not None:
            plot_efficiency_lines(df_cached, fig_path, font_size=font_size, stage_order=stages)
            print("[Efficiency] Metrics plotted from cached data.")
            return True

    df = run_efficiency_table(
        stages=stages,
        region_lr_map=region_lr_map,
        cci_source=cci_source,
        threshold=threshold,
        verbose=verbose,
    )
    if df.empty:
        print("[Efficiency] No efficiency data computed; skipping plot.")
        return False

    df.to_csv(data_path, index=False)
    meta = {
        "threshold": threshold,
        "stages": stages,
        "region_lr_map": region_lr_map,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    plot_efficiency_lines(df, fig_path, font_size=font_size, stage_order=stages)
    print(f"[Efficiency] Metrics figure saved to {fig_path}.")
    return True

