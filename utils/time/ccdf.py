"""
CCDF utilities for degree and strength of CCI networks.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import numpy as np
import pandas as pd

from utils.viz.matplotlib_svg import savefig as savefig_vector


def compute_degree_strength_chunked(
    csv_path: Path,
    threshold: float = 0.0,
    chunksize: int = 5000,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-node out-degree, in-degree, out-strength, in-strength from CCI matrix.
    Returns (k_out, k_in, s_out, s_in) as 1d arrays.
    """
    out_degree: Dict[str, float] = {}
    in_degree: Dict[str, float] = {}
    out_strength: Dict[str, float] = {}
    in_strength: Dict[str, float] = {}
    col_order: Optional[List[str]] = None

    reader = pd.read_csv(csv_path, index_col=0, chunksize=chunksize)
    chunk_idx = 0
    for chunk in reader:
        if col_order is None:
            col_order = list(chunk.columns)
            for c in col_order:
                in_degree[c] = 0.0
                in_strength[c] = 0.0
        for idx, row in chunk.iterrows():
            out_degree[idx] = float((row > threshold).sum())
            out_strength[idx] = float(row.sum())
        for c in chunk.columns:
            in_degree[c] += float((chunk[c] > threshold).sum())
            in_strength[c] += float(chunk[c].sum())
        chunk_idx += 1
        if verbose and chunk_idx % 50 == 0:
            print(f"  ... {csv_path.name}: chunk {chunk_idx}", flush=True)

    if col_order is None:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )

    row_order = list(out_degree.keys())
    k_out = np.array([out_degree[i] for i in row_order], dtype=np.float64)
    s_out = np.array([out_strength[i] for i in row_order], dtype=np.float64)
    k_in = np.array([in_degree[c] for c in col_order], dtype=np.float64)
    s_in = np.array([in_strength[c] for c in col_order], dtype=np.float64)
    return k_out, k_in, s_out, s_in


def ccdf_curve(values: np.ndarray, min_val: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return (x, y) for CCDF: y = P(X >= x). Exclude below min_val for log-scale."""
    v = np.asarray(values).ravel()
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.array([]), np.array([])
    n = len(v)
    x = np.unique(v)
    if min_val is not None:
        x = x[x >= min_val]
    if len(x) == 0:
        return np.array([]), np.array([])
    x = np.sort(x)
    y = np.array([np.sum(v >= xi) / n for xi in x])
    return x, y


def compute_ccdf_rows_for_networks(
    cci_root: Path,
    stages: List[str],
    organs: Optional[List[str]] = None,
    threshold: float = 0.0,
    verbose: bool = True,
    region_lr_map: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Load degree/strength and CCDF curves for each (organ, stage).
    cci_root/<stage>/<organ>/*.csv is assumed (same layout as time-series utilities).
    """
    cci_root = Path(cci_root)
    if not cci_root.exists():
        raise FileNotFoundError(f"CCI root directory not found: {cci_root}")

    # If a region->LR map is provided, derive organs from its keys.
    if region_lr_map is not None:
        organs = sorted(region_lr_map.keys())

    # Infer organs from directory structure if not explicitly provided
    if organs is None:
        organ_set = set()
        for stage in stages:
            stage_dir = cci_root / str(stage)
            if not stage_dir.exists():
                continue
            for d in stage_dir.iterdir():
                if d.is_dir():
                    organ_set.add(d.name)
        organs = sorted(organ_set)

    results: List[Dict[str, Any]] = []
    for stage in stages:
        for organ in organs:
            stage_dir = cci_root / str(stage) / organ
            if not stage_dir.exists():
                if verbose:
                    print(f"Skip (missing organ dir): {stage}/{organ}", flush=True)
                continue

            # Determine which LR CSV to use for this organ
            if region_lr_map is None:
                raise ValueError("region_lr_map must be provided for CCDF computation.")
            lr_pair = region_lr_map.get(organ)
            if not lr_pair:
                continue

            # Be robust to '-' vs '_' in LR names, e.g. F2-F2r vs F2_F2r
            candidate_names = [
                lr_pair,
                lr_pair.replace("-", "_"),
                lr_pair.replace("_", "-"),
            ]
            csv_paths = []
            for name in candidate_names:
                path = stage_dir / f"{name}.csv"
                if path.exists():
                    csv_paths = [path]
                    break
            if not csv_paths:
                if verbose:
                    print(f"Skip (missing LR CSV): {stage_dir}/{lr_pair}.csv", flush=True)
                continue

            for csv_path in csv_paths:
                if not csv_path.exists():
                    if verbose:
                        print(f"Skip (missing): {csv_path}", flush=True)
                    continue
                if verbose:
                    print(f"Load: {stage} / {organ} / {csv_path.stem} ...", flush=True)
                k_out, k_in, s_out, s_in = compute_degree_strength_chunked(
                    csv_path, threshold=threshold, verbose=verbose
                )
                eps = 1e-12
                metric_curves = {
                    "k_out": ccdf_curve(k_out, min_val=0.5),
                    "k_in": ccdf_curve(k_in, min_val=0.5),
                    "s_out": ccdf_curve(s_out, min_val=eps),
                    "s_in": ccdf_curve(s_in, min_val=eps),
                }
                for metric, (x, y) in metric_curves.items():
                    for xi, yi in zip(x, y):
                        results.append(
                            {
                                "organ": organ,
                                "stage": stage,
                                "metric": metric,
                                "x": xi,
                                "y": yi,
                            }
                        )
    return results


def load_ccdf_from_cache(
    data_path: Path,
    meta_path: Path,
    threshold: float,
    stages: List[str],
    region_lr_map: Optional[Dict[str, str]],
) -> Optional[pd.DataFrame]:
    """
    Try to load CCDF long-format data from cache.
    Returns a DataFrame if cache is valid, otherwise None.
    """
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
        print(f"[CCDF] Failed to load cache from {meta_path}: {e}; recomputing.")
    return None


def plot_ccdf_from_df(
    df: pd.DataFrame,
    out_path: Path,
    font_size: Optional[float] = None,
    stage_order: Optional[List[str]] = None,
) -> None:
    """Draw 2x2 CCDF panels in a compact, publication-ready style."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mm = 1 / 25.4
    fig_w, fig_h = 89 * mm * 2, 2 * (70 * mm)
    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h))
    axes = axes.flatten()
    tick_size = 6 if font_size is None else max(font_size - 1, 1)
    label_size = 7 if font_size is None else font_size
    legend_size = 5 if font_size is None else max(font_size - 2, 1)

    # White background, no panel facecolor
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("none")
        # Show main axes clearly
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

    panel_metrics = ["k_out", "k_in", "s_out", "s_in"]
    xlabels = [
        "Out-degree, $k^{\\mathrm{out}}$",
        "In-degree, $k^{\\mathrm{in}}$",
        "Out-strength, $s^{\\mathrm{out}}$",
        "In-strength, $s^{\\mathrm{in}}$",
    ]
    organ_colors = {"Brain": "#2166ac", "Liver": "#b2182b"}
    cmap = plt.get_cmap("tab10")
    present_organs = sorted({str(x) for x in df["organ"].dropna().unique().tolist()})
    auto_organs = [o for o in present_organs if o not in organ_colors]
    for i, organ in enumerate(auto_organs):
        organ_colors[organ] = cmap(i % cmap.N)
    # Assign distinct line styles per stage. For non-numeric stage labels (e.g. develop_44),
    # fall back to an ordered categorical mapping (prefer the caller-provided stage_order).
    base_styles = [
        "-",
        "--",
        "-.",
        ":",
        (0, (3, 1)),
        (0, (5, 1)),
        (0, (1, 1)),
        (0, (3, 1, 1, 1)),
        (0, (5, 1, 1, 1)),
    ]
    present_stages = sorted({str(s) for s in df["stage"].dropna().unique().tolist()})
    if stage_order:
        ordered = [str(s) for s in stage_order if str(s) in set(present_stages)]
        ordered.extend([s for s in present_stages if s not in set(ordered)])
    else:
        ordered = present_stages
    stage_ls_map = {s: base_styles[i % len(base_styles)] for i, s in enumerate(ordered)}

    for ax, metric, xlabel in zip(axes, panel_metrics, xlabels):
        ax.set_xscale("log")
        ax.set_yscale("log")

        sub = df[df["metric"] == metric]
        for (organ, stage), g in sub.groupby(["organ", "stage"], sort=False):
            xs = g["x"].values
            ys = np.clip(g["y"].values, 1e-6, 1.0)
            if xs.size == 0:
                continue
            color = organ_colors.get(organ, "#333333")
            ls = stage_ls_map.get(str(stage), "-")
            ax.plot(xs, ys, color=color, ls=ls, lw=1.2, label=f"{organ} {stage}")

        ax.set_xlabel(xlabel, fontsize=label_size)
        ax.set_ylabel("CCDF $P(X \\geq x)$", fontsize=label_size)
        ax.legend(fontsize=legend_size, frameon=False, ncol=2)

    plt.tight_layout()
    out_path = Path(out_path)
    savefig_vector(fig, out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_save_and_plot_ccdf(
    cci_root: Path,
    stages: List[str],
    region_lr_map: Optional[Dict[str, str]],
    threshold: float,
    output_dir: Path,
    recompute: bool = False,
    verbose: bool = True,
    font_size: Optional[float] = None,
    fig_format: str = "png",
) -> bool:
    """
    End-to-end helper: try cache, otherwise compute CCDF rows from raw CCI CSVs,
    save CSV and meta, and plot the figure.
    Returns True if a figure was created (from cache or fresh), False otherwise.
    """
    # Construct standard output paths
    output_dir = Path(output_dir)
    data_path = output_dir / "ccdf_degree_strength_data.csv"
    meta_path = output_dir / "ccdf_degree_strength_meta.json"
    fig_ext = fig_format.lstrip(".")
    fig_path = output_dir / f"ccdf_degree_strength.{fig_ext}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Try cache first (unless recompute=True)
    if not recompute:
        df_cached = load_ccdf_from_cache(
            data_path=data_path,
            meta_path=meta_path,
            threshold=threshold,
            stages=stages,
            region_lr_map=region_lr_map,
        )
        if df_cached is not None:
            plot_ccdf_from_df(df_cached, fig_path, font_size=font_size, stage_order=stages)
            print("[CCDF] CCDF plotted from cached data.")
            return True

    # Fallback to recomputing from raw CSVs
    rows = compute_ccdf_rows_for_networks(
        cci_root=cci_root,
        stages=stages,
        threshold=threshold,
        verbose=verbose,
        region_lr_map=region_lr_map,
    )
    if not rows:
        print("[CCDF] No CCDF data computed; skipping plot.")
        return False

    df = pd.DataFrame(rows)
    data_path = Path(data_path)
    meta_path = Path(meta_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_path, index=False)

    meta = {
        "threshold": threshold,
        "stages": stages,
        "region_lr_map": region_lr_map,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    plot_ccdf_from_df(df, fig_path, font_size=font_size, stage_order=stages)
    print(f"[CCDF] CCDF degree/strength figure saved to {fig_path}.")
    return True


