"""
Strength–distance curves for CCI networks over developmental stages.

For each organ, we compute strength-versus-spatial-distance curves for a
single ligand–receptor pair (taken from `region_lr_map[organ]`) across stages
and draw one figure per organ, with one line per stage.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import numpy as np
import pandas as pd

from utils.viz.matplotlib_svg import savefig as savefig_vector


_LOG_PREFIX = "[StrengthDistance]"


def _spatial_coords_from_adata(adata, verbose: bool = True) -> Optional[pd.DataFrame]:
    if "spatial" not in adata.obsm_keys():
        if verbose:
            print(f"{_LOG_PREFIX} Warning: missing 'spatial' in AnnData; skipping.")
        return None
    coords = adata.obsm.get("spatial")
    if coords is None or coords.shape[0] != adata.n_obs:
        if verbose:
            print(f"{_LOG_PREFIX} Warning: invalid 'spatial' coordinates; skipping.")
        return None
    df = pd.DataFrame(coords, index=adata.obs_names)
    if df.shape[1] >= 2:
        df = df.iloc[:, :2]
        df.columns = ["x", "y"]
    return df


def _collect_strength_distance_pairs(
    cci_matrix: pd.DataFrame,
    coords: pd.DataFrame,
    threshold: float,
    max_points: int = 5000,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect (distance, strength) pairs for a single (stage, organ, lr_pair).
    All edges with weight > threshold are converted to (distance, strength) points.
    To avoid excessive memory usage, points can be randomly subsampled
    down to at most `max_points`.
    Returns (distances, strengths) as 1D numpy arrays.
    """
    from scipy.spatial.distance import cdist

    mat = cci_matrix
    if mat.empty:
        return np.array([]), np.array([])

    src_ids = mat.index.astype(str)
    dst_ids = mat.columns.astype(str)
    cells_with_coords = coords.index.astype(str)
    src_mask = src_ids.isin(cells_with_coords)
    dst_mask = dst_ids.isin(cells_with_coords)
    if not src_mask.any() or not dst_mask.any():
        return np.array([]), np.array([])

    mat = mat.loc[src_ids[src_mask], dst_ids[dst_mask]]
    if mat.empty:
        return np.array([]), np.array([])

    src_ids = mat.index.to_numpy(dtype=str)
    dst_ids = mat.columns.to_numpy(dtype=str)

    src_coords = coords.reindex(src_ids).to_numpy(dtype=float)
    dst_coords = coords.reindex(dst_ids).to_numpy(dtype=float)

    # 计算所有 (src, dst) 之间的欧氏距离
    dists = cdist(src_coords, dst_coords, metric="euclidean")
    values = mat.to_numpy(dtype=float)

    mask = values > threshold
    if not np.any(mask):
        return np.array([]), np.array([])

    d_valid = dists[mask].astype(float)
    w_valid = values[mask].astype(float)

    # 下采样，避免点数过多
    n = d_valid.size
    if n > max_points:
        idx = np.random.choice(n, size=max_points, replace=False)
        d_valid = d_valid[idx]
        w_valid = w_valid[idx]

    return d_valid, w_valid


def _load_strength_distance_from_cache(
    data_path: Path,
    meta_path: Path,
    stages: List[str],
    region_lr_map: Dict[str, str],
    threshold: float,
) -> Optional[pd.DataFrame]:
    """Try to load long-format strength–distance data from cache."""
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
        ):
            return pd.read_csv(data_path)
    except Exception as e:
        print(f"{_LOG_PREFIX} Failed to load cache from {meta_path}: {e}; recomputing.")
    return None


def _plot_strength_distance_per_organ(
    df: pd.DataFrame,
    output_dir: Path,
    font_size: Optional[float] = None,
    fig_format: str = "png",
) -> None:
    """
    One figure per organ:

    - For each stage, fit a smooth polynomial curve strength(distance).
    - Only the fitted curves are drawn (no raw scatter points).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    organs = sorted(df["organ"].unique())
    if not organs:
        return

    fig_ext = fig_format.lstrip(".")

    # Simple color/linestyle maps
    organ_colors = {
        "Brain": "#2166ac",
        "Liver": "#b2182b",
    }
    cmap = plt.get_cmap("tab10")
    auto_organs = [o for o in organs if o not in organ_colors]
    for i, organ in enumerate(auto_organs):
        organ_colors[organ] = cmap(i % cmap.N)
    # For stages, use categorical mapping; we keep it simple
    unique_stages = sorted({str(s) for s in df["stage"].unique()})
    stage_linestyles = [
        "-",
        "--",
        "-.",
        (0, (3, 1)),
        (0, (5, 1)),
        (0, (1, 1)),
        (0, (3, 1, 1, 1)),
        (0, (5, 1, 1, 1)),
    ]
    stage_ls_map = {
        s: stage_linestyles[i % len(stage_linestyles)]
        for i, s in enumerate(unique_stages)
    }

    for organ in organs:
        sub_org = df[df["organ"] == organ].copy()
        if sub_org.empty:
            continue

        fig, ax = plt.subplots(figsize=(4.0, 3.0), dpi=300)
        tick_size = 7 if font_size is None else max(font_size - 1, 1)
        label_size = 8 if font_size is None else font_size
        legend_size = 6 if font_size is None else max(font_size - 1, 1)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("none")

        # Basic styling
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_linewidth(0.8)
            ax.spines[spine].set_color("black")
        ax.tick_params(
            axis="both",
            which="both",
            direction="out",
            labelsize=tick_size,
            width=0.6,
            length=3,
            color="black",
        )

        for stage, g in sub_org.groupby("stage", sort=False):
            g = g.sort_values("distance")
            x = g["distance"].to_numpy(dtype=float)
            y = g["strength"].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            if x.size == 0:
                continue

            color = organ_colors.get(organ, "#333333")
            ls = stage_ls_map.get(str(stage), "-")

            # Polynomial fit curve (cubic) – draw curve only.
            if x.size >= 4:
                try:
                    x_fit = np.linspace(x.min(), x.max(), 200)
                    coeffs = np.polyfit(x, y, deg=3)
                    y_fit = np.polyval(coeffs, x_fit)
                    ax.plot(
                        x_fit,
                        y_fit,
                        color=color,
                        linestyle=ls,
                        linewidth=1.4,
                        label=f"E{stage}",
                    )
                except Exception as e:
                    # If fitting fails, just skip this stage.
                    print(f"{_LOG_PREFIX} Polyfit failed for {organ}, stage {stage}: {e}")

        ax.set_xlabel("Spatial distance", fontsize=label_size)
        ax.set_ylabel("Communication strength", fontsize=label_size)
        if ax.get_lines():
            ax.legend(fontsize=legend_size, frameon=False)

        plt.tight_layout()
        out_path = output_dir / f"strength_vs_distance_{organ}.{fig_ext}"
        savefig_vector(fig, out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"{_LOG_PREFIX} Figure saved for {organ} -> {out_path}")


def compute_save_and_plot_strength_vs_distance(
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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = output_dir / "strength_vs_distance_data.csv"
    meta_path = output_dir / "strength_vs_distance_meta.json"

    if not recompute:
        df_cached = _load_strength_distance_from_cache(
            data_path=data_path,
            meta_path=meta_path,
            stages=stages,
            region_lr_map=region_lr_map,
            threshold=threshold,
        )
        if df_cached is not None and not df_cached.empty:
            _plot_strength_distance_per_organ(
                df_cached, output_dir, font_size=font_size, fig_format=fig_format
            )
            print(f"{_LOG_PREFIX} Plotted strength–distance curves from cached data.")
            return True

    rows = []
    for stage in stages:
        for organ, lr_pair in region_lr_map.items():
            adata = cci_source.get_adata(str(stage), organ)
            if adata is None:
                continue
            coords = _spatial_coords_from_adata(adata, verbose=verbose)
            mat = cci_source.load_cci_for_lr_pair(str(stage), organ, lr_pair)
            if mat is None or coords is None:
                if verbose:
                    print(f"{_LOG_PREFIX} Skip: {stage}/{organ}/{lr_pair}")
                continue

            d_valid, w_valid = _collect_strength_distance_pairs(
                mat, coords, threshold=threshold, max_points=5000, verbose=verbose,
            )
            if d_valid.size == 0:
                continue

            for d, w in zip(d_valid, w_valid):
                if not (np.isfinite(d) and np.isfinite(w)):
                    continue
                rows.append(
                    {
                        "organ": organ,
                        "stage": str(stage),
                        "lr_pair": lr_pair,
                        "distance": float(d),
                        "strength": float(w),
                    }
                )

    if not rows:
        print(f"{_LOG_PREFIX} No strength–distance data computed; skipping plot.")
        return False

    df = pd.DataFrame(rows)
    df.to_csv(data_path, index=False)
    meta = {
        "schema_version": 1,
        "stages": stages,
        "region_lr_map": region_lr_map,
        "threshold": threshold,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    _plot_strength_distance_per_organ(df, output_dir, font_size=font_size, fig_format=fig_format)
    print(f"{_LOG_PREFIX} Strength–distance figures saved under {output_dir}.")
    return True

