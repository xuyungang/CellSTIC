"""
Stacked bar plots of cell-type communication strength across developmental stages.

For each (stage, organ) CCI matrix, we aggregate cell–cell communication strength
to cell-type level using cell-type annotations from the corresponding h5ad files,
then build stacked bar charts over stages for the top-N cell types per organ.

The layout and caching pattern mirror other time-series utilities in this package.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import numpy as np
import pandas as pd

from utils.time.stage_axis import stage_axis_from_present
from utils.viz.matplotlib_svg import savefig as savefig_vector


_LOG_PREFIX = "[CellTypeStrengthOverStages]"


def _cell_type_mapping_from_adata(adata, cell_type_key: str) -> Dict[str, str]:
    if cell_type_key not in adata.obs:
        return {}
    cell_types = adata.obs[cell_type_key].astype(str)
    return {
        str(cid): str(ct)
        for cid, ct in zip(adata.obs_names, cell_types.values)
        if str(ct) not in ("nan", "")
    }


def _aggregate_cci_matrix_by_cell_type(
    mat: pd.DataFrame,
    cell_to_type: Dict[str, str],
    threshold: float = 0.0,
    verbose: bool = True,
    source_label: str = "CCI matrix",
) -> Dict[str, float]:
    if mat.empty:
        return {}
    rows = mat.index.astype(str)
    cols = mat.columns.astype(str)
    row_mask = np.array([r in cell_to_type for r in rows])
    col_mask = np.array([c in cell_to_type for c in cols])
    if not row_mask.any() or not col_mask.any():
        if verbose:
            print(
                f"{_LOG_PREFIX} No overlap between CCI cells and cell_type mapping for {source_label}",
                flush=True,
            )
        return {}
    mat = mat.loc[rows[row_mask], cols[col_mask]]
    if mat.empty:
        return {}
    rows = mat.index.to_numpy(dtype=str)
    cols = mat.columns.to_numpy(dtype=str)
    values = mat.to_numpy(dtype=float)
    values = np.where(values > threshold, values, 0.0)
    out_per_cell = values.sum(axis=1)
    in_per_cell = values.sum(axis=0)
    strength_total: Dict[str, float] = {}
    for cid, s_out in zip(rows, out_per_cell):
        ct = cell_to_type.get(cid)
        if ct is None:
            continue
        strength_total[ct] = strength_total.get(ct, 0.0) + float(s_out)
    for cid, s_in in zip(cols, in_per_cell):
        ct = cell_to_type.get(cid)
        if ct is None:
            continue
        strength_total[ct] = strength_total.get(ct, 0.0) + float(s_in)
    return strength_total


def compute_celltype_strength_table_over_stages(
    stages: List[str],
    region_lr_map: Dict[str, str],
    cci_source,
    threshold: float = 0.0,
    cell_type_key: str = "cell_type",
    verbose: bool = True,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for stage in stages:
        for organ, lr_pair in region_lr_map.items():
            adata = cci_source.get_adata(str(stage), organ)
            if adata is None:
                continue
            cell_to_type = _cell_type_mapping_from_adata(adata, cell_type_key)
            mat = cci_source.load_cci_for_lr_pair(str(stage), organ, lr_pair)
            if mat is None or not cell_to_type:
                if verbose:
                    print(f"{_LOG_PREFIX} Skip: {stage} / {organ} / {lr_pair}", flush=True)
                continue
            if verbose:
                print(f"{_LOG_PREFIX} Aggregate: {stage} / {organ} / {lr_pair}", flush=True)
            strength_total = _aggregate_cci_matrix_by_cell_type(
                mat, cell_to_type, threshold=threshold, verbose=verbose,
                source_label=f"{stage}/{organ}/{lr_pair}",
            )
            if not strength_total:
                continue

            for ct, total in strength_total.items():
                if total <= 0:
                    continue
                rows.append(
                    {
                        "organ": organ,
                        "stage": str(stage),
                        "cell_type": str(ct),
                        "strength_total": float(total),
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["organ", "stage", "cell_type", "strength_total"])

    return pd.DataFrame(rows)


def _prepare_top_celltypes_for_organ(
    df: pd.DataFrame,
    organ: str,
    top_n: int,
    stage_order: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    For one organ, compute stacked values matrix for stages x cell_types.

    Returns:
        stages_sorted: np.ndarray of stage values
        cell_types_ordered: list of column names (top-N + 'Other' if applicable)
        values: array with shape (n_stages, n_cell_types)
    """
    df_o = df[df["organ"] == organ].copy()
    if df_o.empty:
        return np.array([]), [], np.zeros((0, 0), dtype=float)
    df_o["stage"] = df_o["stage"].astype(str)

    # Global top-N cell types over all stages
    totals = (
        df_o.groupby("cell_type")["strength_total"]
        .sum()
        .sort_values(ascending=False)
    )
    top_cts = totals.head(top_n).index.tolist()

    # Everything not in top_cts is grouped into "Other"
    df_o["cell_type_top"] = df_o["cell_type"].where(
        df_o["cell_type"].isin(top_cts),
        other="Other",
    )

    _, stages_sorted_list, _, _, _ = stage_axis_from_present(
        df_o["stage"].unique(),
        full_order=stage_order,
    )
    stages_sorted = np.array(stages_sorted_list, dtype=object)
    df_pivot = (
        df_o.groupby(["stage", "cell_type_top"])["strength_total"]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(index=stages_sorted, fill_value=0.0)
    )

    cols: List[str] = [ct for ct in top_cts if ct in df_pivot.columns]
    if "Other" in df_pivot.columns:
        cols.append("Other")
    df_pivot = df_pivot[cols]

    values = df_pivot.values.astype(float)
    return stages_sorted, cols, values


def _plot_celltype_strength_bars_per_organ(
    df: pd.DataFrame,
    output_dir: Path,
    top_n: int = 10,
    font_size: Optional[float] = None,
    fig_format: str = "png",
    stage_order: Optional[List[str]] = None,
) -> None:
    """
    Plot stacked bar charts of total communication strength per cell type across stages.

    One figure per organ; within each figure, bars = stages and stacks = cell types.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_ext = fig_format.lstrip(".")

    organs = sorted(df["organ"].dropna().unique().tolist())
    if not organs:
        return

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["font.size"] = 7 if font_size is None else font_size
    plt.rcParams["axes.linewidth"] = 0.5

    cmap = plt.get_cmap("tab20")

    for organ in organs:
        stages, cts, values = _prepare_top_celltypes_for_organ(
            df, organ, top_n, stage_order=stage_order
        )
        fig, ax = plt.subplots(figsize=(4.0, 3.0), dpi=300)
        tick_size = 7 if font_size is None else max(font_size - 1, 1)
        legend_size = 6 if font_size is None else max(font_size - 1, 1)
        # White figure background, transparent axes background (no panel fill)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("none")

        if values.size == 0:
            ax.axis("off")
            out_path = output_dir / f"celltype_strength_over_stages_{organ}.{fig_ext}"
            savefig_vector(fig, out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            continue

        x = np.arange(len(stages), dtype=float)
        bottom = np.zeros_like(x, dtype=float)

        handles: List = []
        labels: List[str] = []

        for i, ct in enumerate(cts):
            height = values[:, i]
            if not np.any(height > 0):
                continue
            color = cmap(i % cmap.N)
            bars = ax.bar(
                x,
                height,
                bottom=bottom,
                width=0.8,
                label=ct,
                color=color,
                edgecolor="black",
                linewidth=0.2,
            )
            bottom = bottom + height
            handles.append(bars[0])
            labels.append(ct)

        _, _, _, tick_labels, xlab = stage_axis_from_present(stages.tolist(), full_order=stage_order)
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.set_xlabel(xlab)
        ax.set_ylabel("Total communication strength (out + in)")

        # Ensure visible, clear axes (no background fill, but axis lines on)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.spines["left"].set_color("black")
        ax.spines["bottom"].set_color("black")
        ax.tick_params(axis="both", labelsize=tick_size)

        if handles:
            # Legend inside axes, top-left corner (single column).
            ax.legend(
                handles,
                labels,
                loc="upper left",
                fontsize=legend_size,
                frameon=False,
                borderaxespad=0.2,
                ncol=1,
            )

        fig.tight_layout()
        out_path = output_dir / f"celltype_strength_over_stages_{organ}.{fig_ext}"
        savefig_vector(fig, out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"{_LOG_PREFIX} Figure saved for {organ} -> {out_path}")


def _load_celltype_strength_from_cache(
    data_path: Path,
    meta_path: Path,
    stages: List[str],
    region_lr_map: Dict[str, str],
    threshold: float,
    cell_type_key: str,
    top_n: int,
    merge_celltypes_by_colon_prefix: bool,
) -> Optional[pd.DataFrame]:
    """Try to load cell-type strength table from cache."""
    if not data_path.exists() or not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        if (
            meta.get("schema_version") == 2
            and meta.get("stages") == stages
            and meta.get("region_lr_map") == region_lr_map
            and meta.get("threshold") == threshold
            and meta.get("cell_type_key") == cell_type_key
            and meta.get("top_n") == top_n
            and meta.get("merge_celltypes_by_colon_prefix") == merge_celltypes_by_colon_prefix
        ):
            return pd.read_csv(data_path)
    except Exception as e:
        print(f"{_LOG_PREFIX} Failed to load cache from {meta_path}: {e}; recomputing.")
    return None


def _merge_celltypes_by_colon_prefix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge cell types like "A: B" into "A" (prefix before the first ':').

    This is useful when the annotation encodes a coarse type plus subtype
    separated by a colon, and we want to aggregate strengths at the coarse level.
    """
    if df.empty or "cell_type" not in df.columns:
        return df
    df_m = df.copy()
    s = df_m["cell_type"].astype(str)
    prefix = s.str.split(":", n=1).str[0].str.strip()
    merged = np.where(s.str.contains(":"), prefix, s.str.strip())
    df_m["cell_type"] = merged
    # Re-aggregate after merging (keep the same tidy schema)
    df_m = (
        df_m.groupby(["organ", "stage", "cell_type"], as_index=False)["strength_total"]
        .sum()
    )
    return df_m


def compute_save_and_plot_celltype_strength_over_stages(
    stages: List[str],
    region_lr_map: Dict[str, str],
    threshold: float,
    output_dir: Path,
    *,
    cci_source,
    cell_type_key: str = "cell_type",
    top_n: int = 10,
    merge_celltypes_by_colon_prefix: bool = True,
    recompute: bool = False,
    verbose: bool = True,
    font_size: Optional[float] = None,
    fig_format: str = "png",
) -> bool:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = output_dir / "celltype_strength_over_stages_data.csv"
    meta_path = output_dir / "celltype_strength_over_stages_meta.json"

    if not recompute:
        df_cached = _load_celltype_strength_from_cache(
            data_path=data_path,
            meta_path=meta_path,
            stages=stages,
            region_lr_map=region_lr_map,
            threshold=threshold,
            cell_type_key=cell_type_key,
            top_n=top_n,
            merge_celltypes_by_colon_prefix=merge_celltypes_by_colon_prefix,
        )
        if df_cached is not None and not df_cached.empty:
            if merge_celltypes_by_colon_prefix:
                df_cached = _merge_celltypes_by_colon_prefix(df_cached)
            _plot_celltype_strength_bars_per_organ(
                df_cached,
                output_dir=output_dir,
                top_n=top_n,
                font_size=font_size,
                fig_format=fig_format,
                stage_order=stages,
            )
            print(f"{_LOG_PREFIX} Plotted cell-type strength bars from cached data.")
            return True

    df = compute_celltype_strength_table_over_stages(
        stages=stages,
        region_lr_map=region_lr_map,
        cci_source=cci_source,
        threshold=threshold,
        cell_type_key=cell_type_key,
        verbose=verbose,
    )
    if df.empty:
        print(f"{_LOG_PREFIX} No cell-type strength data computed; skipping plot.")
        return False

    if merge_celltypes_by_colon_prefix:
        df = _merge_celltypes_by_colon_prefix(df)

    df.to_csv(data_path, index=False)
    meta = {
        "schema_version": 2,
        "stages": stages,
        "region_lr_map": region_lr_map,
        "threshold": threshold,
        "cell_type_key": cell_type_key,
        "top_n": top_n,
        "merge_celltypes_by_colon_prefix": merge_celltypes_by_colon_prefix,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    _plot_celltype_strength_bars_per_organ(
        df,
        output_dir=output_dir,
        top_n=top_n,
        font_size=font_size,
        fig_format=fig_format,
        stage_order=stages,
    )
    print(f"{_LOG_PREFIX} Cell-type strength figures saved under {output_dir}.")
    return True
