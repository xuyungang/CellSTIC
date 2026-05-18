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

from utils.time.strength_vs_distance import _find_cci_csv_for_organ_lr


_LOG_PREFIX = "[CellTypeStrengthOverStages]"


def _load_cell_type_mapping_from_h5ad(
    spatial_root: Path,
    stage: str,
    organ: str,
    cell_type_key: str = "cell_type",
    verbose: bool = True,
) -> Dict[str, str]:
    """
    Load cell_id -> cell_type mapping from the standard spatial h5ad:
        spatial_root/<stage>/preprocess/<organ>/preprocessed_RNA_filtered.h5ad
    """
    import scanpy as sc

    data_path = (
        spatial_root
        / str(stage)
        / "preprocess"
        / organ
        / "preprocessed_RNA_filtered.h5ad"
    )
    if not data_path.exists():
        if verbose:
            print(f"{_LOG_PREFIX} Skip {stage}/{organ}: h5ad not found at {data_path}")
        return {}

    adata = sc.read_h5ad(data_path)
    if cell_type_key not in adata.obs:
        if verbose:
            print(
                f"{_LOG_PREFIX} Warning: '{cell_type_key}' not in adata.obs for {stage}/{organ}; skipping.",
                flush=True,
            )
        return {}

    cell_types = adata.obs[cell_type_key].astype(str)
    mapping: Dict[str, str] = {
        str(cid): str(ct)
        for cid, ct in zip(adata.obs_names, cell_types.values)
        if str(ct) not in ("nan", "")
    }
    if verbose:
        print(
            f"{_LOG_PREFIX} Loaded cell types for {stage}/{organ} "
            f"(n_cells_with_type={len(mapping)})",
            flush=True,
        )
    return mapping


def _aggregate_cci_by_cell_type(
    csv_path: Path,
    cell_to_type: Dict[str, str],
    threshold: float = 0.0,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Aggregate a cell×cell CCI matrix to cell-type total strength:
        strength_total[cell_type] = s_out_ct + s_in_ct

    where:
        s_out_ct = sum of outgoing weights from cells of this type
        s_in_ct  = sum of incoming weights to cells of this type
    """
    try:
        mat = pd.read_csv(csv_path, index_col=0)
    except Exception as e:
        print(f"{_LOG_PREFIX} Warning: failed to read {csv_path}: {e}")
        return {}
    if mat.empty:
        return {}

    # Intersect cells with available type annotation
    rows = mat.index.astype(str)
    cols = mat.columns.astype(str)
    row_mask = np.array([r in cell_to_type for r in rows])
    col_mask = np.array([c in cell_to_type for c in cols])
    if not row_mask.any() or not col_mask.any():
        if verbose:
            print(
                f"{_LOG_PREFIX} No overlap between CCI cells and cell_type mapping for {csv_path}",
                flush=True,
            )
        return {}

    mat = mat.loc[rows[row_mask], cols[col_mask]]
    if mat.empty:
        return {}

    rows = mat.index.to_numpy(dtype=str)
    cols = mat.columns.to_numpy(dtype=str)
    values = mat.to_numpy(dtype=float)

    # Apply threshold
    values = np.where(values > threshold, values, 0.0)

    # Out-going strength per source cell
    out_per_cell = values.sum(axis=1)
    # In-going strength per target cell
    in_per_cell = values.sum(axis=0)

    strength_total: Dict[str, float] = {}
    # Accumulate by cell_type
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
    cci_root: Path,
    spatial_root: Path,
    stages: List[str],
    region_lr_map: Dict[str, str],
    threshold: float = 0.0,
    cell_type_key: str = "cell_type",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute total communication strength per cell type for each (stage, organ).

    Returns a tidy DataFrame with columns:
        organ, stage, cell_type, strength_total
    where strength_total = s_out_ct + s_in_ct for that cell type.
    """
    cci_root = Path(cci_root)
    spatial_root = Path(spatial_root)
    if not cci_root.exists():
        raise FileNotFoundError(f"CCI root directory not found: {cci_root}")

    rows: List[Dict[str, object]] = []

    for stage in stages:
        for organ, lr_pair in region_lr_map.items():
            csv_path = _find_cci_csv_for_organ_lr(
                cci_root=cci_root,
                stage=str(stage),
                organ=organ,
                lr_pair=lr_pair,
            )
            if csv_path is None:
                if verbose:
                    print(
                        f"{_LOG_PREFIX} Skip (missing CCI): {stage} / {organ} / {lr_pair}.csv",
                        flush=True,
                    )
                continue

            if verbose:
                print(
                    f"{_LOG_PREFIX} Aggregate CCI by cell type: {stage} / {organ} / {lr_pair}",
                    flush=True,
                )

            cell_to_type = _load_cell_type_mapping_from_h5ad(
                spatial_root=spatial_root,
                stage=str(stage),
                organ=organ,
                cell_type_key=cell_type_key,
                verbose=verbose,
            )
            if not cell_to_type:
                continue

            strength_total = _aggregate_cci_by_cell_type(
                csv_path=csv_path,
                cell_to_type=cell_to_type,
                threshold=threshold,
                verbose=verbose,
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
    cci_root: Path,
    spatial_root: Path,
    stages: List[str],
    region_lr_map: Dict[str, str],
    threshold: float,
    output_dir: Path,
    cell_type_key: str = "cell_type",
    top_n: int = 10,
    merge_celltypes_by_colon_prefix: bool = True,
    recompute: bool = False,
    verbose: bool = True,
    font_size: Optional[float] = None,
    fig_format: str = "png",
) -> bool:
    """
    End-to-end helper:

    - For each (stage, organ), read the CCI matrix and cell-type annotations.
    - Aggregate communication strength to cell-type level.
    - Cache the long-format table and re-plot from cache if possible.
    - Plot, for each organ, a stacked bar chart across stages.
    """
    cci_root = Path(cci_root)
    spatial_root = Path(spatial_root)
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
        cci_root=cci_root,
        spatial_root=spatial_root,
        stages=stages,
        region_lr_map=region_lr_map,
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
