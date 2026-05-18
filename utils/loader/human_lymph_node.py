"""
Human lymph node: load and preprocess RNA + ADT.
RNA built from raw 4 files: matrix, features, tissue_positions, isotype_norm_factors. ADT loaded from h5ad.
"""

import gzip
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io
import scipy.sparse as sparse
from scipy.spatial.distance import pdist, squareform

from utils.data import SpatialPreprocessorUtils
from .file_utils import get_ligand_receptor_map_with_edge_types

_RAW_ADT_H5AD = "adt/GSM8195498_A1_LN_Protein.h5ad"
_RAW_RNA_FEATURES = "rna/GSM8195494_A1LN_features.tsv.gz"
_RAW_RNA_MATRIX = "rna/GSM8195494_A1LN_matrix.mtx.gz"
_RAW_SPATIAL_POSITIONS = "spatial/GSM8195494_A1LN_tissue_positions.csv"
_RAW_ADT_ISOTYPE = "adt/GSM8195498_A1LN_isotype_normalization_factors.csv.gz"
_RNA_BUILT_H5AD = "rna/rna.h5ad"
_ADT_BUILT_H5AD = "adt/adt.h5ad"
_CACHE_NAMES = ("preprocessed_RNA.h5ad", "preprocessed_ADT.h5ad")
_LR_FILE = "l-r/LR.csv"
_CELL_TYPE_CSV = "type/cell_type_annotations_mapped.csv"


def _load_tissue_positions_table(raw_path: Path) -> pd.DataFrame:
    """Load tissue positions CSV; return DataFrame indexed by barcode with columns pxl_col_in_fullres, pxl_row_in_fullres."""
    tp_df = pd.read_csv(raw_path / _RAW_SPATIAL_POSITIONS)
    tp_df["barcode"] = tp_df["barcode"].astype(str)
    return tp_df.set_index("barcode")[["pxl_col_in_fullres", "pxl_row_in_fullres"]]


def _apply_tissue_positions_to_adata(adata: ad.AnnData, raw_path: Path) -> None:
    """Overwrite adata.obsm['spatial'] with positions from the tissue positions table (barcode -> x,y)."""
    tp_indexed = _load_tissue_positions_table(raw_path)
    spatial = np.array([
        tp_indexed.loc[bc].values if bc in tp_indexed.index else np.array([np.nan, np.nan])
        for bc in adata.obs_names
    ], dtype=np.float64)
    # Fallback for any missing: use 0,0 so we don't break distance computation
    np.nan_to_num(spatial, copy=False, nan=0.0)
    adata.obsm["spatial"] = spatial


def _build_rna_from_raw(raw_path: Path) -> ad.AnnData:
    """Construct RNA AnnData from 4 raw 10x files (matrix, features, tissue_positions, isotype_norm_factors)."""
    iso_df = pd.read_csv(raw_path / _RAW_ADT_ISOTYPE)
    iso_df = iso_df[iso_df["in_tissue"] == 1]
    barcodes = iso_df["barcode"].astype(str).tolist()

    mtx = scipy.io.mmread(str(raw_path / _RAW_RNA_MATRIX))
    X = mtx.T.tocsr() if sparse.issparse(mtx) else sparse.csr_matrix(mtx.T)

    with gzip.open(raw_path / _RAW_RNA_FEATURES, "rt") as f:
        lines = [line.strip().split("\t") for line in f]
    gene_ids, gene_names, feature_types = zip(*[(r[0], r[1], r[2]) for r in lines])
    gene_expr_mask = [t == "Gene Expression" for t in feature_types]
    var_names = [gene_names[i] for i in range(len(gene_names)) if gene_expr_mask[i]]
    var_gene_ids = [gene_ids[i] for i in range(len(gene_ids)) if gene_expr_mask[i]]
    X = X[:, np.where(gene_expr_mask)[0]]

    tp_indexed = _load_tissue_positions_table(raw_path)
    spatial = np.array([
        tp_indexed.loc[bc].values if bc in tp_indexed.index else np.array([0.0, 0.0])
        for bc in barcodes
    ], dtype=np.float64)

    adata = ad.AnnData(X=X, obs=pd.DataFrame(index=barcodes), var=pd.DataFrame(index=var_names))
    adata.var["gene_ids"] = var_gene_ids
    adata.obsm["spatial"] = spatial
    adata.obs["isotype_norm_factor"] = iso_df["normalization_factor"].values
    adata.var_names_make_unique()
    return adata


def _load_cell_types_from_csv(
    rna_adata: ad.AnnData,
    other_adatas: Optional[List[ad.AnnData]] = None,
    csv_path: Optional[Path] = None,
    cell_type_key: str = "cell_type",
    cell_id_col: str = "cell_id",
    cell_type_col: str = "predicted_labels",
) -> None:
    """
    Load cell types from CSV file and assign to adata objects.

    Args:
        rna_adata: RNA AnnData object
        other_adatas: Optional list of other AnnData objects to assign cell types to
        csv_path: Path to CSV file containing cell type annotations
        cell_type_key: Key to store cell types in obs (default: 'cell_type')
        cell_id_col: Column name for cell IDs in CSV (default: 'cell_id')
        cell_type_col: Column name for cell type labels in CSV (default: 'predicted_labels')
    """
    if csv_path is None:
        raise ValueError("csv_path must be provided")
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Cell type CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, comment="#")
    cell_id_to_cell_type = dict(
        zip(df[cell_id_col].astype(str), df[cell_type_col].astype(str))
    )
    cell_types = [
        cell_id_to_cell_type.get(str(cid), "Unknown") for cid in rna_adata.obs_names
    ]
    unique_cell_types = sorted(set(cell_types))
    rna_adata.obs[cell_type_key] = pd.Categorical(
        cell_types, categories=unique_cell_types
    )

    if other_adatas:
        all_categories = sorted(set(unique_cell_types) | {"Unknown"})
        for adata_i in other_adatas:
            cell_types_i = [
                cell_id_to_cell_type.get(str(cid), "Unknown") for cid in adata_i.obs_names
            ]
            adata_i.obs[cell_type_key] = pd.Categorical(
                cell_types_i, categories=all_categories
            )


def load_human_lymph_node(
    raw_path: Path,
    preprocess_path: Optional[Path] = None,
    use_cache: bool = True,
    cell_type_csv_path: Optional[Path] = None,
) -> Tuple[ad.AnnData, ad.AnnData, Optional[Dict[str, List[str]]], Optional[Dict[str, int]]]:
    """
    Load and preprocess human lymph node RNA + ADT.

    raw_path: directory containing raw 4 files (RNA matrix/features/spatial/isotype), ADT h5ad, l-r/LR.csv.
    preprocess_path: directory to load/save preprocessed h5ad (_CACHE_NAMES) (defaults to raw_path).
    use_cache: if True and cache exists under preprocess_path, load from cache; else preprocess and save.
    cell_type_csv_path: optional path to CSV with cell type annotations (cell_id, predicted_labels).
        If None, no cell types are loaded. Default location: raw_path/type/cell_type_annotations.csv.

    Returns:
        (rna_adata, adt_adata, ligand_receptor_map, edge_type_map) with obsm['feat'], obsp['spatial_distances'].
        ligand_receptor_map and edge_type_map are from raw_path/l-r/LR.csv; (None, None) if file not found.
        When cell_type_csv_path is provided, obs['cell_type'] is populated for rna_adata and adt_adata.
    """
    raw_path = Path(raw_path)
    if preprocess_path is None:
        preprocess_path = raw_path
    preprocess_path = Path(preprocess_path)
    cache_files = [preprocess_path / n for n in _CACHE_NAMES]
    lr_path = raw_path / _LR_FILE

    ligand_receptor_map: Optional[Dict[str, List[str]]] = None
    if lr_path.exists():
        ligand_receptor_map, _ = get_ligand_receptor_map_with_edge_types(str(lr_path))

    if use_cache and all(p.exists() for p in cache_files):
        rna_adata = sc.read_h5ad(cache_files[0])
        adt_adata = sc.read_h5ad(cache_files[1])
        csv_path = cell_type_csv_path or raw_path / _CELL_TYPE_CSV
        if Path(csv_path).exists():
            _load_cell_types_from_csv(rna_adata, other_adatas=[adt_adata], csv_path=csv_path)
        return rna_adata, adt_adata, ligand_receptor_map

    # RNA: build from raw 4 files
    rna_adata = _build_rna_from_raw(raw_path)
    rna_adata.raw = rna_adata.copy()
    (raw_path / "rna").mkdir(parents=True, exist_ok=True)
    rna_adata.write_h5ad(raw_path / _RNA_BUILT_H5AD)

    adt_adata = sc.read_h5ad(raw_path / _RAW_ADT_H5AD)
    common = rna_adata.obs_names.intersection(adt_adata.obs_names)
    rna_adata = rna_adata[common].copy()
    adt_adata = adt_adata[common].copy()

    _apply_tissue_positions_to_adata(adt_adata, raw_path)
    (raw_path / "adt").mkdir(parents=True, exist_ok=True)
    adt_adata.write_h5ad(raw_path / _ADT_BUILT_H5AD)

    rna_adata.var["mt"] = rna_adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(rna_adata, percent_top=None, log1p=False, inplace=True)
    sc.pp.calculate_qc_metrics(adt_adata, percent_top=None, log1p=False, inplace=True)
    sc.pp.filter_genes(rna_adata, min_cells=3)
    sc.pp.highly_variable_genes(rna_adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(rna_adata, target_sum=1e4)
    sc.pp.log1p(rna_adata)
    rna_adata.obsm["feat"] = SpatialPreprocessorUtils.pca(
        rna_adata[:, rna_adata.var["highly_variable"]], n_comps=500
    )
    adt_adata = SpatialPreprocessorUtils.clr_normalize_each_cell(adt_adata)
    adt_adata.obsm["feat"] = SpatialPreprocessorUtils.pca(adt_adata, n_comps=30)

    dist = squareform(pdist(rna_adata.obsm["spatial"], metric="euclidean"))
    rna_adata.obsp["spatial_distances"] = sparse.csr_matrix(dist)
    adt_adata.obsp["spatial_distances"] = sparse.csr_matrix(dist)

    csv_path = cell_type_csv_path or raw_path / _CELL_TYPE_CSV
    if Path(csv_path).exists():
        _load_cell_types_from_csv(rna_adata, other_adatas=[adt_adata], csv_path=csv_path)

    if use_cache:
        preprocess_path.mkdir(parents=True, exist_ok=True)
        rna_adata.write_h5ad(preprocess_path / _CACHE_NAMES[0])
        adt_adata.write_h5ad(preprocess_path / _CACHE_NAMES[1])
    return rna_adata, adt_adata, ligand_receptor_map
