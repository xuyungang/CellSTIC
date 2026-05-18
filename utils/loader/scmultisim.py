"""
Scmultisim: load and preprocess RNA + ATAC from simulated CSVs.
Call load_scmultisim(raw_path) to get
    (rna_adata, atac_adata, true_labels, ligand_receptor_map, edge_type_map),
where rna_adata / atac_adata already contain:
    - obsm['feat']: modality features
    - obsm['spatial']: spatial coordinates
    - obsp['spatial_distances']: pairwise spatial distances
    - obs['cell_type']: integer cell type annotations (if meta/cell.csv exists).
"""

from pathlib import Path
from typing import Dict, List, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sparse
from scipy.spatial.distance import pdist, squareform

from utils.data import SpatialPreprocessorUtils
from .file_utils import load_scmultisim_true_labels
from .file_utils import get_ligand_receptor_map_with_edge_types


# File / directory names for scmultisim dataset
SCMULTISIM_CACHE_RNA = "preprocessed_rna.h5ad"
SCMULTISIM_CACHE_ATAC = "preprocessed_atac.h5ad"
SCMULTISIM_TRAN_COUNT = "rna/tran_count.csv"
SCMULTISIM_ATAC_COUNT = "atac/atac_count.csv"
SCMULTISIM_COORD = "spatial/coord.csv"
SCMULTISIM_LR_FILE = "l-r/LR.csv"
SCMULTISIM_LABEL_FILE = "gt/label.h5"
SCMULTISIM_CELL_META = "meta/cell.csv"


def _attach_cell_types_if_available(raw_path: Path, rna_adata: ad.AnnData, atac_adata: ad.AnnData) -> None:
    """
    Attach cell type annotations from meta/cell.csv to rna_adata / atac_adata if the file exists.
    Expects columns:
        - cell.type
        - cell.type.idx
    and index matching cell ids.
    """
    cell_meta_path = raw_path / SCMULTISIM_CELL_META
    if not cell_meta_path.exists():
        return

    cell_df = pd.read_csv(cell_meta_path, index_col=0)
    cell_df.index = cell_df.index.astype(str)

    # Align to existing cells (intersection to be safe)
    common_cells = rna_adata.obs_names.intersection(cell_df.index)
    if len(common_cells) == 0:
        return

    cell_df = cell_df.loc[common_cells]

    # Ensure ordering is consistent with adata.obs_names
    cell_df = cell_df.reindex(rna_adata.obs_names)

    # Use numeric cell.type.idx as the canonical 'cell_type' used for LR constraints
    if "cell.type.idx" in cell_df.columns:
        rna_adata.obs["cell_type"] = cell_df["cell.type.idx"].astype(int).values
        atac_adata.obs["cell_type"] = cell_df["cell.type.idx"].astype(int).values


def _load_lr_pair_type_constraints(raw_path: Path) -> Dict[str, List[Tuple[int, int]]]:
    """
    From l-r/LR.csv, build:
        pair_type_constraints: \"Ligand:Receptor\" -> list of (ct1, ct2) integer pairs.
    This is specific to the scmultisim dataset.
    """
    lr_path = raw_path / SCMULTISIM_LR_FILE
    if not lr_path.exists():
        return {}
    lr_df = pd.read_csv(lr_path, index_col=0)
    # Assume scmultisim-style columns: ligand, receptor, ct1, ct2
    if not {"ligand", "receptor", "ct1", "ct2"}.issubset(lr_df.columns):
        return {}
    pair_constraints: Dict[str, List[Tuple[int, int]]] = {}
    for _, row in lr_df.iterrows():
        ligand = str(row["ligand"]).strip()
        receptor = str(row["receptor"]).strip()
        ct1 = int(row["ct1"])
        ct2 = int(row["ct2"])
        key = f"{ligand}:{receptor}"
        pair_constraints.setdefault(key, []).append((ct1, ct2))
    return pair_constraints


def load_scmultisim(
    raw_path: Path,
    preprocess_path: Path,
    use_cache: bool = True,
    ) -> Tuple[
        ad.AnnData,
        ad.AnnData,
        np.ndarray,
        Dict[str, List[str]],
        Dict[str, int],
        Dict[str, List[Tuple[int, int]]],
    ]:
    """
    Load and preprocess scmultisim RNA + ATAC and true labels.

    Args:
        raw_path: directory with tran_count.csv, atac_count.csv, coord.csv and label.h5
                  (cell.csv optional, not used for annotation).
        preprocess_path: directory to save/load preprocessed_rna.h5ad, preprocessed_atac.h5ad.
        use_cache: load/save preprocessed_rna.h5ad, preprocessed_atac.h5ad under preprocess_path.

    Returns:
        (rna_adata, atac_adata, true_labels) with obsm['feat'], obsp['spatial_distances'].
    """
    raw_path = Path(raw_path)
    preprocess_path = Path(preprocess_path)
    cache_rna = preprocess_path / SCMULTISIM_CACHE_RNA
    cache_atac = preprocess_path / SCMULTISIM_CACHE_ATAC
    if use_cache and cache_rna.exists() and cache_atac.exists():
        rna_adata = ad.read_h5ad(cache_rna)
        atac_adata = ad.read_h5ad(cache_atac)
        # Even when using cache, (re-)attach cell type annotations if meta/cell.csv exists
        _attach_cell_types_if_available(raw_path, rna_adata, atac_adata)
        true_labels = load_scmultisim_true_labels(raw_path / SCMULTISIM_LABEL_FILE)
        ligand_receptor_map, edge_type_map = get_ligand_receptor_map_with_edge_types(raw_path / SCMULTISIM_LR_FILE)
        pair_type_constraints = _load_lr_pair_type_constraints(raw_path)
        return (
            rna_adata,
            atac_adata,
            true_labels,
            ligand_receptor_map,
            edge_type_map,
            pair_type_constraints,
        )

    tran_df = pd.read_csv(raw_path / SCMULTISIM_TRAN_COUNT, index_col=0)
    atac_df = pd.read_csv(raw_path / SCMULTISIM_ATAC_COUNT, index_col=0)
    coord_df = pd.read_csv(raw_path / SCMULTISIM_COORD, index_col=0)
    for df in [tran_df, atac_df, coord_df]:
        df.index = df.index.astype(str)
    common = coord_df.index.intersection(tran_df.columns).intersection(atac_df.columns)

    rna_adata = ad.AnnData(X=tran_df[common].T.values)
    rna_adata.var_names = tran_df.index.tolist()
    rna_adata.obs_names = common.tolist()
    atac_adata = ad.AnnData(X=atac_df[common].T.values)
    atac_adata.var_names = atac_df.index.tolist()
    atac_adata.obs_names = common.tolist()
    coords = coord_df.loc[common][["x", "y"]].values
    rna_adata.obsm["spatial"] = atac_adata.obsm["spatial"] = coords

    # Attach cell type annotations (only for scmultisim, if meta/cell.csv exists)
    _attach_cell_types_if_available(raw_path, rna_adata, atac_adata)

    rna_adata.var["mt"] = rna_adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(rna_adata, percent_top=None, log1p=False, inplace=True)
    sc.pp.calculate_qc_metrics(atac_adata, percent_top=None, log1p=False, inplace=True)
    sc.pp.filter_genes(rna_adata, min_cells=3)
    n_components = min(rna_adata.n_vars, atac_adata.n_vars)
    sc.pp.highly_variable_genes(rna_adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(rna_adata, target_sum=1e4)
    sc.pp.log1p(rna_adata)
    rna_adata.obsm["feat"] = SpatialPreprocessorUtils.pca(
        rna_adata[:, rna_adata.var["highly_variable"]], n_comps=n_components
    )
    SpatialPreprocessorUtils.lsi(atac_adata, use_highly_variable=False, n_components=n_components + 1)
    atac_adata.obsm["feat"] = atac_adata.obsm["X_lsi"].copy()
    dist = squareform(pdist(rna_adata.obsm["spatial"], metric="euclidean"))
    rna_adata.obsp["spatial_distances"] = sparse.csr_matrix(dist)
    atac_adata.obsp["spatial_distances"] = sparse.csr_matrix(dist)

    if use_cache:
        # Ensure cache directory exists (e.g. raw/preprocess)
        cache_rna.parent.mkdir(parents=True, exist_ok=True)
        rna_adata.write(cache_rna)
        atac_adata.write(cache_atac)

    true_labels = load_scmultisim_true_labels(raw_path / SCMULTISIM_LABEL_FILE)
    ligand_receptor_map, edge_type_map = get_ligand_receptor_map_with_edge_types(
        raw_path / SCMULTISIM_LR_FILE
    )
    pair_type_constraints = _load_lr_pair_type_constraints(raw_path)
    return (
        rna_adata,
        atac_adata,
        true_labels,
        ligand_receptor_map,
        edge_type_map,
        pair_type_constraints,
    )
