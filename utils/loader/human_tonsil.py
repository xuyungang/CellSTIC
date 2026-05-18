"""
Human tonsil: load and preprocess RNA + ADT (no cell type annotation).
Call load_human_tonsil(raw_path, preprocess_path) to get (rna_adata, adt_adata).
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import anndata as ad
import pandas as pd
import scanpy as sc
import scipy.sparse as sparse
from scipy.spatial.distance import pdist, squareform

from utils.data import SpatialPreprocessorUtils

_RAW_NAMES = ("rna.h5ad", "adt.h5ad")
_CACHE_NAMES = ("preprocessed_RNA.h5ad", "preprocessed_ADT.h5ad")
_TSV_NAMES = ("rna/GSM6578062_humantonsil_RNA.tsv", "adt/GSM6578071_humantonsil_protein.tsv")


def _parse_coord(idx: str) -> Tuple[float, float]:
    x_str, y_str = idx.replace(" ", "").split("x")
    return float(x_str), float(y_str)


def _preprocess(rna_adata: ad.AnnData, adt_adata: ad.AnnData) -> Tuple[ad.AnnData, ad.AnnData]:
    # Handle infinite values
    if sparse.issparse(rna_adata.X):
        rna_adata.X.data[np.isinf(rna_adata.X.data)] = 0.0
    else:
        rna_adata.X[np.isinf(rna_adata.X)] = 0.0
    if sparse.issparse(adt_adata.X):
        adt_adata.X.data[np.isinf(adt_adata.X.data)] = 0.0
    else:
        adt_adata.X[np.isinf(adt_adata.X)] = 0.0
    
    rna_adata.raw = rna_adata.copy()
    rna_adata.var["mt"] = rna_adata.var_names.str.startswith("MT-")
    for a in [rna_adata, adt_adata]:
        sc.pp.calculate_qc_metrics(a, percent_top=None, log1p=False, inplace=True)
    n_components = 250
    sc.pp.filter_genes(rna_adata, min_cells=3)
    sc.pp.highly_variable_genes(rna_adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(rna_adata, target_sum=1e4)
    sc.pp.log1p(rna_adata)
    rna_adata.obsm["feat"] = SpatialPreprocessorUtils.pca(
        rna_adata[:, rna_adata.var["highly_variable"]], n_comps=n_components
    )
    adt_adata = SpatialPreprocessorUtils.clr_normalize_each_cell(adt_adata)
    adt_adata.obsm["feat"] = SpatialPreprocessorUtils.pca(adt_adata, n_comps=n_components)
    dist = squareform(pdist(rna_adata.obsm["spatial"], metric="euclidean"))
    for a in [rna_adata, adt_adata]:
        a.obsp["spatial_distances"] = sparse.csr_matrix(dist)
    return rna_adata, adt_adata


def load_human_tonsil(
    raw_path: Path,
    preprocess_path: Path = None,
    use_cache: bool = True,
) -> Tuple[ad.AnnData, ad.AnnData]:
    """
    Load and preprocess human tonsil RNA + ADT. No cell type annotation.

    raw_path: directory for original materials only (TSV files under rna/, adt/).
    preprocess_path: directory to load/save rna.h5ad, adt.h5ad (_RAW_NAMES) and preprocessed_*.h5ad (_CACHE_NAMES) (defaults to raw_path)
    use_cache: if True and cache exists under preprocess_path, load from cache; else preprocess and save.

    Returns (rna_adata, adt_adata) with obsm['feat'], obsp['spatial_distances'].
    """
    raw_path = Path(raw_path)
    if preprocess_path is None:
        preprocess_path = raw_path
    preprocess_path = Path(preprocess_path)

    if use_cache:
        cache_files = [preprocess_path / n for n in _CACHE_NAMES]
        if all(p.exists() for p in cache_files):
            return (
                sc.read_h5ad(cache_files[0]),
                sc.read_h5ad(cache_files[1]),
            )

    # Load or build RNA/ADT: prefer pre-built h5ad under preprocess_path; else build from raw_path (TSV)
    raw_rna = preprocess_path / _RAW_NAMES[0]
    raw_adt = preprocess_path / _RAW_NAMES[1]
    if raw_rna.exists() and raw_adt.exists():
        rna_adata = sc.read_h5ad(raw_rna)
        adt_adata = sc.read_h5ad(raw_adt)
    else:
        rna_df = pd.read_csv(raw_path / _TSV_NAMES[0], sep="\t", index_col=0)
        adt_df = pd.read_csv(raw_path / _TSV_NAMES[1], sep="\t", index_col=0)
        common = rna_df.index.intersection(adt_df.index)
        if len(common) == 0:
            raise ValueError("No common spots between RNA and ADT TSV files.")
        rna_df = rna_df.loc[common].astype(np.float32)
        adt_df = adt_df.loc[common].astype(np.float32)
        rna_adata = ad.AnnData(rna_df)
        adt_adata = ad.AnnData(adt_df)
        coords = np.array([_parse_coord(i) for i in common])
        rna_adata.obsm["spatial"] = adt_adata.obsm["spatial"] = coords
        preprocess_path.mkdir(parents=True, exist_ok=True)
        rna_adata.write_h5ad(raw_rna)
        adt_adata.write_h5ad(raw_adt)

    rna_adata.var_names_make_unique()
    adt_adata.var_names_make_unique()

    preprocess_path.mkdir(parents=True, exist_ok=True)
    rna_adata.write_h5ad(raw_rna)
    adt_adata.write_h5ad(raw_adt)

    rna_adata, adt_adata = _preprocess(rna_adata, adt_adata)

    rna_adata.write_h5ad(preprocess_path / _CACHE_NAMES[0])
    adt_adata.write_h5ad(preprocess_path / _CACHE_NAMES[1])

    return rna_adata, adt_adata
