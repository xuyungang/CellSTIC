"""
NSF: load and preprocess RNA + ADT + ATAC (no cell type / true label annotation).
Call load_nsf(raw_path, preprocess_path) to get (rna_adata, adt_adata, atac_adata, true_labels).
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import anndata as ad
import scanpy as sc
import scipy.sparse as sparse
from scipy.spatial.distance import pdist, squareform

from utils.data import SpatialPreprocessorUtils

_RAW_NAMES = ("rna/adata_RNA.h5ad", "adt/adata_ADT.h5ad", "atac/adata_ATAC.h5ad")
_CACHE_NAMES = ("preprocessed_RNA.h5ad", "preprocessed_ADT.h5ad", "preprocessed_ATAC.h5ad")


def _preprocess(rna_adata: ad.AnnData, adt_adata: ad.AnnData, atac_adata: ad.AnnData) -> Tuple[ad.AnnData, ad.AnnData, ad.AnnData]:
    rna_adata.var["mt"] = rna_adata.var_names.str.startswith("MT-")
    for a in [rna_adata, adt_adata, atac_adata]:
        sc.pp.calculate_qc_metrics(a, percent_top=None, log1p=False, inplace=True)
    n_components = min(rna_adata.n_vars, adt_adata.n_vars)
    sc.pp.highly_variable_genes(rna_adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(rna_adata, target_sum=1e4)
    sc.pp.log1p(rna_adata)
    rna_adata.obsm["feat"] = SpatialPreprocessorUtils.pca(
        rna_adata[:, rna_adata.var["highly_variable"]], n_comps=n_components
    )
    sc.pp.highly_variable_genes(atac_adata, flavor="seurat_v3", n_top_genes=3000)
    SpatialPreprocessorUtils.lsi(atac_adata, use_highly_variable=False, n_components=n_components + 1)
    atac_adata.obsm["feat"] = atac_adata.obsm["X_lsi"].copy()
    adt_adata = SpatialPreprocessorUtils.clr_normalize_each_cell(adt_adata)
    adt_adata.obsm["feat"] = SpatialPreprocessorUtils.pca(adt_adata, n_comps=n_components)
    dist = squareform(pdist(rna_adata.obsm["spatial"], metric="euclidean"))
    for a in [rna_adata, adt_adata, atac_adata]:
        a.obsp["spatial_distances"] = sparse.csr_matrix(dist)
    return rna_adata, adt_adata, atac_adata


def load_nsf(
    raw_path: Path,
    preprocess_path: Path,
    use_cache: bool = True,
) -> Tuple[ad.AnnData, ad.AnnData, ad.AnnData, np.ndarray]:
    """
    Load and preprocess NSF RNA, ADT, ATAC, and true labels.

    raw_path: directory with adata_RNA.h5ad, adata_ADT.h5ad, adata_ATAC.h5ad and gt/spatial_factors.npy
    preprocess_path: directory to save/load preprocessed_RNA.h5ad, preprocessed_ADT.h5ad, preprocessed_ATAC.h5ad
    use_cache: if True and cache exists under preprocess_path, load from cache; else preprocess and save.

    Returns (rna_adata, adt_adata, atac_adata, true_labels) with obsm['feat'], obsp['spatial_distances'].
    """
    raw_path = Path(raw_path)
    preprocess_path = Path(preprocess_path)

    if use_cache:
        cache_files = [preprocess_path / n for n in _CACHE_NAMES]
        if all(p.exists() for p in cache_files):
            return (
                sc.read_h5ad(cache_files[0]),
                sc.read_h5ad(cache_files[1]),
                sc.read_h5ad(cache_files[2]),
                load_true_labels(raw_path),
            )

    rna_adata = sc.read_h5ad(raw_path / _RAW_NAMES[0])
    adt_adata = sc.read_h5ad(raw_path / _RAW_NAMES[1])
    atac_adata = sc.read_h5ad(raw_path / _RAW_NAMES[2])
    rna_adata.var_names_make_unique()
    adt_adata.var_names_make_unique()
    atac_adata.var_names_make_unique()

    rna_adata, adt_adata, atac_adata = _preprocess(rna_adata, adt_adata, atac_adata)

    preprocess_path.mkdir(parents=True, exist_ok=True)
    rna_adata.write_h5ad(preprocess_path / _CACHE_NAMES[0])
    adt_adata.write_h5ad(preprocess_path / _CACHE_NAMES[1])
    atac_adata.write_h5ad(preprocess_path / _CACHE_NAMES[2])

    return rna_adata, adt_adata, atac_adata, load_true_labels(raw_path)


def load_true_labels(raw_path: Path) -> np.ndarray:
    """
    Get true labels from spatial_factors.npy file under raw_path.
    """
    label_file_path = Path(raw_path) /"gt"/ "spatial_factors.npy"
    spatial_factors = np.load(label_file_path)
    true_labels = np.argmax(spatial_factors, axis=1)
    background_mask = np.all(spatial_factors == 0, axis=1)
    true_labels[background_mask] = spatial_factors.shape[1]
    return true_labels
