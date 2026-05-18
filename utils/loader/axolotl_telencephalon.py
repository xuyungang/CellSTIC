from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import scanpy as sc
import scipy.sparse as sparse
from scipy.spatial.distance import pdist, squareform

from utils.data import SpatialPreprocessorUtils
from utils.loader.file_utils import get_ligand_receptor_map_with_edge_types

_CACHE_NAME = "preprocessed_RNA.h5ad"

def _normalize_var_names_pipe_format(adata: ad.AnnData) -> None:
    """
    Normalize gene names like 'WNT7B | ...' -> 'WNT7B' in-place.

    This helps downstream code that expects plain gene symbols.
    """
    import pandas as pd

    if adata is None or adata.n_vars == 0:
        return
    s = pd.Index(adata.var_names.astype(str))
    if not s.str.contains(r"\|").any():
        return
    new = s.str.split("|", n=1, expand=False).str[0].str.strip()
    # Ensure uniqueness after collapsing names (keep deterministic suffixes if needed)
    new = new.to_series().reset_index(drop=True)
    new = pd.Index(new).astype(str)
    adata.var_names = new
    adata.var_names_make_unique()


def load_axolotl_telencephalon(
    raw_path: Path,
    preprocess_path: Optional[Path] = None,
    use_cache: bool = True,
    lr_path: Optional[Path] = None,
) -> Tuple[ad.AnnData, Optional[Dict[str, List[str]]]]:
    """
    Load the axolotl telencephalon dataset.
    """
    raw_path = Path(raw_path)
    if preprocess_path is None:
        preprocess_path = raw_path
    preprocess_path = Path(preprocess_path)

    # Ligand-receptor map
    lr: Optional[Dict[str, List[str]]] = None
    if lr_path is not None:
        lr_path = Path(lr_path)
        if not lr_path.exists():
            print(f"[load_axolotl_telencephalon] lr_path not found: {lr_path.absolute()}")
        else:
            try:
                lr, _ = get_ligand_receptor_map_with_edge_types(str(lr_path))
            except Exception as e:
                print(f"[load_axolotl_telencephalon] Failed to load LR from {lr_path}: {e}")
                lr = None

    cache_path = preprocess_path / _CACHE_NAME
    if use_cache and cache_path.exists():
        rna_adata = sc.read_h5ad(cache_path)
        # Cache files may have been generated before var-name normalization existed.
        # Normalize here as well to keep downstream (LR matching) stable.
        _normalize_var_names_pipe_format(rna_adata)
        return rna_adata, lr

    raw_files = [f for f in raw_path.glob("*.h5ad") if f.name != _CACHE_NAME]
    if not raw_files:
        raise FileNotFoundError(
            f"No raw h5ad found under raw_path (excluding cache names). raw_path: {raw_path.absolute()}"
        )
    rna_adata = sc.read_h5ad(raw_files[0])
    _normalize_var_names_pipe_format(rna_adata)

    # Ensure spatial coordinates exist
    if "spatial" not in rna_adata.obsm:
        if "x" in rna_adata.obs and "y" in rna_adata.obs:
            rna_adata.obsm["spatial"] = rna_adata.obs[["x", "y"]].to_numpy()
        else:
            raise ValueError("Spatial coordinates not found in adata.obsm['spatial'] or obs['x','y'].")

    # If already preprocessed, just cache and return
    if "feat" in rna_adata.obsm and "spatial_distances" in rna_adata.obsp:
        if use_cache:
            preprocess_path.mkdir(parents=True, exist_ok=True)
            rna_adata.write_h5ad(cache_path)
        return rna_adata, lr

    # Basic preprocessing (mirrors mouse_embryo / lymph_node loaders)
    rna_adata = rna_adata.copy()
    rna_adata.raw = rna_adata.copy()
    rna_adata.var["mt"] = rna_adata.var_names.astype(str).str.startswith("MT-")
    sc.pp.calculate_qc_metrics(rna_adata, percent_top=None, log1p=False, inplace=True)
    sc.pp.highly_variable_genes(rna_adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(rna_adata, target_sum=1e4)
    sc.pp.log1p(rna_adata)

    rna_adata.obsm["feat"] = SpatialPreprocessorUtils.pca(
        rna_adata[:, rna_adata.var["highly_variable"]], n_comps=500
    )
    rna_adata.obsp["spatial_distances"] = sparse.csr_matrix(
        squareform(pdist(rna_adata.obsm["spatial"], metric="euclidean"))
    )

    if use_cache:
        preprocess_path.mkdir(parents=True, exist_ok=True)
        rna_adata.write_h5ad(cache_path)
    return rna_adata, lr
