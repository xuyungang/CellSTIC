"""
Mouse embryo stage: load and preprocess RNA from raw h5ad, optional annotation filtering.
Call load_mouse_embryo(raw_path, preprocess_path) to get (rna_adata, lr).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import scanpy as sc
import scipy.sparse as sparse
from scipy.spatial.distance import pdist, squareform

from utils.data import SpatialPreprocessorUtils
from utils.loader.file_utils import get_ligand_receptor_map_with_edge_types
from utils.tools.celltypist_utils import CellTypistAnnotator, SPECIES_MODEL_MAP

_CACHE_NAME_FILTERED = "preprocessed_RNA_filtered.h5ad"


def _annotate_cell_type_by_domain(
    rna_adata: ad.AnnData,
    annotation_key: str,
    annotation_model_map: Dict[str, str],
    cell_type_key: str = "cell_type",
) -> ad.AnnData:
    if annotation_key not in rna_adata.obs:
        return rna_adata
    domains = rna_adata.obs[annotation_key].astype(str).unique()
    rna_adata.obs[cell_type_key] = ""
    for domain in domains:
        if domain not in annotation_model_map:
            continue
        model_spec = annotation_model_map[domain]
        mask = rna_adata.obs[annotation_key].astype(str) == domain
        sub = rna_adata[mask].copy()
        if sub.n_obs == 0:
            continue
        model_path = SPECIES_MODEL_MAP.get(model_spec, model_spec)
        annotator = CellTypistAnnotator(model=model_path, species=None)
        annotator.annotate(sub, output_path=None)
        rna_adata.obs.loc[mask, cell_type_key] = sub.obs["predicted_labels"].values

    # Post-process labels like "Blood: Undefined" -> "Blood"
    s = rna_adata.obs[cell_type_key]
    not_na = s.notna()
    s_str = s[not_na].astype(str)
    has_colon = s_str.str.contains(":", regex=False)
    if has_colon.any():
        s_str.loc[has_colon] = s_str.loc[has_colon].str.split(":", n=1).str[0].str.strip()
        rna_adata.obs.loc[not_na, cell_type_key] = s_str.values
    rna_adata.obs[cell_type_key] = rna_adata.obs[cell_type_key].astype("category")
    n_types = rna_adata.obs[cell_type_key].nunique()
    print(f"[load_mouse_embryo] cell type annotation by domain: {n_types} types in obs[{cell_type_key!r}]")
    return rna_adata


def load_mouse_embryo(
    raw_path: Path,
    preprocess_path: Optional[Path] = None,
    use_cache: bool = True,
    annotation_key: Optional[str] = "annotation",
    annotation_value: Optional[str] = None,
    lr_path: Optional[Path] = None,
    annotation_model_map: Optional[Dict[str, str]] = None,
) -> Tuple[ad.AnnData, Optional[Dict[str, List[str]]]]:
    """
    Load and preprocess mouse embryo stage RNA, with optional annotation filtering.

    raw_path: directory containing original *.h5ad (first file not matching cache name is used).
    preprocess_path: directory to load/save preprocessed_RNA_filtered.h5ad (defaults to raw_path).
    use_cache: if True and cache exists under preprocess_path, load from cache; else preprocess and save.
    annotation_key: column in adata.obs used for filtering (default "annotation"). Ignored if annotation_value is None.
    annotation_value: if provided, keep only cells whose obs[annotation_key] == this value;
        then preprocess and save the filtered data for this single region.
    lr_path: path to ligand-receptor CSV (e.g. LR.csv). If None, tries raw_path/l-r/LR.csv and raw_path/LR.csv.
    annotate_cell_type: if True, run CellTypist per domain (Brain/Liver) with different models and write obs['cell_type'].
    annotation_model_map: domain label -> model (species key or .pkl path). Default: Brain->mouse_brain, Liver->mouse_liver.

    Returns:
        (rna_adata, lr): rna_adata with obsm['feat'], obsp['spatial_distances'];
        lr is ligand_receptor_map (Dict[ligand, List[receptor]]) from lr_path or auto-discovered CSV, or None.
    """
    raw_path = Path(raw_path)
    if preprocess_path is None:
        preprocess_path = raw_path
    preprocess_path = Path(preprocess_path)

    # Load ligand-receptor map from CSV
    lr: Optional[Dict[str, List[str]]] = None
    if lr_path is not None:
        lr_path = Path(lr_path)
        if not lr_path.exists():
            print(f"[load_mouse_embryo] lr_path not found: {lr_path.absolute()}")
        else:
            try:
                lr, _ = get_ligand_receptor_map_with_edge_types(str(lr_path))
            except Exception as e:
                print(f"[load_mouse_embryo] Failed to load LR from {lr_path}: {e}")
                lr = None

    do_filter = annotation_value is not None
    cache_path = preprocess_path / _CACHE_NAME_FILTERED
    if use_cache and cache_path.exists():
        rna_adata = sc.read_h5ad(cache_path)
        return rna_adata, lr

    raw_files = [f for f in raw_path.glob("*.h5ad") if f.name != _CACHE_NAME_FILTERED]
    if not raw_files:
        raise FileNotFoundError(
            f"No raw h5ad found under raw_path (excluding cache names). "
            f"raw_path: {raw_path.absolute()}, preprocess_path: {preprocess_path.absolute()}"
        )
    rna_adata = sc.read_h5ad(raw_files[0])

    # Annotation filtering: keep only cells with a single annotation_value
    if do_filter:
        if not annotation_key:
            raise ValueError("annotation_key must be set when annotation_value is provided")
        if annotation_key not in rna_adata.obs:
            raise ValueError(
                f"Annotation key '{annotation_key}' not in adata.obs. Keys: {list(rna_adata.obs.keys())}"
            )
        mask = rna_adata.obs[annotation_key].astype(str).eq(str(annotation_value)).values
        if mask.sum() == 0:
            raise ValueError(
                f"No cells with annotation_value {annotation_value!r}. "
                f"Available: {sorted(rna_adata.obs[annotation_key].astype(str).unique())}"
            )
        n_before = rna_adata.n_obs
        rna_adata = rna_adata[mask, :].copy()
        n_after = rna_adata.n_obs
        print(
            f"[load_mouse_embryo] annotation filter: key={annotation_key!r}, value={annotation_value!r}; "
            f"cells {n_before} -> {n_after} (kept {n_after}, removed {n_before - n_after})"
        )

    if "feat" in rna_adata.obsm and "spatial_distances" in rna_adata.obsp:
        if use_cache:
            preprocess_path.mkdir(parents=True, exist_ok=True)
            rna_adata.write_h5ad(cache_path)
        return rna_adata, lr

    rna_adata.raw = rna_adata.copy()
    rna_adata.var["mt"] = rna_adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(rna_adata, percent_top=None, log1p=False, inplace=True)
    # sc.pp.filter_genes(rna_adata, min_cells=100)
    sc.pp.highly_variable_genes(rna_adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(rna_adata, target_sum=1e4)
    sc.pp.log1p(rna_adata)
    rna_adata.obsm["feat"] = SpatialPreprocessorUtils.pca(
        rna_adata[:, rna_adata.var["highly_variable"]], n_comps=500
    )
    rna_adata.obsp["spatial_distances"] = sparse.csr_matrix(
        squareform(pdist(rna_adata.obsm["spatial"], metric="euclidean"))
    )

    if annotation_model_map:
        rna_adata = _annotate_cell_type_by_domain(
            rna_adata, annotation_key, annotation_model_map, cell_type_key="cell_type"
        )

    if use_cache:
        preprocess_path.mkdir(parents=True, exist_ok=True)
        rna_adata.write_h5ad(cache_path)
    return rna_adata, lr
