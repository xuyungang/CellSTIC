"""
Mouse brain 5M 20um: load and preprocess RNA, ADT, ATAC, H3K27ac, H3K27me3 (no domain/cell type annotation).
Call load_mouse_brain(raw_path, preprocess_path) to get (rna_adata, adt_adata, atac_adata, h3k27ac_adata, h3k27me3_adata).
"""

import gzip
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
import anndata as ad
import pandas as pd
import scanpy as sc
import scipy.io
import scipy.sparse as sparse
from scipy.spatial.distance import pdist, squareform

from utils.data import SpatialPreprocessorUtils
from utils.loader.file_utils import get_ligand_receptor_map_with_edge_types

_RAW_NAMES = ("rna.h5ad", "adt.h5ad", "atac.h5ad", "h3k27ac.h5ad", "h3k27me3.h5ad")
_CACHE_NAMES = ("preprocessed_RNA.h5ad", "preprocessed_ADT.h5ad", "preprocessed_ATAC.h5ad", "preprocessed_H3K27ac.h5ad", "preprocessed_H3K27me3.h5ad")
_PEAK_CSV_NAMES = (
    "atac/GSM8494157_5M_20um_ATAC_PeakMatrix.csv",
    "h3k27ac/GSM8494157_5M_20um_H3K27ac_PeakMatrix.csv",
    "h3k27me3/GSM8494157_5M_20um_H3K27me3_PeakMatrix.csv",
)
_PEAK_KEYS = ("atac", "h3k27ac", "h3k27me3")
_LR_FILE = "l-r/LR.csv"
_TYPE_CSV = "type/rna_10xWholeMouseBrain(CCN20230722)_CorrelationMapping_UTC_1769068788450.csv"


def _attach_mouse_brain_cell_types(
    raw_path: Path,
    rna_adata: ad.AnnData,
    adt_adata: ad.AnnData,
    atac_adata: ad.AnnData,
    h3k27ac_adata: ad.AnnData,
    h3k27me3_adata: ad.AnnData,
) -> None:
    """
    Attach cell type annotations from the 10x Whole Mouse Brain correlation mapping CSV.

    The CSV is expected at:
        raw_path / "type" / "rna_10xWholeMouseBrain(CCN20230722)_CorrelationMapping_UTC_1769068788450.csv"

    with columns:
        - cell_id
        - subclass_name (used as cell type label)

    The function adds obs["cell_type"] to all modalities, aligned by obs_names.
    """
    type_path = raw_path / _TYPE_CSV
    if not type_path.exists():
        return

    type_df = pd.read_csv(type_path)
    if "cell_id" not in type_df.columns or "subclass_name" not in type_df.columns:
        return

    type_df = type_df.set_index("cell_id")
    # Align to RNA cells; missing cells will become NaN
    cell_type_series = type_df.reindex(rna_adata.obs_names)["subclass_name"]
    cell_type_series = cell_type_series.astype("category")
    rna_adata.obs["cell_type"] = cell_type_series

    # Propagate to other modalities that share the same cell order
    for a in [adt_adata, atac_adata, h3k27ac_adata, h3k27me3_adata]:
        a.obs["cell_type"] = rna_adata.obs["cell_type"].copy()


def load_mouse_brain_gene_scores(
    raw_path: Path,
    preprocess_path: Optional[Path] = None,
    common_cells: Optional[pd.Index] = None,
    spatial_coords: Optional[np.ndarray] = None,
) -> Tuple[ad.AnnData, ad.AnnData, ad.AnnData]:
    """
    Load ATAC/H3K27ac/H3K27me3 gene score matrices as AnnData objects.

    This uses gene score CSVs under raw_path with the following relative paths:
        - atac:     ATAC/GSM8494157_5M_20um_ATAC_PeakMatrix.csv
        - h3k27ac:  H3K27ac/GSM8494157_5M_20um_H3K27ac_PeakMatrix.csv
        - h3k27me3: H3K27me3/GSM8494157_5M_20um_H3K27me3_PeakMatrix.csv

    Each returned AnnData has:
        - X: sparse matrix of gene scores (cells × genes)
        - var_names: gene names
        - obs_names: cell barcodes
        - obsm["spatial"]: spatial coordinates (N × 2)
        - obsm["gene_score"]: dense gene score matrix (float32)
        - uns["feature_type"]: "gene_scores"
    """
    raw_path = Path(raw_path)
    preprocess_path = Path(preprocess_path) if preprocess_path is not None else raw_path / "preprocess"

    gs_files = {
        "atac": preprocess_path / "atac" / "GSM8494157_5M_20um_ATAC_GeneScoreMatrix.csv",
        "h3k27ac": preprocess_path / "h3k27ac" / "GSM8494157_5M_20um_H3K27ac_GeneScoreMatrix.csv",
        "h3k27me3": preprocess_path / "h3k27me3" / "GSM8494157_5M_20um_H3K27me3_GeneScoreMatrix.csv",
    }

    for key, path in gs_files.items():
        if not path.exists():
            raise FileNotFoundError(
                f"GeneScoreMatrix CSV for {key} not found at {path}. "
                f"Please run MouseBrainRunner.peak_count with convert_to_csv=True "
                f"to generate gene score matrices before calling load_mouse_brain_gene_scores."
            )

    gs_dfs = {k: pd.read_csv(v, index_col=0) for k, v in gs_files.items()}

    # Restrict to provided cells if any
    if common_cells is not None:
        gs_dfs = {k: df.loc[common_cells] for k, df in gs_dfs.items()}

    # If spatial coordinates are not provided, load from tissue_positions_list.csv
    if spatial_coords is None:
        spatial_df = pd.read_csv(raw_path / "spatial" / "tissue_positions_list.csv", header=None)
        spatial_df.columns = [
            "barcode",
            "in_tissue",
            "array_row",
            "array_col",
            "pxl_row_in_fullres",
            "pxl_col_in_fullres",
        ]
        spatial_df.set_index("barcode", inplace=True)

        # Use intersection across all three gene score matrices and spatial barcodes
        common = gs_dfs["atac"].index
        for k in ["h3k27ac", "h3k27me3"]:
            common = common.intersection(gs_dfs[k].index)
        common = common.intersection(spatial_df.index)

        # Align all gene score matrices and spatial coordinates to the common cell set
        gs_dfs = {k: df.loc[common] for k, df in gs_dfs.items()}
        spatial_coords = spatial_df.loc[common][["pxl_col_in_fullres", "pxl_row_in_fullres"]].values

    adatas: List[ad.AnnData] = []
    for key in ["atac", "h3k27ac", "h3k27me3"]:
        # Replace NaNs with zeros to make PCA/UMAP happy
        df = gs_dfs[key].fillna(0.0).astype(np.float32)
        a = ad.AnnData(X=sparse.csr_matrix(df.values))
        a.var_names = df.columns
        a.obs_names = df.index
        a.obsm["spatial"] = spatial_coords
        a.obsm["gene_score"] = df.values
        a.uns["feature_type"] = "gene_scores"
        adatas.append(a)

    return adatas[0], adatas[1], adatas[2]


def _preprocess(
    rna_adata: ad.AnnData,
    adt_adata: ad.AnnData,
    atac_adata: ad.AnnData,
    h3k27ac_adata: ad.AnnData,
    h3k27me3_adata: ad.AnnData,
    n_components: int = 131,
) -> Tuple[ad.AnnData, ad.AnnData, ad.AnnData, ad.AnnData, ad.AnnData]:
    """Normalize, PCA/LSI, and add obsm['feat'] and obsp['spatial_distances']."""
    rna_adata.raw = rna_adata.copy()
    rna_adata.var["mt"] = rna_adata.var_names.str.startswith("MT-")
    for a in [rna_adata, adt_adata, atac_adata, h3k27ac_adata, h3k27me3_adata]:
        sc.pp.calculate_qc_metrics(a, percent_top=None, log1p=False, inplace=True)
    sc.pp.highly_variable_genes(rna_adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(rna_adata, target_sum=1e4)
    sc.pp.log1p(rna_adata)
    rna_adata.obsm["feat"] = SpatialPreprocessorUtils.pca(
        rna_adata[:, rna_adata.var["highly_variable"]], n_comps=n_components
    )
    adt_adata = SpatialPreprocessorUtils.clr_normalize_each_cell(adt_adata)
    adt_adata.obsm["feat"] = SpatialPreprocessorUtils.pca(adt_adata, n_comps=n_components)
    for chrom_adata in [atac_adata, h3k27ac_adata, h3k27me3_adata]:
        sc.pp.highly_variable_genes(chrom_adata, flavor="seurat_v3", n_top_genes=3000)
        SpatialPreprocessorUtils.lsi(chrom_adata, use_highly_variable=True, n_components=n_components + 1)
        chrom_adata.obsm["feat"] = chrom_adata.obsm["X_lsi"].copy()
    dist = squareform(pdist(rna_adata.obsm["spatial"], metric="euclidean"))
    for a in [rna_adata, adt_adata, atac_adata, h3k27ac_adata, h3k27me3_adata]:
        a.obsp["spatial_distances"] = sparse.csr_matrix(dist)
    return rna_adata, adt_adata, atac_adata, h3k27ac_adata, h3k27me3_adata


def load_mouse_brain(
    raw_path: Path,
    preprocess_path: Path = None,
    use_cache: bool = True,
) -> Tuple[ad.AnnData, ad.AnnData, ad.AnnData, ad.AnnData, ad.AnnData, Dict[str, List[str]]]:
    """
    Load and preprocess mouse brain 5M (RNA, ADT, ATAC, H3K27ac, H3K27me3). No domain or cell type annotation.

    raw_path: directory for original materials only (RNA/, ADT/, spatial/ 10x outputs).
    preprocess_path: directory to load/save rna.h5ad, adt.h5ad (_RAW_NAMES), preprocessed_*.h5ad (_CACHE_NAMES), and to read atac/h3k27ac/h3k27me3 PeakMatrix CSVs (defaults to raw_path).
    use_cache: if True and cache exists under preprocess_path, load from cache; else preprocess and save.

    Returns (rna_adata, adt_adata, atac_adata, h3k27ac_adata, h3k27me3_adata, ligand_receptor_map)
    with obsm['feat'], obsp['spatial_distances'].
    ligand_receptor_map is loaded from raw_path/l-r/LR.csv if the file exists; otherwise None.
    """
    raw_path = Path(raw_path)
    if preprocess_path is None:
        preprocess_path = raw_path
    preprocess_path = Path(preprocess_path)

    # Try to load ligand–receptor map from CSV under raw_path/l-r/LR.csv
    lr_path = raw_path / _LR_FILE
    ligand_receptor_map: Dict[str, List[str]] = None
    if lr_path.exists():
        ligand_receptor_map, _ = get_ligand_receptor_map_with_edge_types(str(lr_path))

    if use_cache:
        cache_files = [preprocess_path / n for n in _CACHE_NAMES]
        if all(p.exists() for p in cache_files):
            # When loading from cache, still return ligand_receptor_map loaded above (may be None)
            rna_adata, adt_adata, atac_adata, h3k27ac_adata, h3k27me3_adata = (
                sc.read_h5ad(p) for p in cache_files
            )
            _attach_mouse_brain_cell_types(
                raw_path, rna_adata, adt_adata, atac_adata, h3k27ac_adata, h3k27me3_adata
            )
            return rna_adata, adt_adata, atac_adata, h3k27ac_adata, h3k27me3_adata, ligand_receptor_map

    # Load or build RNA/ADT: prefer pre-built h5ad under preprocess_path; else build from raw_path (10x RNA/ADT, spatial)
    raw_rna = preprocess_path / _RAW_NAMES[0]
    raw_adt = preprocess_path / _RAW_NAMES[1]
    if raw_rna.exists() and raw_adt.exists():
        rna_adata = sc.read_h5ad(raw_rna)
        adt_adata = sc.read_h5ad(raw_adt)
    else:
        rna_dir = raw_path / "rna"
        rna_mtx = rna_dir / "matrix.mtx.gz"
        adt_dir = raw_path / "adt"
        adt_mtx = adt_dir / "matrix.mtx.gz"
        if not rna_mtx.exists() or not adt_mtx.exists():
            raise FileNotFoundError(
                "Mouse brain RNA/ADT not found. Either provide under preprocess_path:\n"
                f"  - {raw_rna} and {raw_adt} (pre-built h5ad), or\n"
                "under raw_path (original 10x materials):\n"
                f"  - {rna_mtx} and {adt_mtx}.\n"
                f"raw_path: {raw_path.absolute()}\npreprocess_path: {preprocess_path.absolute()}"
            )
        rna_adata = sc.read_10x_mtx(str(rna_dir), var_names="gene_symbols", cache=True)
        rna_adata.var_names_make_unique()
        spatial_df = pd.read_csv(raw_path / "spatial" / "tissue_positions_list.csv", header=None)
        spatial_df.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
        spatial_df.set_index("barcode", inplace=True)
        protein_path = raw_path / "adt"
        with gzip.open(protein_path / "features.tsv.gz", "rt") as f:
            features = [line.strip() for line in f]
        with gzip.open(protein_path / "barcodes.tsv.gz", "rt") as f:
            barcodes = [line.strip() for line in f]
        adt_adata = ad.AnnData(X=scipy.io.mmread(str(protein_path / "matrix.mtx.gz")).T.tocsr())
        adt_adata.var_names, adt_adata.obs_names = features, barcodes
        adt_adata.var_names_make_unique()
        common = rna_adata.obs_names.intersection(spatial_df.index).intersection(adt_adata.obs_names)
        rna_adata = rna_adata[common, :].copy()
        adt_adata = adt_adata[common, :].copy()
        coords = spatial_df.loc[common][["pxl_col_in_fullres", "pxl_row_in_fullres"]].values.astype(np.float64)
        rna_adata.obsm["spatial"] = adt_adata.obsm["spatial"] = coords
        preprocess_path.mkdir(parents=True, exist_ok=True)
        rna_adata.write_h5ad(raw_rna)
        adt_adata.write_h5ad(raw_adt)

    peak_dfs = {
        k: pd.read_csv(preprocess_path / v, index_col=0)
        for k, v in zip(_PEAK_KEYS, _PEAK_CSV_NAMES)
    }
    common_all = rna_adata.obs_names.intersection(adt_adata.obs_names)
    for k in _PEAK_KEYS:
        common_all = common_all.intersection(peak_dfs[k].index)
    rna_adata = rna_adata[common_all, :].copy()
    adt_adata = adt_adata[common_all, :].copy()
    if "spatial" not in rna_adata.obsm:
        spatial_df = pd.read_csv(raw_path / "spatial" / "tissue_positions_list.csv", header=None)
        spatial_df.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
        spatial_df.set_index("barcode", inplace=True)
        coords = spatial_df.loc[common_all][["pxl_col_in_fullres", "pxl_row_in_fullres"]].values.astype(np.float64)
        rna_adata.obsm["spatial"] = coords
    adt_adata.obsm["spatial"] = rna_adata.obsm["spatial"]

    peak_adatas = {}
    for key in _PEAK_KEYS:
        df = peak_dfs[key].loc[common_all]
        a = ad.AnnData(X=sparse.csr_matrix(df.values.astype(np.float32)))
        a.var_names = df.columns.tolist()
        a.obs_names = df.index.tolist()
        a.obsm["spatial"] = rna_adata.obsm["spatial"]
        a.uns["feature_type"] = "peaks"
        peak_adatas[key] = a
    atac_adata = peak_adatas["atac"]
    h3k27ac_adata = peak_adatas["h3k27ac"]
    h3k27me3_adata = peak_adatas["h3k27me3"]

    raw_atac = preprocess_path / _RAW_NAMES[2]
    raw_h3k27ac = preprocess_path / _RAW_NAMES[3]
    raw_h3k27me3 = preprocess_path / _RAW_NAMES[4]
    preprocess_path.mkdir(parents=True, exist_ok=True)
    atac_adata.write_h5ad(raw_atac)
    h3k27ac_adata.write_h5ad(raw_h3k27ac)
    h3k27me3_adata.write_h5ad(raw_h3k27me3)

    preprocess_path.mkdir(parents=True, exist_ok=True)
    atac_adata.write_h5ad(preprocess_path / _RAW_NAMES[2])
    h3k27ac_adata.write_h5ad(preprocess_path / _RAW_NAMES[3])
    h3k27me3_adata.write_h5ad(preprocess_path / _RAW_NAMES[4])

    for a in [rna_adata, adt_adata, atac_adata, h3k27ac_adata, h3k27me3_adata]:
        a.var_names_make_unique()

    # Attach cell type annotations before preprocessing and caching, so they are saved to disk.
    _attach_mouse_brain_cell_types(
        raw_path, rna_adata, adt_adata, atac_adata, h3k27ac_adata, h3k27me3_adata
    )

    rna_adata, adt_adata, atac_adata, h3k27ac_adata, h3k27me3_adata = _preprocess(
        rna_adata, adt_adata, atac_adata, h3k27ac_adata, h3k27me3_adata
    )

    preprocess_path.mkdir(parents=True, exist_ok=True)
    for i, adata in enumerate([rna_adata, adt_adata, atac_adata, h3k27ac_adata, h3k27me3_adata]):
        adata.write_h5ad(preprocess_path / _CACHE_NAMES[i])

    return rna_adata, adt_adata, atac_adata, h3k27ac_adata, h3k27me3_adata, ligand_receptor_map
