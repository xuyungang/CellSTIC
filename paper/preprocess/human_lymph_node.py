"""Assemble human lymph-node RNA and ADT from ``raw/`` into ``h5ad/``.

By default, both directories are under ``paper/data/human_lymph_node``.
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path
from typing import Dict, List, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODALITIES = ("rna", "adt")
RAW_RNA_FEATURES = "raw/rna/GSM8195494_A1LN_features.tsv.gz"
RAW_RNA_MATRIX = "raw/rna/GSM8195494_A1LN_matrix.mtx.gz"
RAW_ADT_H5AD = "raw/adt/GSM8195498_A1_LN_Protein.h5ad"
RAW_ADT_ISOTYPE = "raw/adt/GSM8195498_A1LN_isotype_normalization_factors.csv.gz"
RAW_SPATIAL_POSITIONS = "raw/spatial/GSM8195494_A1LN_tissue_positions.csv"
CELL_TYPE_TABLE = "raw/type/cell_type_annotations_mapped.csv"
LIGAND_RECEPTOR_PAIRS = (
    ("CD80", "CD28"),
    ("CD80", "CTLA4"),
    ("CD40LG", "ITGA2B:ITGB3"),
    ("SELE", "CEACAM1"),
    ("CD209", "CEACAM1"),
    ("CXCL12", "CXCR4"),
    ("CCL21", "CCR7"),
    ("CXCL13", "CXCR5"),
    ("SELPLG", "SELE"),
    ("CCL19", "CCR7"),
    ("CD86", "CD28"),
    ("CXCL8", "ACKR1"),
    ("IL17A", "IL17RA:IL17RC"),
    ("IL17B", "IL17RB:IL17RC"),
    ("IL10", "IL10RA:IL10RB"),
    ("LGALS9", "CD44"),
    ("CXCL11", "CXCR3"),
    ("CCL17", "CCR4"),
    ("CCL22", "CCR4"),
    ("CCL8", "CCR2"),
    ("CCL7", "CCR2"),
    ("ENTPD1", "TMIGD3"),
    ("TNF", "TNFRSF1A"),
    ("ICAM1", "ITGAX:ITGB2"),
    ("FASLG", "FAS"),
    ("NECTIN1", "NECTIN4"),
)


def _load_tissue_positions(dataset_dir: Path) -> pd.DataFrame:
    """Return the reference spatial coordinates indexed by cell barcode."""
    path = dataset_dir / RAW_SPATIAL_POSITIONS
    positions = pd.read_csv(path)
    required_columns = {"barcode", "pxl_col_in_fullres", "pxl_row_in_fullres"}
    missing_columns = required_columns.difference(positions.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")
    positions["barcode"] = positions["barcode"].astype(str)
    return positions.set_index("barcode")[["pxl_col_in_fullres", "pxl_row_in_fullres"]]


def _spatial_for_cells(cell_ids: pd.Index, positions: pd.DataFrame) -> np.ndarray:
    coordinates = positions.reindex(cell_ids).to_numpy(dtype=np.float64)
    # A source barcode without a tissue-position record cannot be spatially
    # located; retain the reference loader's safe zero-coordinate fallback.
    return np.nan_to_num(coordinates, nan=0.0)


def _build_rna_from_raw(dataset_dir: Path, positions: pd.DataFrame) -> ad.AnnData:
    """Build the in-tissue RNA AnnData from the downloaded 10x files."""
    isotype = pd.read_csv(dataset_dir / RAW_ADT_ISOTYPE)
    required_columns = {"barcode", "in_tissue", "normalization_factor"}
    missing_columns = required_columns.difference(isotype.columns)
    if missing_columns:
        raise ValueError(
            f"{dataset_dir / RAW_ADT_ISOTYPE} is missing columns: "
            f"{sorted(missing_columns)}"
        )
    isotype = isotype.loc[isotype["in_tissue"] == 1].copy()
    barcodes = pd.Index(isotype["barcode"].astype(str))

    matrix = scipy.io.mmread(dataset_dir / RAW_RNA_MATRIX)
    matrix = (
        matrix.T.tocsr() if sparse.issparse(matrix) else sparse.csr_matrix(matrix.T)
    )
    with gzip.open(dataset_dir / RAW_RNA_FEATURES, "rt") as handle:
        features = pd.read_csv(
            handle,
            sep="\t",
            header=None,
            names=["gene_id", "gene_name", "feature_type"],
        )
    gene_mask = features["feature_type"].eq("Gene Expression").to_numpy()
    matrix = matrix[:, gene_mask]
    genes = features.loc[gene_mask, ["gene_id", "gene_name"]].reset_index(drop=True)
    if matrix.shape[0] != len(barcodes):
        raise ValueError(
            f"RNA matrix has {matrix.shape[0]} cells, but isotype metadata has "
            f"{len(barcodes)} in-tissue barcodes"
        )

    rna = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(index=barcodes),
        var=pd.DataFrame(index=pd.Index(genes["gene_name"], name="gene_name")),
    )
    rna.var["gene_ids"] = genes["gene_id"].to_numpy()
    rna.var_names_make_unique()
    rna.obs["isotype_norm_factor"] = isotype["normalization_factor"].to_numpy()
    rna.obsm["spatial"] = _spatial_for_cells(rna.obs_names, positions)
    return rna


def _load_source_modalities(dataset_dir: Path) -> List[ad.AnnData]:
    """Load RNA and ADT exclusively from the raw downloads."""
    positions = _load_tissue_positions(dataset_dir)
    rna = _build_rna_from_raw(dataset_dir, positions)
    adt_path = dataset_dir / RAW_ADT_H5AD
    if not adt_path.is_file():
        raise FileNotFoundError(adt_path)
    adt = ad.read_h5ad(adt_path)
    adt.obsm["spatial"] = _spatial_for_cells(adt.obs_names, positions)

    return [rna, adt]


def _attach_human_lymph_node_cell_types(
    modalities: List[ad.AnnData], dataset_dir: Path
) -> None:
    """Attach the curated mapped labels to RNA and ADT by cell barcode."""
    path = dataset_dir / CELL_TYPE_TABLE
    if not path.is_file():
        raise FileNotFoundError(f"Human lymph-node cell-type table is required: {path}")
    annotations = pd.read_csv(path, comment="#")
    required_columns = {"cell_id", "predicted_labels"}
    missing_columns = required_columns.difference(annotations.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")
    mapping = annotations.drop_duplicates("cell_id").set_index("cell_id")[
        "predicted_labels"
    ]
    categories = sorted(set(mapping.dropna().astype(str)) | {"Unknown"})
    for adata in modalities:
        labels = mapping.reindex(adata.obs_names).fillna("Unknown")
        adata.obs["cell_type"] = pd.Categorical(labels, categories=categories)


def _attach_dataset_metadata(modalities: List[ad.AnnData]) -> None:
    """Attach the fixed human lymph-node LR panel without a CSV dependency."""
    ligand_receptor_map: Dict[str, List[str]] = {}
    for ligand, receptor in LIGAND_RECEPTOR_PAIRS:
        ligand_receptor_map.setdefault(ligand, []).append(receptor)
    lr_table = pd.DataFrame(
        [
            {"ligand": ligand, "receptor": receptor, "idx": index}
            for index, (ligand, receptor) in enumerate(LIGAND_RECEPTOR_PAIRS, start=1)
        ]
    )
    metadata = {
        "dataset": "human_lymph_node",
        "ligand_receptor_map": ligand_receptor_map,
        "lr_table": lr_table,
    }
    for adata in modalities:
        adata.uns.update(metadata)


def assemble(dataset_dir: Path) -> Tuple[ad.AnnData, ad.AnnData]:
    """Read ``raw/``, align cells, and write ``h5ad/{rna,adt}.h5ad``."""
    dataset_dir = Path(dataset_dir)
    source_modalities = _load_source_modalities(dataset_dir)

    common_cells = source_modalities[0].obs_names.intersection(
        source_modalities[1].obs_names, sort=False
    )
    if len(common_cells) == 0:
        raise ValueError("Human lymph-node RNA and ADT do not share cell identifiers")

    modalities = [adata[common_cells].copy() for adata in source_modalities]
    spatial = next(
        (
            np.asarray(adata.obsm["spatial"])
            for adata in modalities
            if "spatial" in adata.obsm
        ),
        None,
    )
    if spatial is None:
        raise KeyError("Human lymph-node inputs contain no obsm['spatial']")
    for adata in modalities:
        if "counts" not in adata.layers:
            adata.layers["counts"] = adata.X.copy()
        adata.obsm["spatial"] = spatial.copy()

    _attach_human_lymph_node_cell_types(modalities, dataset_dir)
    _attach_dataset_metadata(modalities)

    output_dir = dataset_dir / "h5ad"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, adata in zip(MODALITIES, modalities):
        adata.write_h5ad(output_dir / f"{name}.h5ad")
    print(
        f"human_lymph_node: RNA {modalities[0].shape}, "
        f"ADT {modalities[1].shape} -> {output_dir}"
    )
    return modalities[0], modalities[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "paper" / "data" / "human_lymph_node",
        help="Dataset root: read from raw/ and write to h5ad/ (default: %(default)s)",
    )
    args = parser.parse_args()
    assemble(args.dataset_dir)


if __name__ == "__main__":
    main()
