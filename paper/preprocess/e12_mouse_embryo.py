#!/usr/bin/env python3
"""Assemble the paired E12.5 mouse embryo Visium HD–PLA samples."""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import io
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyreadr
import scipy.io
import scipy.sparse as sparse
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "paper" / "data" / "e12_mouse_embryo"

SAMPLES = {
    "A1": {
        "accession": "GSM8829020",
        "library_id": "GSM8829020_A1",
        "barcode_file": "GSM8829020_A1-16um-barcodes.recovered.tsv.gz",
        "pla_pairs": (
            ("Fgf8", "Fgfr1", "A1_fused_Fgf8-Fgfr1_CH2.png"),
            ("Dll1", "Notch1", "A1_fused_DLL-NOTCH_ch02.png"),
        ),
    },
    "D1": {
        "accession": "GSM8829021",
        "library_id": "GSM8829021_D1",
        "barcode_file": "GSM8829021_D1-16um-barcodes.tsv.gz",
        "pla_pairs": (
            ("Spp1", "Cd44", "D1_fused_Spp1-Cd44_CH2.png"),
            ("Igf2", "Igf2r", "D1_fused_Igf-Igf2r_CH2.png"),
            ("Efna3", "Epha5", "D1_fused_Efna3-Epha5_brain_CH2.png"),
        ),
    },
}

# Protein labels used in the authors' Figure 4c metadata filenames.
AUTHOR_PAIR_NAMES = {
    "Fgf8_Fgfr1": "FGF8-FGFR1",
    "Dll1_Notch1": "DLL1-NOTC1",
    "Spp1_Cd44": "OSTP-CD44",
    "Igf2_Igf2r": "IGF2-MPRI",
    "Efna3_Epha5": "EFNA3-EPHA5",
}


def _read_lines(path: Path) -> list[str]:
    with gzip.open(path, "rt") as handle:
        return [line.rstrip("\n") for line in handle if line.strip()]


def _recover_a1_barcodes(raw_dir: Path) -> Path:
    """Recover the A1 matrix barcodes from the ordered graph-cluster table."""
    target = raw_dir / SAMPLES["A1"]["barcode_file"]
    if target.is_file():
        return target
    source = raw_dir / "GSM8829020_A1-16um-clusters_graphclust.csv.gz"
    with gzip.open(source, "rt", newline="") as handle:
        barcodes = [row["Barcode"] for row in csv.DictReader(handle)]
    if len(barcodes) != 85_017 or len(set(barcodes)) != len(barcodes):
        raise ValueError(
            "A1 graph-cluster table does not contain the expected 85,017 "
            "unique barcodes"
        )
    with gzip.open(target, "wt", newline="") as handle:
        handle.writelines(f"{barcode}\n" for barcode in barcodes)
    return target


def _read_positions(path: Path) -> pd.DataFrame:
    # GEO supplies an already compressed Parquet file with an additional .gz
    # layer, which pandas.read_parquet does not decompress automatically.
    with gzip.open(path, "rb") as handle:
        payload = handle.read()
    positions = pq.read_table(pa.BufferReader(payload)).to_pandas()
    required = {
        "barcode",
        "in_tissue",
        "array_row",
        "array_col",
        "pxl_row_in_fullres",
        "pxl_col_in_fullres",
    }
    missing = required.difference(positions.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if positions["barcode"].duplicated().any():
        raise ValueError(f"{path} contains duplicate barcodes")
    return positions.set_index("barcode")


def _read_image(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        with Image.open(io.BytesIO(handle.read())) as image:
            return np.asarray(image.convert("RGB"))


def _load_matrix(path: Path) -> sparse.csr_matrix:
    print(f"  reading {path.name}", flush=True)
    with gzip.open(path, "rb") as handle:
        matrix = scipy.io.mmread(handle)
    if not sparse.issparse(matrix):
        matrix = sparse.coo_matrix(matrix)
    if matrix.dtype != np.int32:
        matrix = matrix.astype(np.int32)
    return matrix.T.tocsr()


def _add_author_metadata(adata: ad.AnnData, dataset_dir: Path, sample: str) -> None:
    """Join pair-specific ROI/PLA values by barcode; retain unannotated bins."""
    metadata_dir = dataset_dir / "raw" / "cytosignal_figure"
    manifest_path = metadata_dir / "download_manifest.json"
    with manifest_path.open() as handle:
        manifest = json.load(handle)
    sources = {entry["file"]: entry for entry in manifest["files"]}
    if len(sources) != len(manifest["files"]):
        raise ValueError(f"{manifest_path} contains duplicate filenames")

    pairs = adata.uns["pla_pairs"].copy()
    author_metadata = {}
    checksums = {}
    for index, pair in pairs.iterrows():
        pair_key = f"{pair['ligand']}_{pair['receptor']}"
        filename = f"MetaData_{sample}_16um_{AUTHOR_PAIR_NAMES[pair_key]}.rds"
        path = metadata_dir / filename
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        if checksum != sources[filename]["sha256"]:
            raise ValueError(f"{path} does not match its recorded SHA-256 checksum")

        metadata = pyreadr.read_r(str(path))[None]
        required = {"region", "PLA_Median", "x", "y"}
        missing_columns = required.difference(metadata.columns)
        if missing_columns:
            raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")
        if metadata.empty or metadata.index.has_duplicates or metadata.index.isna().any():
            raise ValueError(f"{path} must contain nonempty, unique barcodes")
        metadata.index = pd.Index(metadata.index, name="barcode")
        missing_barcodes = metadata.index.difference(adata.obs_names)
        if len(missing_barcodes):
            raise ValueError(
                f"{path}: {len(missing_barcodes)} author barcodes are absent from {sample}"
            )
        if not metadata["region"].isin(["roi", "not_roi"]).all():
            raise ValueError(f"{path} contains missing or unknown region labels")

        # Author plotting coordinates flip the full-resolution image's y axis.
        expected_xy = adata.obs.loc[
            metadata.index, ["pxl_col_in_fullres", "pxl_row_in_fullres"]
        ].to_numpy(dtype=np.float64)
        expected_xy[:, 1] *= -1
        if not np.allclose(
            metadata[["x", "y"]].to_numpy(), expected_xy, rtol=0, atol=1e-6
        ):
            raise ValueError(f"{path}: author coordinates do not match barcode positions")

        aligned = metadata.reindex(adata.obs_names)
        region_column = f"region_{pair_key}"
        pla_column = f"pla_median_{pair_key}"
        adata.obs[region_column] = pd.Categorical(
            aligned["region"], categories=["not_roi", "roi"]
        )
        adata.obs[pla_column] = aligned["PLA_Median"]
        author_metadata[pair_key] = metadata
        checksums[pair_key] = checksum
        pairs.loc[index, "author_metadata_file"] = str(path.relative_to(dataset_dir))
        pairs.loc[index, "region_column"] = region_column
        pairs.loc[index, "pla_median_column"] = pla_column
        print(
            f"  {pair_key}: {len(metadata):,} author annotations, "
            f"{aligned['region'].eq('roi').sum():,} ROI bins, "
            f"{aligned['region'].isna().sum():,} unannotated bins",
            flush=True,
        )

    adata.uns["pla_pairs"] = pairs
    # Keep complete source tables, including benchmark scores and original x/y.
    # These tables contain only author-annotated bins and are indexed by barcode.
    adata.uns["author_metadata"] = author_metadata
    adata.uns["author_metadata_provenance"] = {
        "repository": manifest["repository"],
        "commit": manifest["commit"],
        "sha256": checksums,
        "coordinate_convention": "x = pxl_col_in_fullres; y = -pxl_row_in_fullres",
        "missing_annotation": "NA in obs means no author annotation, not not_roi",
    }
    adata.uns["source_files"]["author_metadata_manifest"] = str(
        manifest_path.relative_to(dataset_dir)
    )


def assemble_sample(dataset_dir: Path, sample: str, overwrite: bool) -> Path:
    config = SAMPLES[sample]
    raw_dir = dataset_dir / "raw" / "visium_hd"
    output_dir = dataset_dir / "h5ad"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{sample}.h5ad"
    if output.is_file() and not overwrite:
        print(f"{sample}: exists, skipping {output}", flush=True)
        return output

    prefix = config["library_id"]
    if sample == "A1":
        barcode_path = _recover_a1_barcodes(raw_dir)
    else:
        barcode_path = raw_dir / config["barcode_file"]
    barcodes = pd.Index(_read_lines(barcode_path), name="barcode")
    if barcodes.has_duplicates:
        raise ValueError(f"{barcode_path} contains duplicate barcodes")

    features_path = raw_dir / f"{prefix}-16um-features.tsv.gz"
    features = pd.read_csv(
        features_path,
        sep="\t",
        header=None,
        names=["gene_ids", "gene_name", "feature_types"],
    )
    matrix_path = raw_dir / f"{prefix}-16um-matrix.mtx.gz"
    matrix = _load_matrix(matrix_path)
    if matrix.shape != (len(barcodes), len(features)):
        raise ValueError(
            f"{sample}: matrix {matrix.shape} does not match "
            f"{len(barcodes)} barcodes and {len(features)} features"
        )

    positions_path = raw_dir / f"{prefix}-16um-tissue_positions.parquet.gz"
    positions = _read_positions(positions_path)
    selected = positions.reindex(barcodes)
    if selected.isna().any(axis=None):
        missing = int(selected.isna().any(axis=1).sum())
        raise ValueError(f"{sample}: {missing} matrix barcodes lack spatial positions")
    if not selected["in_tissue"].eq(1).all():
        raise ValueError(f"{sample}: matrix contains out-of-tissue barcodes")

    cluster_path = raw_dir / f"{prefix}-16um-clusters_graphclust.csv.gz"
    clusters = pd.read_csv(cluster_path, dtype={"Barcode": str, "Cluster": str})
    if clusters["Barcode"].duplicated().any():
        raise ValueError(f"{cluster_path} contains duplicate barcodes")
    cluster_labels = clusters.set_index("Barcode")["Cluster"].reindex(barcodes)

    obs = selected.copy()
    obs["graphclust"] = pd.Categorical(cluster_labels.fillna("Unassigned"))
    obs["sample"] = sample
    obs["geo_accession"] = config["accession"]
    obs["developmental_stage"] = "E12.5"
    obs["species"] = "Mus musculus"

    var = features[["gene_ids", "feature_types"]].copy()
    var.index = pd.Index(features["gene_name"].astype(str), name="gene_name")
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.var_names_make_unique()
    adata.obsm["spatial"] = selected[
        ["pxl_col_in_fullres", "pxl_row_in_fullres"]
    ].to_numpy(dtype=np.float64)

    scalefactors_path = raw_dir / f"{prefix}-16um-scalefactors_json.json.gz"
    with gzip.open(scalefactors_path, "rt") as handle:
        scalefactors = json.load(handle)
    hires_path = raw_dir / f"{prefix.split('_')[0]}_{sample}-tissue_hires_image.png.gz"
    lowres_path = raw_dir / f"{prefix.split('_')[0]}_{sample}-tissue_lowres_image.png.gz"
    fullres_path = raw_dir / f"{prefix.split('_')[0]}_{sample}-tissue_fullres_image.jpg.gz"
    adata.uns["spatial"] = {
        prefix: {
            "images": {
                "hires": _read_image(hires_path),
                "lowres": _read_image(lowres_path),
            },
            "scalefactors": scalefactors,
            "metadata": {
                "assay": "10x Visium HD",
                "bin_size_um": 16.0,
                "fullres_image_file": str(fullres_path.relative_to(dataset_dir)),
            },
        }
    }
    adata.uns["dataset"] = "e12_mouse_embryo"
    adata.uns["source"] = "GEO GSE291088"
    adata.uns["pla_record"] = "Zenodo 18625864"
    adata.uns["pla_pairs"] = pd.DataFrame(
        config["pla_pairs"], columns=["ligand", "receptor", "image_file"]
    )
    adata.uns["pla_pairs"]["image_file"] = adata.uns["pla_pairs"][
        "image_file"
    ].map(lambda name: f"raw/pla/{name}")
    adata.uns["source_files"] = {
        "matrix": str(matrix_path.relative_to(dataset_dir)),
        "features": str(features_path.relative_to(dataset_dir)),
        "barcodes": str(barcode_path.relative_to(dataset_dir)),
        "positions": str(positions_path.relative_to(dataset_dir)),
        "clusters": str(cluster_path.relative_to(dataset_dir)),
        "scalefactors": str(scalefactors_path.relative_to(dataset_dir)),
    }
    _add_author_metadata(adata, dataset_dir, sample)

    temporary = output.with_suffix(".h5ad.tmp")
    adata.write_h5ad(temporary, compression="gzip", compression_opts=4)
    temporary.replace(output)
    print(
        f"{sample}: {adata.n_obs:,} bins × {adata.n_vars:,} genes, "
        f"{adata.X.nnz:,} non-zero counts -> {output}",
        flush=True,
    )
    del adata, matrix
    gc.collect()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--samples", nargs="+", choices=tuple(SAMPLES), default=list(SAMPLES)
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    for sample in dict.fromkeys(args.samples):
        assemble_sample(dataset_dir, sample, args.overwrite)


if __name__ == "__main__":
    main()
