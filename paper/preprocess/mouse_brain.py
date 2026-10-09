"""Finalize the five mouse-brain modalities under ``paper/data/mouse_brain``.

Existing h5ad files supply the registered spots and metadata. ``--peak-count``
rebuilds the three chromatin assays from raw fragments, using narrow peaks for
ATAC/H3K27ac and broad domains for H3K27me3, then installs the NEW count matrices.
Counts are uncapped fragment overlaps, not capped ATAC insertion counts.
Old chromatin h5ad files are backed up in the timestamped count workspace.
RNA/ADT and trained CellSTIC out.h5ad are not rewritten by this operation.

Peak-count details
------------------
Uncapped fragment-overlap counts for the mouse-brain chromatin assays.

MACS2 BEDPE consumes the first three columns of the deduplicated fragment file:
the fifth column is PCR support, NOT an independent molecule multiplier.
ATAC/H3K27ac use narrow peaks with their called boundaries; H3K27me3 uses broad
domains. Unlike ArchR's insertion-centred, 501-bp/ceiling=4 default, these count
each fragment once per overlapping feature and retain the integer raw counts.
Peak discovery pools the original assay; no RNA labels or spatial smoothing
are used. Peaks on non-primary chromosomes or overlapping the mm10 blacklist
are excluded. All coordinates in this module are BED, zero-based half-open.

Sources: https://www.archrproject.com/reference/addPeakMatrix.html
https://www.archrproject.com/reference/addReproduciblePeakSet.html
https://macs3-project.github.io/MACS/docs/callpeak.html
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

DEFAULT_DATASET_DIR = PROJECT_ROOT / "paper" / "data" / "mouse_brain"
MODALITIES = ("rna", "adt", "atac", "h3k27ac", "h3k27me3")
CELL_TYPE_TABLE = (
    "rna_10xWholeMouseBrain(CCN20230722)_CorrelationMapping_UTC_1769068788450.csv"
)
LIGAND_RECEPTOR_PAIRS = (
    ("Col6a3", "Sdc4"),
    ("Cd200", "Cd200r4"),
    ("Penk", "Oprk1"),
)


PRIMARY_CHROMS = tuple([f"chr{i}" for i in range(1, 20)] + ["chrX"])
DEFAULT_MACS2 = Path("/home/wangshuai/anaconda3/envs/spaGem/bin/macs2")
DEFAULT_RSCRIPT = Path("/home/wangshuai/anaconda3/envs/R4.3.1-3/bin/Rscript")
FRAGMENTS = {
    "atac": "GSM8494157_5M_20um_ATAC.fragments.tsv.gz",
    "h3k27ac": "GSM8494157_5M_20um_H3K27ac.fragments.tsv.gz",
    "h3k27me3": "GSM8494157_5M_20um_H3K27me3.fragments.tsv.gz",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merge_intervals(table: pd.DataFrame) -> pd.DataFrame:
    """Union overlapping features; touching half-open intervals remain distinct."""
    rows = []
    for chrom in PRIMARY_CHROMS:
        group = table.loc[table.chrom.eq(chrom)].sort_values(["start", "end"])
        current = None
        for start, end in group[["start", "end"]].itertuples(index=False, name=None):
            start, end = int(start), int(end)
            if start < 0 or end <= start:
                raise ValueError(f"Invalid BED interval: {chrom}:{start}-{end}")
            if current is not None and start < current[1]:
                current[1] = max(current[1], end)
            else:
                if current is not None:
                    rows.append((chrom, *current))
                current = [start, end]
        if current is not None:
            rows.append((chrom, *current))
    return pd.DataFrame(rows, columns=["chrom", "start", "end"])


def overlap_bounds(starts, ends, features):
    """Feature range intersecting each fragment, for sorted disjoint features."""
    left = np.searchsorted(features.end.to_numpy(), starts, side="right")
    right = np.searchsorted(features.start.to_numpy(), ends, side="left")
    return left, right


def export_blacklist(output: Path, rscript: Path) -> Path:
    """Reuse the same mm10 blacklist as the installed ArchR; convert 1-based R."""
    if not rscript.is_file():
        raise FileNotFoundError(f"Rscript not found: {rscript}; supply --blacklist-bed instead")
    expression = '''
suppressPackageStartupMessages(library(ArchR))
addArchRGenome("mm10")
b <- getGenomeAnnotation()$blacklist
write.table(data.frame(chrom=as.character(seqnames(b)), start=start(b)-1L, end=end(b)),
            file=commandArgs(trailingOnly=TRUE)[1], sep="\\t", quote=FALSE,
            row.names=FALSE, col.names=FALSE)
'''
    result = subprocess.run([str(rscript), "-e", expression, str(output)],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Failed to export the mm10 blacklist:\n{result.stderr}")
    return output


def call_peaks(fragments: Path, output: Path, modality: str, macs2: Path,
               blacklist: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    output.mkdir(parents=True, exist_ok=True)
    command = [str(macs2), "callpeak", "--treatment", str(fragments), "--format", "BEDPE",
               "--gsize", "1.87e9", "--name", modality, "--outdir", str(output),
               "--keep-dup", "all", "--nomodel", "--nolambda", "--qvalue", "0.01"]
    if modality == "h3k27me3":
        command += ["--broad", "--broad-cutoff", "0.1"]
    print(f"[peak_count] {modality}: calling {'broad' if modality == 'h3k27me3' else 'narrow'} peaks", flush=True)
    with (output / "macs2.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    suffix = "broadPeak" if modality == "h3k27me3" else "narrowPeak"
    peaks = pd.read_csv(output / f"{modality}_peaks.{suffix}", sep="\t", header=None,
                        usecols=[0, 1, 2], names=["chrom", "start", "end"])
    peaks = merge_intervals(peaks)
    keep = np.ones(len(peaks), dtype=bool)
    for chrom, group in peaks.groupby("chrom", sort=False):
        bad = blacklist.loc[blacklist.chrom.eq(chrom)]
        left, right = overlap_bounds(group.start.to_numpy(), group.end.to_numpy(), bad)
        keep[group.index] = left == right
    peaks = peaks.loc[keep].reset_index(drop=True)
    if peaks.empty:
        raise ValueError(f"No peaks survived primary-chromosome/blacklist filtering: {modality}")
    return peaks, command


def legacy_peak_intervals(adata: ad.AnnData) -> pd.DataFrame:
    """Existing peak IDs were exported from ArchR GRanges (one-based closed)."""
    rows = []
    bed = adata.uns.get("peak_count", {}).get("coordinate_system") == "0-based-half-open"
    for label in adata.var_names:
        match = re.fullmatch(r"([^:]+?)[-:](\d+)-(\d+)", label)
        if match is None:
            raise ValueError(f"Not a genomic peak identifier: {label}")
        chrom, start, end = match.groups()
        rows.append((chrom, int(start) if bed else int(start) - 1, int(end)))
    return merge_intervals(pd.DataFrame(rows, columns=["chrom", "start", "end"]))


def count_fragments(fragments: Path, peaks: pd.DataFrame, ids: pd.Index,
                    old_peaks: pd.DataFrame, chunksize: int = 500_000):
    """Sparse uncapped overlap counting and comparable old/new fragment FRiP."""
    if not ids.is_unique:
        raise ValueError("Duplicate spot IDs in the reference modality")
    peak_groups = {c: g for c, g in peaks.groupby("chrom", sort=False)}
    old_groups = {c: g for c, g in old_peaks.groupby("chrom", sort=False)}
    total = np.zeros(len(ids), dtype=np.int64)
    in_new = np.zeros(len(ids), dtype=np.int64)
    in_old = np.zeros(len(ids), dtype=np.int64)
    all_rows, all_cols = [], []
    reader = pd.read_csv(fragments, sep="\t", header=None, comment="#",
                         usecols=[0, 1, 2, 3], names=["chrom", "start", "end", "barcode"],
                         dtype={"chrom": str, "start": np.int64, "end": np.int64, "barcode": str},
                         chunksize=chunksize)
    for block in reader:
        row_indices = ids.get_indexer(block.barcode)
        valid = (row_indices >= 0) & block.chrom.isin(PRIMARY_CHROMS).to_numpy()
        block = block.loc[valid].copy()
        block["row"] = row_indices[valid]
        if (block.start < 0).any() or (block.end <= block.start).any():
            raise ValueError(f"Malformed fragment coordinates in {fragments}")
        total += np.bincount(block.row, minlength=len(ids))
        for chrom, group in block.groupby("chrom", sort=False):
            starts, ends = group.start.to_numpy(), group.end.to_numpy()
            rows = group.row.to_numpy(dtype=np.int32)
            for groups, hit_counts, collect in ((peak_groups, in_new, True), (old_groups, in_old, False)):
                feature_group = groups.get(chrom)
                if feature_group is None:
                    continue
                left, right = overlap_bounds(starts, ends, feature_group)
                hit_counts += np.bincount(rows[right > left], minlength=len(ids))
                if collect:
                    for offset in range(int((right - left).max(initial=0))):
                        hit = left + offset < right
                        all_rows.append(rows[hit])
                        all_cols.append(feature_group.index.to_numpy(dtype=np.int32)[left[hit] + offset])
    if not all_rows:
        raise ValueError(f"No reference spots/fragments overlap called peaks: {fragments}")
    rows, cols = np.concatenate(all_rows), np.concatenate(all_cols)
    matrix = sparse.coo_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)),
                               shape=(len(ids), len(peaks))).tocsr()
    matrix.sum_duplicates()
    return matrix, total, in_old, in_new


def rebuild_one(modality: str, dataset: Path, source: Path, output: Path,
                macs2: Path, blacklist: pd.DataFrame, blacklist_hash: str):
    old = ad.read_h5ad(source / f"{modality}.h5ad")
    fragments = dataset / "raw" / modality / FRAGMENTS[modality]
    workdir = output / modality
    peaks, command = call_peaks(fragments, workdir, modality, macs2, blacklist)
    print(f"[peak_count] {modality}: counting {len(peaks):,} peak regions", flush=True)
    matrix, total, in_old, in_new = count_fragments(fragments, peaks, old.obs_names,
                                                  legacy_peak_intervals(old))
    nonzero = np.asarray(matrix.sum(axis=0)).ravel() > 0
    matrix, peaks = matrix[:, nonzero].tocsr(), peaks.loc[nonzero].reset_index(drop=True)
    peaks["width"] = peaks.end - peaks.start
    peaks.index = pd.Index([f"{r.chrom}:{r.start}-{r.end}" for r in peaks.itertuples()], name="peak_id")
    new = ad.AnnData(matrix, obs=old.obs.copy(), var=peaks.copy(), uns=dict(old.uns))
    new.layers["counts"] = matrix.copy()
    new.obsm["spatial"] = old.obsm["spatial"].copy()
    new.obs["n_fragments"] = total
    new.obs["fragments_in_peaks"] = in_new
    new.obs["frip"] = np.divide(in_new, total, out=np.zeros(len(total), dtype=float), where=total > 0)
    new.obs["n_detected_peaks"] = matrix.getnnz(axis=1)
    new.uns["feature_type"] = "peaks"
    provenance = {
        "version": "fragment_overlap_v2", "modality": modality,
        "coordinate_system": "0-based-half-open", "count_unit": "deduplicated_fragment_overlap",
        "peak_mode": "broad" if modality == "h3k27me3" else "narrow",
        "count_ceiling": "none", "fifth_column": "PCR support ignored",
        "genome": "mm10", "qvalue": 0.01, "broad_cutoff": 0.1 if modality == "h3k27me3" else "not_applicable",
        "discovery": "pooled raw fragments; no cell-type labels or spatial coordinates",
        "limitations": "No matched input/control; these peak calls are not independent biological validation",
        "fragments_path": str(fragments), "fragments_sha256": sha256(fragments),
        "previous_h5ad_sha256": sha256(source / f"{modality}.h5ad"),
        "blacklist_sha256": blacklist_hash, "macs2_command": json.dumps(command),
    }
    new.uns["peak_count"] = provenance
    new.write_h5ad(output / f"{modality}.h5ad", compression="gzip")
    peaks.to_csv(workdir / "peak_regions.tsv", sep="\t")
    old_counts = sparse.csr_matrix(old.layers.get("counts", old.X))
    qc = pd.DataFrame({
        "cell_id": old.obs_names, "n_fragments": total,
        "old_regions_fragment_frip": np.divide(in_old, total, out=np.zeros(len(total), dtype=float), where=total > 0),
        "new_regions_fragment_frip": new.obs.frip.to_numpy(),
        "old_detected_peaks": old_counts.getnnz(axis=1),
        "new_detected_peaks": matrix.getnnz(axis=1),
    })
    qc.to_csv(workdir / "count_qc.csv", index=False)
    summary = dict(modality=modality, n_spots=new.n_obs, old_n_peaks=old.n_vars, new_n_peaks=new.n_vars,
                   old_median_detected_peaks=float(qc.old_detected_peaks.median()),
                   new_median_detected_peaks=float(qc.new_detected_peaks.median()),
                   old_median_fragment_frip=float(qc.old_regions_fragment_frip.median()),
                   new_median_fragment_frip=float(qc.new_regions_fragment_frip.median()),
                   old_zero_spots=int((qc.old_detected_peaks == 0).sum()),
                   new_zero_spots=int((qc.new_detected_peaks == 0).sum()),
                   median_peak_width=float(peaks.width.median()),
                   max_raw_count=int(matrix.max()))
    (workdir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"[peak_count] {modality}: {summary}", flush=True)
    return summary


def rebuild_chromatin(dataset_dir: Path, *, source_dir: Path | None = None,
                       output_dir: Path | None = None, macs2: Path = DEFAULT_MACS2,
                       rscript: Path = DEFAULT_RSCRIPT, blacklist_bed: Path | None = None,
                       jobs: int = 3) -> Path:
    dataset = Path(dataset_dir).resolve()
    source = Path(source_dir).resolve() if source_dir is not None else dataset / "h5ad"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = Path(output_dir).resolve() if output_dir else dataset / "count" / "peak_count_v2" / stamp
    macs2 = Path(macs2).resolve()
    if not macs2.is_file():
        raise FileNotFoundError(macs2)
    if jobs < 1 or jobs > 3:
        raise ValueError("Use 1–3 parallel modality jobs")
    for modality, filename in FRAGMENTS.items():
        for path in (source / f"{modality}.h5ad", dataset / "raw" / modality / filename):
            if not path.is_file():
                raise FileNotFoundError(path)
    # Every run has a new directory. Existing counts and previous runs are never
    # overwritten; the caller installs all three completed h5ad files together.
    output.mkdir(parents=True, exist_ok=False)
    backup = output / "previous_h5ad"
    backup.mkdir()
    for modality in FRAGMENTS:
        shutil.copy2(source / f"{modality}.h5ad", backup / f"{modality}.h5ad")
    if blacklist_bed is None:
        blacklist_bed = export_blacklist(output / "mm10_blacklist.bed", Path(rscript))
    else:
        blacklist_bed = Path(blacklist_bed).resolve()
        shutil.copy2(blacklist_bed, output / "mm10_blacklist.bed")
    blacklist = merge_intervals(pd.read_csv(blacklist_bed, sep="\t", header=None, comment="#",
                                            usecols=[0, 1, 2], names=["chrom", "start", "end"]))
    if blacklist.empty:
        raise ValueError("The mm10 blacklist is empty")
    blacklist_hash = sha256(blacklist_bed)
    version = subprocess.run([str(macs2), "--version"], capture_output=True, text=True, check=True).stdout.strip()
    print(f"[peak_count] {version}; output and recoverable backups: {output}", flush=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(rebuild_one, modality, dataset, source, output, macs2, blacklist, blacklist_hash)
                   for modality in FRAGMENTS]
        summaries = [future.result() for future in futures]
    (output / "summary.json").write_text(json.dumps({"macs2_version": version, "modalities": summaries}, indent=2) + "\n")
    return output


def peak_count(dataset_dir: Path, **kwargs) -> Path:
    """Rebuild count matrices and return their directory for immediate assembly."""
    return rebuild_chromatin(dataset_dir, **kwargs)


def _attach_dataset_metadata(modalities: List[ad.AnnData]) -> None:
    """Attach the fixed mouse-brain LR panel without a CSV dependency."""
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
        "dataset": "mouse_brain",
        "ligand_receptor_map": ligand_receptor_map,
        "lr_table": lr_table,
    }
    for adata in modalities:
        adata.uns.update(metadata)


def _attach_mouse_brain_cell_types(
    modalities: List[ad.AnnData], dataset_dir: Path
) -> None:
    """Annotate every modality from the curated CCN subclass mapping table.

    The table is keyed by the shared cell barcode, so the same RNA-derived
    ``subclass_name`` label is assigned to RNA, ADT, and chromatin modalities.
    """
    annotation_path = dataset_dir / "raw" / "type" / CELL_TYPE_TABLE
    if not annotation_path.is_file():
        raise FileNotFoundError(
            f"Mouse-brain cell-type mapping table is required: {annotation_path}"
        )

    annotation = pd.read_csv(
        annotation_path,
        encoding="utf-8-sig",
        dtype={"cell_id": str, "subclass_name": str},
    )
    required_columns = {"cell_id", "subclass_name"}
    missing_columns = required_columns.difference(annotation.columns)
    if missing_columns:
        raise ValueError(
            f"{annotation_path} is missing columns: {sorted(missing_columns)}"
        )
    if annotation["cell_id"].duplicated().any():
        duplicates = annotation.loc[annotation["cell_id"].duplicated(), "cell_id"].head(
            5
        )
        raise ValueError(
            f"{annotation_path} contains duplicate cell IDs, e.g. "
            f"{duplicates.tolist()}"
        )
    cell_types = annotation.set_index("cell_id")["subclass_name"]
    for adata in modalities:
        aligned = cell_types.reindex(adata.obs_names)
        if aligned.isna().any():
            missing = aligned.index[aligned.isna()].tolist()[:5]
            raise ValueError(
                f"Cell-type annotations are missing for {aligned.isna().sum()} "
                f"cells, e.g. {missing}"
            )
        adata.obs["cell_type"] = pd.Categorical(aligned)


def _validate_modality(name: str, adata: ad.AnnData, source: Path) -> None:
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError(f"{source} is empty: shape={adata.shape}")
    if not adata.obs_names.is_unique:
        raise ValueError(f"{source} contains duplicate cell identifiers")
    if not adata.var_names.is_unique:
        raise ValueError(f"{source} contains duplicate feature identifiers")
    if name == "rna":
        genes = {gene for pair in LIGAND_RECEPTOR_PAIRS for gene in pair}
        missing = sorted(genes.difference(adata.var_names))
        if missing:
            raise ValueError(f"{source} is missing LR genes: {missing}")


def _shared_spatial(modalities: List[ad.AnnData]) -> np.ndarray:
    spatial = next(
        (
            np.asarray(adata.obsm["spatial"], dtype=np.float32)
            for adata in modalities
            if "spatial" in adata.obsm
        ),
        None,
    )
    if spatial is None:
        raise KeyError("Mouse-brain inputs contain no obsm['spatial']")
    expected_shape = (modalities[0].n_obs, 2)
    if spatial.shape != expected_shape:
        raise ValueError(
            f"obsm['spatial'] must have shape {expected_shape}, got {spatial.shape}"
        )
    if not np.isfinite(spatial).all():
        raise ValueError("obsm['spatial'] contains non-finite coordinates")
    for adata in modalities:
        if "spatial" in adata.obsm and not np.array_equal(
            np.asarray(adata.obsm["spatial"], dtype=np.float32), spatial
        ):
            raise ValueError("Spatial coordinates disagree after barcode alignment")
    return spatial


def assemble(
    dataset_dir: Path,
    *,
    source_h5ad_dir: Path | None = None,
    chromatin_h5ad_dir: Path | None = None,
    write: bool = True,
) -> Tuple[ad.AnnData, ...]:
    """Align five assembled count modalities and write canonical ``h5ad/`` files."""
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    source_dir = (
        Path(source_h5ad_dir).expanduser().resolve()
        if source_h5ad_dir is not None
        else dataset_dir / "h5ad"
    )
    source_modalities = []
    for name in MODALITIES:
        directory = (Path(chromatin_h5ad_dir) if chromatin_h5ad_dir is not None
                     and name in {"atac", "h3k27ac", "h3k27me3"} else source_dir)
        source = directory / f"{name}.h5ad"
        if not source.is_file():
            raise FileNotFoundError(source)
        adata = ad.read_h5ad(source)
        _validate_modality(name, adata, source)
        source_modalities.append(adata)

    common_cells = source_modalities[0].obs_names
    for adata in source_modalities[1:]:
        common_cells = common_cells.intersection(adata.obs_names, sort=False)
    if len(common_cells) == 0:
        raise ValueError("Mouse-brain modalities do not share cell identifiers")
    if chromatin_h5ad_dir is not None:
        for adata in source_modalities:
            if not adata.obs_names.equals(common_cells):
                raise ValueError("Rebuilt matrices must preserve all original spot IDs and order")

    modalities = [adata[common_cells].copy() for adata in source_modalities]
    spatial = _shared_spatial(modalities)
    for name, adata in zip(MODALITIES, modalities):
        if "counts" not in adata.layers:
            adata.layers["counts"] = adata.X.copy()
        adata.obsm["spatial"] = spatial.copy()
        adata.uns["modality"] = name
        if name in {"atac", "h3k27ac", "h3k27me3"}:
            adata.uns["feature_type"] = "peaks"

    _attach_mouse_brain_cell_types(modalities, dataset_dir)
    _attach_dataset_metadata(modalities)

    output_dir = dataset_dir / "h5ad"
    if write:
        output_dir.mkdir(parents=True, exist_ok=True)
        staged = []
        for name, adata in zip(MODALITIES, modalities):
            if chromatin_h5ad_dir is not None and name in {"rna", "adt"}:
                continue
            output = output_dir / f"{name}.h5ad"
            temporary = output.with_suffix(".h5ad.tmp")
            adata.write_h5ad(temporary, compression="gzip", compression_opts=4)
            staged.append((temporary, output))
        # Complete every serialization before replacing any canonical input.
        for temporary, output in staged:
            temporary.replace(output)
    action = "validated" if not write else "wrote"
    print(
        f"mouse_brain: {len(modalities)} modalities, "
        f"{modalities[0].n_obs:,} shared cells, RNA {modalities[0].shape}; "
        f"{action} {output_dir}"
    )
    return tuple(modalities)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
    )
    parser.add_argument(
        "--source-h5ad-dir",
        type=Path,
        default=None,
        help="Optional source directory when initial count h5ad files live elsewhere.",
    )
    parser.add_argument(
        "--peak-count",
        action="store_true",
        help="Rebuild assay-specific fragment counts and install the new chromatin h5ad files.",
    )
    parser.add_argument("--peak-count-output-dir", type=Path, default=None,
                        help="New empty workspace; default is count/peak_count_v2/<timestamp>.")
    parser.add_argument("--chromatin-h5ad-dir", type=Path, default=None,
                        help="Install/validate already rebuilt chromatin h5ad files without recounting.")
    parser.add_argument("--macs2-path", type=Path, default=DEFAULT_MACS2)
    parser.add_argument("--rscript-path", type=Path, default=DEFAULT_RSCRIPT)
    parser.add_argument("--blacklist-bed", type=Path, default=None,
                        help="Optional mm10 zero-based BED blacklist; otherwise use the installed ArchR blacklist.")
    parser.add_argument("--peak-count-jobs", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate alignment and metadata without rewriting h5ad files.",
    )
    args = parser.parse_args()
    if args.peak_count and (args.check_only or args.chromatin_h5ad_dir is not None):
        parser.error("--peak-count cannot be combined with --check-only or --chromatin-h5ad-dir")
    chromatin_dir = args.chromatin_h5ad_dir
    if args.peak_count:
        chromatin_dir = peak_count(
            args.dataset_dir, source_dir=args.source_h5ad_dir,
            output_dir=args.peak_count_output_dir, macs2=args.macs2_path,
            rscript=args.rscript_path, blacklist_bed=args.blacklist_bed,
            jobs=args.peak_count_jobs,
        )
    assemble(
        args.dataset_dir,
        source_h5ad_dir=args.source_h5ad_dir,
        chromatin_h5ad_dir=chromatin_dir,
        write=not args.check_only,
    )


if __name__ == "__main__":
    main()
