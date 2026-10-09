#!/usr/bin/env python3
"""
ArchR peak / gene-score matrix generation helper.

This module provides a small, reusable Python API for running an ArchR-based
pipeline on ATAC fragments files and exporting PeakMatrix / GeneScoreMatrix
as RDS files (optionally converted to CSV).

Typical usage:

    from utils.data.peak_count import run_peak_count

    run_peak_count(
        fragments_file="sample.fragments.tsv.gz",
        output_dir="output_dir",
        sample_name="sample1",
        genome="mm10",
        add_peak_mat=True,
        macs2_path="/path/to/macs2",
        convert_to_csv=True,
    )
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_PANDAS = False
    pd = None  # type: ignore


def create_archr_r_script(
    fragments_file: str,
    output_dir: str,
    sample_name: Optional[str] = None,
    genome: str = "mm10",
    threads: int = 8,
    tile_size: int = 5000,
    add_tile_mat: bool = True,
    add_gene_score_mat: bool = True,
    add_peak_mat: bool = False,
    macs2_path: Optional[str] = None,
    min_frags: int = 0,
    max_frags: int = int(1e7),
    filter_tss: float = 0.0,
    filter_frags: float = 0.0,
) -> str:
    """
    Create an R script that uses ArchR to process a fragments.tsv.gz file.

    The script will:
      1. Create Arrow files and an ArchR project.
      2. Optionally run MACS2 to generate a PeakMatrix.
      3. Optionally export GeneScoreMatrix.

    When add_peak_mat is True, TileMatrix is always created during Arrow creation
    (i.e. addTileMat=TRUE in createArrowFiles) so that LSI/clustering and peak
    calling follow the same code path as the standalone script for consistent results.

    Returns:
        Path to the generated R script.
    """
    if sample_name is None:
        sample_name = (
            Path(fragments_file)
            .stem.replace(".fragments", "")
            .replace(".tsv", "")
            .replace(".gz", "")
        )

    fragments_file = str(Path(fragments_file).absolute())
    output_dir = str(Path(output_dir).absolute())
    macs2_path = macs2_path or ""

    # When generating PeakMatrix we need TileMatrix for LSI/clustering; create it during
    # Arrow creation so results match ArchR's default path (same as standalone script).
    effective_add_tile_mat = add_tile_mat or add_peak_mat

    script_content = f"""
####### Process fragments files using ArchR and generate count matrices

library(ArchR)
library(BSgenome.Mmusculus.UCSC.mm10)

########## ArchR project creation
threads = {threads}
addArchRThreads(threads = threads)

addArchRGenome("{genome}")

# Set working directory
setwd("{output_dir}")

inputFiles <- "{fragments_file}"
sampleNames <- "{sample_name}"

# Create Arrow files
cat("Creating Arrow files...\\n")
ArrowFiles <- createArrowFiles(
  inputFiles = inputFiles,
  sampleNames = sampleNames,
  filterTSS = {filter_tss},
  filterFrags = {filter_frags},
  minFrags = {min_frags},
  maxFrags = {max_frags},
  addTileMat = {str(effective_add_tile_mat).upper()},
  addGeneScoreMat = {str(add_gene_score_mat).upper()},
  offsetPlus = 0,
  offsetMinus = 0,
  TileMatParams = list(tileSize = {tile_size})
)

cat("Arrow files created:", ArrowFiles, "\\n")

# Create ArchR project
cat("Creating ArchR project...\\n")
proj <- ArchRProject(
  ArrowFiles = ArrowFiles,
  outputDirectory = sampleNames,
  copyArrows = TRUE
)

cat("ArchR project created\\n")
cat("Project info:\\n")
print(proj)

# Save project
saveArchRProject(ArchRProj = proj, outputDirectory = paste0("Save-", sampleNames), load = FALSE)
cat("Project saved to:", paste0("Save-", sampleNames), "\\n")

# Generate matrices (PeakMatrix and/or GeneScoreMatrix)
cat("Preparing to generate matrices...\\n")

# Generate PeakMatrix (if needed)
if ({str(add_peak_mat).upper()}) {{
  if (is.null("{macs2_path}") || "{macs2_path}" == "" || "{macs2_path}" == "None") {{
    cat("Warning: MACS2 path required to generate PeakMatrix\\n")
  }} else {{
    tryCatch({{
      cat("Starting PeakMatrix generation...\\n")

      # Ensure TileMatrix exists
      if (!"TileMatrix" %in% getAvailableMatrices(proj)) {{
        cat("Creating TileMatrix...\\n")
        proj <- addTileMatrix(proj, tileSize = {tile_size})
      }}

      # Dimensionality reduction and clustering
      if (!"IterativeLSI" %in% names(proj@reducedDims)) {{
        cat("Performing dimensionality reduction (IterativeLSI)...\\n")
        proj <- addIterativeLSI(
          ArchRProj = proj,
          useMatrix = "TileMatrix",
          name = "IterativeLSI",
          iterations = 2,
          clusterParams = list(
            resolution = c(0.2),
            sampleCells = 10000,
            n.start = 10
          ),
          varFeatures = 25000,
          dimsToUse = 1:30,
          force = TRUE
        )
      }}

      if (!"Clusters" %in% names(proj@cellColData)) {{
        cat("Performing clustering...\\n")
        proj <- addClusters(
          input = proj,
          reducedDims = "IterativeLSI",
          method = "Seurat",
          name = "Clusters",
          resolution = 1,
          force = TRUE
        )
      }}

      # Add group coverages
      cat("Adding group coverages...\\n")
      proj <- addGroupCoverages(ArchRProj = proj, groupBy = "Clusters")

      # Call peaks using MACS2
      cat("Calling peaks using MACS2...\\n")
      proj <- addReproduciblePeakSet(
        ArchRProj = proj,
        groupBy = "Clusters",
        pathToMacs2 = "{macs2_path}",
        force = TRUE
      )

      # Add PeakMatrix
      cat("Adding PeakMatrix...\\n")
      proj <- addPeakMatrix(proj)

      # Save updated project
      saveArchRProject(ArchRProj = proj, outputDirectory = paste0("Save-", sampleNames), load = FALSE, overwrite = TRUE)

      # Export PeakMatrix
      cat("Exporting PeakMatrix...\\n")
      peak_matrix <- getMatrixFromProject(
        ArchRProj = proj,
        useMatrix = "PeakMatrix",
        useSeqnames = NULL,
        verbose = TRUE,
        binarize = FALSE
      )

      peak_matrix_sparse <- assay(peak_matrix)
      peak_set <- getPeakSet(proj)
      peak_names <- paste(seqnames(peak_set), ranges(peak_set), sep = "-")
      cell_names <- colnames(peak_matrix)

      peak_matrix_sparse <- t(peak_matrix_sparse)
      rownames(peak_matrix_sparse) <- cell_names
      colnames(peak_matrix_sparse) <- peak_names

      saveRDS(peak_matrix_sparse, file = paste0(sampleNames, "_PeakMatrix.rds"))
      cat("PeakMatrix saved to:", paste0(sampleNames, "_PeakMatrix.rds"), "\\n")
      cat("Matrix format: cells × peaks (", nrow(peak_matrix_sparse), " cells × ", ncol(peak_matrix_sparse), " peaks)\\n")

      peak_info <- data.frame(
        peak_id = peak_names,
        chrom = as.character(seqnames(peak_set)),
        start = start(peak_set),
        end = end(peak_set)
      )
      write.csv(peak_info, file = paste0(sampleNames, "_peaks_info.csv"), row.names = FALSE)

      matrix_info <- data.frame(
        matrix_type = "PeakMatrix",
        n_cells = nrow(peak_matrix_sparse),
        n_peaks = ncol(peak_matrix_sparse),
        format = "cells_x_peaks",
        file = paste0(sampleNames, "_PeakMatrix.rds")
      )
      write.csv(matrix_info, file = paste0(sampleNames, "_PeakMatrix_info.csv"), row.names = FALSE)

      cat("PeakMatrix generation completed!\\n")
    }}, error = function(e) {{
      cat("Error generating PeakMatrix:", conditionMessage(e), "\\n")
    }})
  }}
}}

# Generate and export GeneScoreMatrix (if needed)
if ({str(add_gene_score_mat).upper()}) {{
  tryCatch({{
    cat("Starting GeneScoreMatrix generation...\\n")

    if (!"GeneScoreMatrix" %in% getAvailableMatrices(proj)) {{
      cat("GeneScoreMatrix not found in project. Attempting to add GeneScoreMatrix...\\n")
      proj <- addGeneScoreMatrix(proj)
    }}

    gene_score_matrix <- getMatrixFromProject(
      ArchRProj = proj,
      useMatrix = "GeneScoreMatrix",
      useSeqnames = NULL,
      verbose = TRUE,
      binarize = FALSE
    )

    gene_score_matrix_sparse <- assay(gene_score_matrix)
    gene_names <- rownames(gene_score_matrix)
    cell_names <- colnames(gene_score_matrix)

    gene_score_matrix_sparse <- t(gene_score_matrix_sparse)
    rownames(gene_score_matrix_sparse) <- cell_names
    colnames(gene_score_matrix_sparse) <- gene_names

    saveRDS(gene_score_matrix_sparse, file = paste0(sampleNames, "_GeneScoreMatrix.rds"))
    cat("GeneScoreMatrix saved to:", paste0(sampleNames, "_GeneScoreMatrix.rds"), "\\n")
    cat("Matrix format: cells × genes (", nrow(gene_score_matrix_sparse), " cells × ", ncol(gene_score_matrix_sparse), " genes)\\n")

    matrix_info <- data.frame(
      matrix_type = "GeneScoreMatrix",
      n_cells = nrow(gene_score_matrix_sparse),
      n_genes = ncol(gene_score_matrix_sparse),
      format = "cells_x_genes",
      file = paste0(sampleNames, "_GeneScoreMatrix.rds")
    )
    write.csv(matrix_info, file = paste0(sampleNames, "_GeneScoreMatrix_info.csv"), row.names = FALSE)

    cat("GeneScoreMatrix generation completed!\\n")
  }}, error = function(e) {{
    cat("Error generating GeneScoreMatrix:", conditionMessage(e), "\\n")
  }})
}}

# Export cell metadata
cat("Exporting cell metadata...\\n")
cell_metadata <- as.data.frame(getCellColData(ArchRProj = proj))
write.csv(cell_metadata, file = paste0(sampleNames, "_cell_metadata.csv"), row.names = TRUE)

cat("Processing completed!\\n")
"""

    script_path = os.path.join(output_dir, f"archr_process_{sample_name}.R")
    with open(script_path, "w") as f:
        f.write(script_content)
    return script_path


def run_archr_processing(
    fragments_file: str,
    output_dir: str,
    sample_name: Optional[str] = None,
    genome: str = "mm10",
    threads: int = 8,
    tile_size: int = 5000,
    add_tile_mat: bool = True,
    add_gene_score_mat: bool = True,
    add_peak_mat: bool = False,
    macs2_path: Optional[str] = None,
    r_home: Optional[str] = None,
    rscript_path: Optional[str] = None,
    min_frags: int = 0,
    max_frags: int = int(1e7),
    filter_tss: float = 0.0,
    filter_frags: float = 0.0,
) -> None:
    """
    Run the ArchR processing pipeline by generating and executing an R script.
    """
    fragments_path = Path(fragments_file)
    if not fragments_path.exists():
        raise FileNotFoundError(f"Fragments file not found: {fragments_file}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if add_peak_mat and not macs2_path:
        raise ValueError("MACS2 path (macs2_path) required to generate PeakMatrix")

    print("Creating ArchR R script...")
    r_script = create_archr_r_script(
        fragments_file=str(fragments_path),
        output_dir=str(output_path),
        sample_name=sample_name,
        genome=genome,
        threads=threads,
        tile_size=tile_size,
        add_tile_mat=add_tile_mat,
        add_gene_score_mat=add_gene_score_mat,
        add_peak_mat=add_peak_mat,
        macs2_path=macs2_path,
        min_frags=min_frags,
        max_frags=max_frags,
        filter_tss=filter_tss,
        filter_frags=filter_frags,
    )

    # Determine Rscript path
    if rscript_path:
        rscript = rscript_path
    elif r_home:
        rscript = os.path.join(r_home, "bin", "Rscript")
        if not os.path.exists(rscript):
            raise FileNotFoundError(f"Rscript not found: {rscript}")
    else:
        rscript = "Rscript"

    # Environment variables
    env = os.environ.copy()
    if r_home:
        env["R_HOME"] = r_home
        r_bin = os.path.join(r_home, "bin")
        if "PATH" in env:
            env["PATH"] = f"{r_bin}:{env['PATH']}"
        else:
            env["PATH"] = r_bin

    print("Running ArchR processing pipeline...")
    print(f"Using Rscript: {rscript}")
    print(f"R script: {r_script}")
    print("-" * 80)

    process = subprocess.Popen(
        [rscript, r_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=str(output_path),
        bufsize=1,
        universal_newlines=True,
    )

    stdout_lines = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        stdout_lines.append(line)

    returncode = process.wait()
    stdout = "".join(stdout_lines)

    if returncode != 0:
        raise RuntimeError(
            f"ArchR processing failed, return code: {returncode}\nOutput: {stdout}"
        )

    print("ArchR processing completed!")
    print(f"Output directory: {output_path}")


def convert_rds_to_csv(
    rds_file: str,
    output_csv: Optional[str] = None,
    r_home: Optional[str] = None,
    rscript_path: Optional[str] = None,
) -> Optional[str]:
    """
    Convert an RDS matrix file to CSV using R.

    Assumes the matrix is cells × features, with row names as cell barcodes.
    Returns the CSV path or None if conversion failed.
    """
    rds_file = str(rds_file)
    if output_csv is None:
        output_csv = rds_file.replace(".rds", ".csv")

    r_script = f"""
library(Matrix)
matrix_data <- readRDS("{rds_file}")
matrix_dense <- as.matrix(matrix_data)
write.csv(matrix_dense, file = "{output_csv}", quote = FALSE)
cat("Matrix converted to CSV:", "{output_csv}", "\\n")
cat("Matrix format: cells × features (", nrow(matrix_dense), " cells × ", ncol(matrix_dense), " features)\\n")
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as f:
        f.write(r_script)
        temp_script = f.name

    # Determine Rscript path
    if rscript_path:
        rscript = rscript_path
    elif r_home:
        rscript = os.path.join(r_home, "bin", "Rscript")
        if not os.path.exists(rscript):
            raise FileNotFoundError(f"Rscript not found: {rscript}")
    else:
        rscript = "Rscript"

    env = os.environ.copy()
    if r_home:
        env["R_HOME"] = r_home
        r_bin = os.path.join(r_home, "bin")
        if "PATH" in env:
            env["PATH"] = f"{r_bin}:{env['PATH']}"
        else:
            env["PATH"] = r_bin

    try:
        result = subprocess.run(
            [rscript, temp_script],
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            print(f"Conversion failed: {result.stderr}")
            if result.stdout:
                print(f"Output: {result.stdout}")
            return None

        print(f"Matrix converted to: {output_csv}")
        if result.stdout:
            print(result.stdout)
        return output_csv
    finally:
        try:
            os.unlink(temp_script)
        except OSError:
            pass


def _clean_cell_name_prefix(csv_file: str, sample_name: str) -> None:
    """
    Remove the prefix '<sample_name>#' from cell barcodes in a CSV file (if pandas is available).
    """
    if not HAS_PANDAS:
        print("Warning: pandas not available, skipping cell name prefix removal")
        return

    path = Path(csv_file)
    if not path.exists():
        print(f"CSV file not found, skip cleaning: {csv_file}")
        return

    df = pd.read_csv(path, index_col=0)  # type: ignore[arg-type]
    prefix = f"{sample_name}#"
    new_index = []
    for cell_name in df.index:
        if isinstance(cell_name, str) and cell_name.startswith(prefix):
            new_index.append(cell_name[len(prefix) :])
        else:
            new_index.append(cell_name)
    df.index = new_index
    df.to_csv(path)
    print(f"Updated CSV with cleaned cell names: {csv_file}")


def process_fragments(
    fragments_file: str,
    output_dir: str,
    sample_name: Optional[str] = None,
    genome: str = "mm10",
    threads: int = 8,
    tile_size: int = 5000,
    add_tile_mat: bool = True,
    add_gene_score_mat: bool = True,
    add_peak_mat: bool = False,
    macs2_path: Optional[str] = None,
    r_home: Optional[str] = None,
    rscript_path: Optional[str] = None,
    convert_to_csv: bool = False,
) -> None:
    """
    High-level helper:
      1. Run the ArchR pipeline via R.
      2. Optionally convert PeakMatrix / GeneScoreMatrix RDS to CSV.
      3. Optionally clean cell name prefixes in CSV.
    """
    run_archr_processing(
        fragments_file=fragments_file,
        output_dir=output_dir,
        sample_name=sample_name,
        genome=genome,
        threads=threads,
        tile_size=tile_size,
        add_tile_mat=add_tile_mat,
        add_gene_score_mat=add_gene_score_mat,
        add_peak_mat=add_peak_mat,
        macs2_path=macs2_path,
        r_home=r_home,
        rscript_path=rscript_path,
    )

    if not convert_to_csv:
        return

    output_path = Path(output_dir)
    if sample_name is None:
        sample_name = (
            Path(fragments_file)
            .stem.replace(".fragments", "")
            .replace(".tsv", "")
            .replace(".gz", "")
        )

    # PeakMatrix
    if add_peak_mat:
        rds_file = output_path / f"{sample_name}_PeakMatrix.rds"
        if rds_file.exists():
            print("Converting PeakMatrix to CSV...")
            csv_file = convert_rds_to_csv(
                str(rds_file), r_home=r_home, rscript_path=rscript_path
            )
            if csv_file and Path(csv_file).exists():
                print("Cleaning cell names in PeakMatrix CSV...")
                _clean_cell_name_prefix(csv_file, sample_name)

    # GeneScoreMatrix
    if add_gene_score_mat:
        rds_file = output_path / f"{sample_name}_GeneScoreMatrix.rds"
        if rds_file.exists():
            print("Converting GeneScoreMatrix to CSV...")
            csv_file = convert_rds_to_csv(
                str(rds_file), r_home=r_home, rscript_path=rscript_path
            )
            if csv_file and Path(csv_file).exists():
                print("Cleaning cell names in GeneScoreMatrix CSV...")
                _clean_cell_name_prefix(csv_file, sample_name)


def run_peak_count(
    fragments_file: str,
    output_dir: str,
    sample_name: Optional[str] = None,
    genome: str = "mm10",
    threads: int = 8,
    tile_size: int = 5000,
    add_tile_mat: bool = True,
    add_gene_score_mat: bool = True,
    add_peak_mat: bool = False,
    macs2_path: Optional[str] = None,
    r_home: Optional[str] = None,
    rscript_path: Optional[str] = None,
    convert_to_csv: bool = False,
) -> None:
    """
    Public API for external callers.

    This is a thin wrapper around `process_fragments` with the same arguments.
    """
    process_fragments(
        fragments_file=fragments_file,
        output_dir=output_dir,
        sample_name=sample_name,
        genome=genome,
        threads=threads,
        tile_size=tile_size,
        add_tile_mat=add_tile_mat,
        add_gene_score_mat=add_gene_score_mat,
        add_peak_mat=add_peak_mat,
        macs2_path=macs2_path,
        r_home=r_home,
        rscript_path=rscript_path,
        convert_to_csv=convert_to_csv,
    )
