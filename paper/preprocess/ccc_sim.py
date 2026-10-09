#!/usr/bin/env python3
"""Generate the complex scMultiSim CCC benchmark and assemble it as h5ad.

The production design contains three spatial scenarios and five fixed seeds
(15 datasets). Every dataset has paired RNA/ATAC measurements, ten parent
cell types with two states each and 25 simulated LR pairs. Each LR is enabled
for two or eight directed sender/receiver state combinations; its other
combinations are inactive. Cell-pair ground truth is stored as one sparse
``obsp`` matrix per LR pair instead of a dense cell x cell x LR array.

Default output layout::

    paper/data/ccc_sim/h5ad/
      domains_11/{rna,atac}.h5ad
      gradients_11/{rna,atac}.h5ad
      heterogeneous_11/{rna,atac}.h5ad
      ...
      manifest.csv

scMultiSim is executed with the R installation from the dedicated conda
environment.  The Python environment running this file needs anndata, numpy,
pandas and scipy.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "paper" / "data" / "ccc_sim" / "h5ad"
DEFAULT_RSCRIPT = Path("/home/wangshuai/anaconda3/envs/scmultisim/bin/Rscript")
DEFAULT_SCENARIOS = ("domains", "gradients", "heterogeneous")
DEFAULT_SEEDS = (11, 23, 37, 51, 73)
LR_DESIGN = "sparse_within_pair_on_off_v2_distinct_lr_genes"

SCENARIO_METADATA: dict[str, dict[str, Any]] = {
    "domains": {
        "noise_level": "low",
        "spatial_pattern": "discrete domains with local contact signaling",
        "capture_efficiency": 0.20,
        "ambient_fraction": 0.01,
        "mixing_fraction": 0.00,
        "coordinate_jitter_sd": 0.00,
    },
    "gradients": {
        "noise_level": "medium",
        "spatial_pattern": "layered interfaces, expression gradients and diffusible signaling",
        "capture_efficiency": 0.10,
        "ambient_fraction": 0.03,
        "mixing_fraction": 0.05,
        "coordinate_jitter_sd": 0.15,
    },
    "heterogeneous": {
        "noise_level": "high",
        "spatial_pattern": "mixed niches, rare cell states and heterogeneous signaling",
        "capture_efficiency": 0.05,
        "ambient_fraction": 0.06,
        "mixing_fraction": 0.10,
        "coordinate_jitter_sd": 0.35,
    },
}


R_GENERATOR = r'''
suppressPackageStartupMessages({
  library(scMultiSim)
  library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
out_dir <- args[[1]]
scenario <- args[[2]]
seed <- as.integer(args[[3]])
num_cells <- as.integer(args[[4]])
num_genes <- as.integer(args[[5]])
num_active_lr <- as.integer(args[[6]])
threads <- as.integer(args[[7]])
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

if (!(scenario %in% c("domains", "gradients", "heterogeneous"))) {
  stop(sprintf("Unknown scenario: %s", scenario))
}
if (num_genes < 2 * num_active_lr + 200) {
  stop("num_genes must leave at least 200 non-LR genes")
}

num_parent_types <- 10L
states_per_type <- 2L
num_states <- num_parent_types * states_per_type
state_labels <- unlist(lapply(seq_len(num_parent_types), function(i) {
  sprintf("T%02d_S%d", i, seq_len(states_per_type))
}))
clades <- vapply(seq_len(num_parent_types), function(i) {
  sprintf("(T%02d_S1:0.25,T%02d_S2:0.25):0.75", i, i)
}, character(1))
tree <- ape::read.tree(text = paste0("(", paste(clades, collapse = ","), ");"))

population_sizes <- rep(num_cells %/% num_states, num_states)
population_sizes[seq_len(num_cells %% num_states)] <-
  population_sizes[seq_len(num_cells %% num_states)] + 1L
if (scenario == "heterogeneous") {
  rare_states <- c(2L, 8L, 15L)
  rare_size <- max(3L, as.integer(round(num_cells * 0.01)))
  removed <- sum(population_sizes[rare_states] - rare_size)
  population_sizes[rare_states] <- rare_size
  common_states <- setdiff(seq_len(num_states), rare_states)
  population_sizes[common_states] <- population_sizes[common_states] + removed %/% length(common_states)
  remainder <- num_cells - sum(population_sizes)
  if (remainder > 0) {
    population_sizes[common_states[seq_len(remainder)]] <-
      population_sizes[common_states[seq_len(remainder)]] + 1L
  }
}
stopifnot(sum(population_sizes) == num_cells)

active_lr <- data.frame(
  pair_id = sprintf("LR%03d", seq_len(num_active_lr)),
  ligand = sprintf("Ligand%03d", seq_len(num_active_lr)),
  receptor = sprintf("Receptor%03d", num_active_lr + seq_len(num_active_lr)),
  effect = rep(c(3.5, 5.0, 7.0), length.out = num_active_lr),
  stringsAsFactors = FALSE
)
stopifnot(
  !any(active_lr$ligand == active_lr$receptor),
  !anyDuplicated(c(active_lr$ligand, active_lr$receptor))
)

scenario_code <- match(scenario, c("domains", "gradients", "heterogeneous"))
state_index <- function(parent_type, state) {
  (parent_type - 1L) * states_per_type + state
}
ctype <- cci_cell_type_params(
  tree, total.lr = num_active_lr, ctype.lr = 1L,
  rand = FALSE, discrete = TRUE
)
stopifnot(identical(names(ctype$type_map), state_labels))
constraints <- list()
summary_rows <- list()
for (k in seq_len(num_active_lr)) {
  sender_type <- ((k - 1L) %% num_parent_types) + 1L
  receiver_type <- ((k * 3L + scenario_code - 1L) %% num_parent_types) + 1L
  if (receiver_type == sender_type) {
    receiver_type <- (receiver_type %% num_parent_types) + 1L
  }
  if (k <= ceiling(num_active_lr / 2)) {
    sender_states <- 1L + ((k + scenario_code) %% states_per_type)
    receiver_states <- 1L + ((k + 1L) %% states_per_type)
    scope <- "state-specific"
  } else {
    sender_states <- seq_len(states_per_type)
    receiver_states <- seq_len(states_per_type)
    scope <- "type-wide"
  }
  # Add a within-state/type anchor so every simulated positive has local
  # support even when the cross-type populations are separated by the chosen
  # spatial layout.  The cross-type constraints below retain directional CCC.
  anchor_type <- ((k + scenario_code - 2L) %% num_parent_types) + 1L
  anchor_states <- if (scope == "state-specific") {
    1L + ((k + scenario_code - 1L) %% states_per_type)
  } else {
    seq_len(states_per_type)
  }
  for (sender_state in anchor_states) {
    for (receiver_state in anchor_states) {
      sender_idx <- state_index(anchor_type, sender_state)
      receiver_idx <- state_index(anchor_type, receiver_state)
      ctype$params[receiver_idx, sender_idx, k] <- 1
      constraints[[length(constraints) + 1L]] <- data.frame(
        pair_id = active_lr$pair_id[k],
        ligand = active_lr$ligand[k],
        receptor = active_lr$receptor[k],
        sender_type = sprintf("T%02d", anchor_type),
        sender_state = sprintf("T%02d_S%d", anchor_type, sender_state),
        receiver_type = sprintf("T%02d", anchor_type),
        receiver_state = sprintf("T%02d_S%d", anchor_type, receiver_state),
        stringsAsFactors = FALSE
      )
    }
  }
  for (sender_state in sender_states) {
    for (receiver_state in receiver_states) {
      sender_idx <- state_index(sender_type, sender_state)
      receiver_idx <- state_index(receiver_type, receiver_state)
      # scMultiSim indexes the focal receiver first and its ligand-producing
      # neighbour second.  Exported ground truth below uses sender x receiver.
      ctype$params[receiver_idx, sender_idx, k] <- 1
      constraints[[length(constraints) + 1L]] <- data.frame(
        pair_id = active_lr$pair_id[k],
        ligand = active_lr$ligand[k],
        receptor = active_lr$receptor[k],
        sender_type = sprintf("T%02d", sender_type),
        sender_state = sprintf("T%02d_S%d", sender_type, sender_state),
        receiver_type = sprintf("T%02d", receiver_type),
        receiver_state = sprintf("T%02d_S%d", receiver_type, receiver_state),
        stringsAsFactors = FALSE
      )
    }
  }
  enabled_contexts <- sum(ctype$params[, , k] > 0)
  expected_contexts <- if (scope == "state-specific") 2L else 8L
  if (enabled_contexts != expected_contexts) {
    stop(sprintf("LR %s has %d enabled contexts; expected %d",
                 active_lr$pair_id[k], enabled_contexts, expected_contexts))
  }
  summary_rows[[k]] <- data.frame(
    pair_id = active_lr$pair_id[k],
    ligand = active_lr$ligand[k],
    receptor = active_lr$receptor[k],
    effect = active_lr$effect[k],
    scope = scope,
    n_enabled_state_contexts = enabled_contexts,
    sender_type = sprintf("T%02d", sender_type),
    receiver_type = sprintf("T%02d", receiver_type),
    stringsAsFactors = FALSE
  )
}
constraint_table <- unique(do.call(rbind, constraints))
active_summary <- do.call(rbind, summary_rows)
stopifnot(nrow(constraint_table) == sum(active_summary$n_enabled_state_contexts))

.gradient_locs <- NULL
gradient_layout <- function(grid_size, cell_types) {
  positions <- as.matrix(expand.grid(x = seq_len(grid_size), y = seq_len(grid_size)))
  positions <- positions[seq_len(length(cell_types)), , drop = FALSE]
  position_order <- order(positions[, 1], positions[, 2])
  cell_order <- order(cell_types, runif(length(cell_types)))
  result <- matrix(NA_real_, nrow = length(cell_types), ncol = 2)
  result[cell_order, ] <- positions[position_order, , drop = FALSE]
  .gradient_locs <<- result
  result
}

gradient_effect <- function(i) {
  if (scenario != "gradients" || i != 3L) return(NULL)
  x <- .gradient_locs[, 1]
  y <- .gradient_locs[, 2]
  x <- (x - min(x)) / max(1, max(x) - min(x))
  y <- (y - min(y)) / max(1, max(y) - min(y))
  extra_cif <- cbind(x, y, abs(x - y))
  extra_giv <- matrix(0, nrow = num_genes, ncol = 3)
  first_gene <- max(2L * num_active_lr + 1L, num_genes - 149L)
  gradient_genes <- seq.int(first_gene, num_genes)
  extra_giv[gradient_genes, 1] <- rep(c(1.2, 0.8), length.out = length(gradient_genes))
  extra_giv[gradient_genes, 2] <- rep(c(0.7, 1.1, 0.5), length.out = length(gradient_genes))
  extra_giv[gradient_genes, 3] <- rep(c(0.6, 0.9), length.out = length(gradient_genes))
  list(extra_cif, extra_giv)
}

grid_size <- ceiling(sqrt(num_cells * 1.25))
if (scenario == "domains") {
  layout <- "enhanced"
  same_type_prob <- 0.95
  radius <- 1
  intrinsic_noise <- 0.35
  alpha_mean <- 0.20
  alpha_sd <- 0.025
  depth_mean <- 50000
  depth_sd <- 5000
  atac_obs_prob <- 0.40
  atac_sd_frac <- 0.20
  ambient_fraction <- 0.01
  mixing_fraction <- 0.00
  jitter_sd <- 0.00
} else if (scenario == "gradients") {
  layout <- gradient_layout
  same_type_prob <- 0.80
  radius <- "gaussian:3.0"
  intrinsic_noise <- 0.55
  alpha_mean <- 0.10
  alpha_sd <- 0.020
  depth_mean <- 30000
  depth_sd <- 6000
  atac_obs_prob <- 0.30
  atac_sd_frac <- 0.35
  ambient_fraction <- 0.03
  mixing_fraction <- 0.05
  jitter_sd <- 0.15
} else {
  layout <- "islands:2,8,15"
  same_type_prob <- 0.55
  # scMultiSim 1.6.0 accepts a labelled islands layout only with the local
  # neighbourhood radius.  Rare-state islands still create the intended
  # heterogeneous niche geometry.
  radius <- 1
  intrinsic_noise <- 0.75
  alpha_mean <- 0.05
  alpha_sd <- 0.015
  depth_mean <- 15000
  depth_sd <- 5000
  atac_obs_prob <- 0.20
  atac_sd_frac <- 0.50
  ambient_fraction <- 0.06
  mixing_fraction <- 0.10
  jitter_sd <- 0.35
}

if (num_genes >= (100L + 2L * num_active_lr)) {
  # Embed the official 100-gene regulatory module in the larger expression
  # background.  The 1,136-gene GRN makes 2,000-cell spatial simulation
  # needlessly memory intensive because parameters are expanded per cell.
  data(GRN_params_100)
  grn_input <- GRN_params_100
} else {
  # This branch makes very small interface tests preserve the requested
  # feature count.
  grn_input <- NA
}
sim_options <- list(
  rand.seed = seed,
  threads = threads,
  speed.up = TRUE,
  GRN = grn_input,
  num.genes = num_genes,
  num.cells = num_cells,
  num.cifs = 80,
  cif.sigma = 0.45,
  diff.cif.fraction = 0.80,
  intrinsic.noise = intrinsic_noise,
  tree = tree,
  discrete.cif = TRUE,
  discrete.pop.size = as.integer(population_sizes),
  do.velocity = FALSE,
  ext.cif.giv = gradient_effect,
  cci = list(
    params = active_lr[, c("receptor", "ligand", "effect")],
    cell.type.interaction = ctype,
    max.neighbors = 4L,
    grid.size = grid_size,
    layout = layout,
    same.type.prob = same_type_prob,
    radius = radius,
    start.layer = num_cells,
    static.state.len = 10L,
    del.lr.pair = FALSE,
    single.cell.gt = FALSE
  )
)

set.seed(seed)
result <- sim_true_counts(sim_options)
add_expr_noise(
  result,
  randseed = seed + 1000L,
  protocol = "UMI",
  alpha_mean = alpha_mean,
  alpha_sd = alpha_sd,
  depth_mean = depth_mean,
  depth_sd = depth_sd,
  atac.obs.prob = atac_obs_prob,
  atac.sd.frac = atac_sd_frac
)
counts_obs <- if (is.list(result$counts_obs)) result$counts_obs$counts else result$counts_obs
atac_obs <- result$atacseq_obs

add_ambient <- function(counts, fraction, random_seed) {
  if (fraction <= 0) return(counts)
  set.seed(random_seed)
  profile <- rowSums(counts) + 1
  profile <- profile / sum(profile)
  for (cell in seq_len(ncol(counts))) {
    molecules <- as.integer(round(sum(counts[, cell]) * fraction))
    if (molecules > 0) {
      counts[, cell] <- counts[, cell] + as.vector(rmultinom(1, molecules, profile))
    }
  }
  counts
}

nearest_grid_neighbour <- function(coords) {
  keys <- paste(coords[, 1], coords[, 2], sep = ":")
  lookup <- setNames(seq_len(nrow(coords)), keys)
  offsets <- matrix(c(-1, 0, 1, 0, 0, -1, 0, 1, -1, -1, -1, 1, 1, -1, 1, 1), ncol = 2, byrow = TRUE)
  neighbours <- seq_len(nrow(coords))
  for (i in seq_len(nrow(coords))) {
    candidate_keys <- paste(coords[i, 1] + offsets[, 1], coords[i, 2] + offsets[, 2], sep = ":")
    hits <- unname(lookup[candidate_keys])
    hits <- hits[!is.na(hits)]
    if (length(hits) > 0) neighbours[i] <- hits[1]
  }
  neighbours
}

set.seed(seed + 2000L)
counts_obs <- add_ambient(counts_obs, ambient_fraction, seed + 2000L)
true_coords <- result$cci_locs
if (mixing_fraction > 0) {
  nearest <- nearest_grid_neighbour(true_coords)
  counts_obs <- round(
    (1 - mixing_fraction) * counts_obs + mixing_fraction * counts_obs[, nearest, drop = FALSE]
  )
}
observed_coords <- true_coords
if (jitter_sd > 0) {
  observed_coords <- observed_coords + matrix(
    rnorm(length(observed_coords), sd = jitter_sd), ncol = 2
  )
}

# Construct the exact sparse local ground truth used by the simulator.  In
# scMultiSim, each focal receiver is influenced by ligand-producing neighbours;
# the exported matrices use the conventional sender x receiver orientation.
truth_dir <- file.path(out_dir, "truth")
dir.create(truth_dir, recursive = TRUE, showWarnings = FALSE)
state_ids <- as.integer(result$cell_meta$cell.type.idx)
for (k in seq_len(num_active_lr)) {
  senders <- integer()
  receivers <- integer()
  for (receiver in seq_len(num_cells)) {
    neighbours <- result$grid$get_neighbours(receiver)
    neighbours <- neighbours[!is.na(neighbours) & neighbours <= num_cells]
    if (length(neighbours) == 0) next
    enabled <- vapply(neighbours, function(sender) {
      ctype$params[state_ids[receiver], state_ids[sender], k] > 0
    }, logical(1))
    if (any(enabled)) {
      senders <- c(senders, neighbours[enabled])
      receivers <- c(receivers, rep(receiver, sum(enabled)))
    }
  }
  truth <- sparseMatrix(
    i = senders, j = receivers, x = rep(1, length(senders)),
    dims = c(num_cells, num_cells), giveCsparse = TRUE
  )
  if (length(senders) == 0) {
    stop(sprintf("Active pair %s has no spatially eligible cell pairs", active_lr$pair_id[k]))
  }
  writeMM(truth, file.path(truth_dir, sprintf("truth_%03d.mtx", k)))
}

write_numeric <- function(name, value) {
  value <- as.array(value)
  dimensions <- dim(value)
  if (is.null(dimensions)) dimensions <- length(value)
  writeLines(paste(dimensions, collapse = ","), file.path(out_dir, paste0(name, ".dims")))
  connection <- file(file.path(out_dir, paste0(name, ".bin")), "wb")
  writeBin(as.double(value), connection, size = 8, endian = "little")
  close(connection)
}
write_text <- function(name, value) {
  writeLines(as.character(value), file.path(out_dir, paste0(name, ".txt")), useBytes = TRUE)
}

write_numeric("counts_true", result$counts)
write_numeric("counts_observed", counts_obs)
write_numeric("atac_true", result$atac_counts)
write_numeric("atac_observed", atac_obs)
write_numeric("coordinates_true", true_coords)
write_numeric("coordinates_observed", observed_coords)
write_numeric("lr_state_mask", ctype$params)
write_text("gene_names", rownames(result$counts))
write_text("cell_names", colnames(result$counts))
write_text("atac_names", paste0("region", seq_len(nrow(result$atac_counts))))
write_text("state_names", state_labels)

cell_meta <- result$cell_meta
cell_meta$cell_id <- rownames(cell_meta)
cell_meta$cell_state <- state_labels[as.integer(cell_meta$cell.type.idx)]
cell_meta$cell_type <- sub("_S[12]$", "", cell_meta$cell_state)
cell_meta$observed_library_size <- colSums(counts_obs)
cell_meta$scenario <- scenario
cell_meta$seed <- seed
write.table(
  cell_meta, file.path(out_dir, "cell_metadata.tsv"), sep = "\t",
  quote = FALSE, row.names = FALSE
)
write.table(active_summary, file.path(out_dir, "active_lr.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
write.table(constraint_table, file.path(out_dir, "lr_constraints.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
write.table(
  data.frame(
    scenario = scenario, seed = seed, num_cells = num_cells,
    num_genes = num_genes, num_active_lr = num_active_lr,
    lr_design = "sparse_within_pair_on_off_v2_distinct_lr_genes",
    receptor_design = "single_gene",
    ligand_receptor_gene_rule = "disjoint_single_gene_pairs",
    num_parent_types = num_parent_types, states_per_type = states_per_type,
    noise_level = c(domains = "low", gradients = "medium", heterogeneous = "high")[[scenario]],
    capture_efficiency = alpha_mean, ambient_fraction = ambient_fraction,
    mixing_fraction = mixing_fraction, coordinate_jitter_sd = jitter_sd,
    scmultisim_version = as.character(packageVersion("scMultiSim")),
    r_version = R.version.string,
    stringsAsFactors = FALSE
  ),
  file.path(out_dir, "generation_metadata.tsv"), sep = "\t",
  quote = FALSE, row.names = FALSE
)
'''


def _read_lines(directory: Path, name: str) -> list[str]:
    return (directory / f"{name}.txt").read_text(encoding="utf-8").splitlines()


def _read_numeric(directory: Path, name: str) -> np.ndarray:
    dims = tuple(
        int(item)
        for item in (directory / f"{name}.dims").read_text().strip().split(",")
    )
    values = np.fromfile(directory / f"{name}.bin", dtype="<f8")
    expected = int(np.prod(dims))
    if values.size != expected:
        raise ValueError(f"{name}: found {values.size} values; expected {expected}")
    return values.reshape(dims, order="F")


def _matrix_as_csr(matrix: np.ndarray, *, integer: bool = False) -> sparse.csr_matrix:
    if integer:
        matrix = np.rint(matrix).astype(np.int32, copy=False)
    else:
        matrix = matrix.astype(np.float32, copy=False)
    result = sparse.csr_matrix(matrix)
    result.eliminate_zeros()
    return result


def _safe_truth_key(pair_id: str) -> str:
    return "ccc_gt_" + re.sub(r"[^A-Za-z0-9_]+", "_", pair_id).strip("_")


def _ligand_receptor_map(frame: pd.DataFrame) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for ligand, receptor in frame[["ligand", "receptor"]].itertuples(index=False):
        result.setdefault(str(ligand), []).append(str(receptor))
    return result


def _assemble_h5ad(
    export_dir: Path,
    output_dir: Path,
    *,
    scenario: str,
    seed: int,
    expected_active_lr: int,
    overwrite: bool,
) -> dict[str, Any]:
    rna_path = output_dir / "rna.h5ad"
    atac_path = output_dir / "atac.h5ad"
    if not overwrite and rna_path.is_file() and atac_path.is_file():
        existing = ad.read_h5ad(rna_path, backed="r")
        try:
            config = json.loads(str(existing.uns.get("generation_config_json", "{}")))
            if (
                config.get("lr_design") != LR_DESIGN
                or int(existing.uns.get("n_active_lr", -1)) != expected_active_lr
                or int(existing.uns.get("n_decoy_lr", -1)) != 0
            ):
                raise ValueError(
                    f"{rna_path} uses an older or different LR design; "
                    "regenerate it with --overwrite"
                )
            record = {
                "dataset_id": f"{scenario}_seed{seed}",
                "scenario": scenario,
                "seed": seed,
                "n_cells": existing.n_obs,
                "n_genes": existing.n_vars,
                "n_active_lr": int(existing.uns["n_active_lr"]),
                "n_decoy_lr": int(existing.uns["n_decoy_lr"]),
                "rna_path": str(rna_path),
                "atac_path": str(atac_path),
                "status": "existing",
            }
        finally:
            existing.file.close()
        return record

    gene_names = _read_lines(export_dir, "gene_names")
    cell_names = _read_lines(export_dir, "cell_names")
    atac_names = _read_lines(export_dir, "atac_names")
    state_names = _read_lines(export_dir, "state_names")
    cell_meta = pd.read_csv(export_dir / "cell_metadata.tsv", sep="\t")
    active = pd.read_csv(export_dir / "active_lr.tsv", sep="\t")
    constraints = pd.read_csv(export_dir / "lr_constraints.tsv", sep="\t")
    generation = pd.read_csv(export_dir / "generation_metadata.tsv", sep="\t").iloc[0]
    if (
        str(generation["lr_design"]) != LR_DESIGN
        or len(active) != expected_active_lr
    ):
        raise ValueError("Exported LR design does not match the requested design")

    mask_raw = _read_numeric(export_dir, "lr_state_mask")
    if (
        mask_raw.shape != (len(state_names), len(state_names), len(active))
        or not np.isin(mask_raw, (0, 1)).all()
    ):
        raise ValueError("Invalid scMultiSim LR state-interaction mask")
    lr_state_mask = mask_raw.astype(np.uint8)
    state_index = {name: index for index, name in enumerate(state_names)}
    pair_index = {
        str(pair_id): index for index, pair_id in enumerate(active["pair_id"])
    }
    if (
        len(state_index) != len(state_names)
        or len(pair_index) != len(active)
        or int(lr_state_mask.sum()) != len(constraints)
    ):
        raise ValueError("LR mask and active context table are inconsistent")
    for row in constraints.itertuples(index=False):
        if not lr_state_mask[
            state_index[str(row.receiver_state)],
            state_index[str(row.sender_state)],
            pair_index[str(row.pair_id)],
        ]:
            raise ValueError("An exported active LR context is disabled in the mask")

    counts_true = _matrix_as_csr(_read_numeric(export_dir, "counts_true").T)
    counts_observed = _matrix_as_csr(
        _read_numeric(export_dir, "counts_observed").T, integer=True
    )
    atac_true = _matrix_as_csr(_read_numeric(export_dir, "atac_true").T)
    atac_observed = _matrix_as_csr(
        _read_numeric(export_dir, "atac_observed").T, integer=True
    )
    coordinates_true = _read_numeric(export_dir, "coordinates_true")
    coordinates_observed = _read_numeric(export_dir, "coordinates_observed")

    if counts_observed.shape != (len(cell_names), len(gene_names)):
        raise ValueError("RNA dimensions do not agree with exported names")
    if atac_observed.shape != (len(cell_names), len(atac_names)):
        raise ValueError("ATAC dimensions do not agree with exported names")
    if cell_meta["cell_id"].astype(str).tolist() != cell_names:
        raise ValueError("Cell metadata and matrices use different cell orders")

    obs = cell_meta.set_index("cell_id").copy()
    obs.index = obs.index.astype(str)
    obs["cell_type"] = obs["cell_type"].astype(str)
    obs["cell_state"] = obs["cell_state"].astype(str)
    obs["cell_state_id"] = obs["cell.type.idx"].astype(int)
    obs["seed"] = int(seed)
    obs["scenario"] = scenario

    active_candidates = active.copy()
    active_candidates["active"] = True
    active_candidates["matched_to"] = active_candidates["pair_id"]
    candidate_columns = [
        "pair_id",
        "ligand",
        "receptor",
        "active",
        "effect",
        "scope",
        "n_enabled_state_contexts",
        "sender_type",
        "receiver_type",
        "matched_to",
    ]
    candidates = active_candidates[candidate_columns].copy()
    lr_genes = pd.concat(
        [candidates["ligand"], candidates["receptor"]], ignore_index=True
    ).astype(str)
    if lr_genes.duplicated().any():
        raise ValueError("Every simulated LR pair must use two distinct, unique genes")
    candidate_genes = set(lr_genes)
    if not candidate_genes <= set(gene_names):
        raise ValueError("scMultiSim did not export every configured LR gene")
    if candidates["pair_id"].duplicated().any():
        raise ValueError("Exported LR candidate IDs are not unique")

    var = pd.DataFrame(index=pd.Index(gene_names, name="gene"))
    var["role"] = "background"
    var.loc[var.index.intersection(active["ligand"]), "role"] = "active_ligand"
    var.loc[var.index.intersection(active["receptor"]), "role"] = "active_receptor"

    rna = ad.AnnData(X=counts_observed, obs=obs.copy(), var=var)
    rna.layers["counts"] = counts_observed.copy()
    rna.layers["true_counts"] = counts_true
    rna.obsm["spatial"] = coordinates_observed.astype(np.float32)
    rna.obsm["spatial_true"] = coordinates_true.astype(np.float32)

    truth_keys: dict[str, str] = {}
    for index, row in active.reset_index(drop=True).iterrows():
        truth_path = export_dir / "truth" / f"truth_{index + 1:03d}.mtx"
        truth = sparse.csr_matrix(mmread(truth_path), dtype=np.uint8)
        truth.eliminate_zeros()
        if truth.nnz == 0:
            raise ValueError(f"Active LR {row['pair_id']} has no truth edges")
        key = _safe_truth_key(str(row["pair_id"]))
        rna.obsp[key] = truth
        truth_keys[str(row["pair_id"])] = key

    pair_constraints: dict[str, list[list[str]]] = {}
    for pair_id, table in constraints.groupby("pair_id", sort=False):
        pair_constraints[str(pair_id)] = table[
            ["sender_state", "receiver_state"]
        ].astype(str).values.tolist()

    config = {
        "generator": "scMultiSim",
        "scmultisim_version": str(generation["scmultisim_version"]),
        "r_version": str(generation["r_version"]),
        "scenario": scenario,
        "seed": int(seed),
        "noise_level": str(generation["noise_level"]),
        "spatial_pattern": SCENARIO_METADATA[scenario]["spatial_pattern"],
        "num_parent_cell_types": 10,
        "states_per_cell_type": 2,
        "capture_efficiency": float(generation["capture_efficiency"]),
        "ambient_fraction": float(generation["ambient_fraction"]),
        "mixing_fraction": float(generation["mixing_fraction"]),
        "coordinate_jitter_sd": float(generation["coordinate_jitter_sd"]),
        "ground_truth_orientation": "sender_by_receiver",
        "ground_truth_storage": "one sparse rna.obsp matrix per LR pair",
        "lr_design": str(generation["lr_design"]),
        "receptor_design": str(generation["receptor_design"]),
        "ligand_receptor_gene_rule": str(
            generation["ligand_receptor_gene_rule"]
        ),
        "negative_definition": "disabled sender/receiver state contexts within each LR",
        "lr_state_mask_orientation": "receiver_state_by_sender_state_by_lr",
    }
    shared_uns = {
        "dataset": "ccc_sim",
        "dataset_id": f"{scenario}_seed{seed}",
        "scenario": scenario,
        "seed": int(seed),
        "n_active_lr": int(len(active)),
        "n_decoy_lr": 0,
        "ligand_receptor_map": _ligand_receptor_map(candidates),
        "edge_type_map": {
            str(pair): int(index)
            for index, pair in enumerate(candidates["pair_id"].astype(str))
        },
        "generation_config_json": json.dumps(config, sort_keys=True),
        "lr_candidates": candidates,
        "active_lr_constraints": constraints,
        "lr_state_mask": lr_state_mask,
        "lr_state_names": np.asarray(state_names, dtype=str),
        "pair_type_constraints_json": json.dumps(pair_constraints, sort_keys=True),
    }
    rna.uns.update(shared_uns)
    rna.uns["ground_truth_obsp_keys_json"] = json.dumps(truth_keys, sort_keys=True)

    atac_var = pd.DataFrame(index=pd.Index(atac_names, name="region"))
    atac = ad.AnnData(X=atac_observed, obs=obs.copy(), var=atac_var)
    atac.layers["counts"] = atac_observed.copy()
    atac.layers["true_counts"] = atac_true
    atac.obsm["spatial"] = coordinates_observed.astype(np.float32)
    atac.obsm["spatial_true"] = coordinates_true.astype(np.float32)
    atac.uns.update(shared_uns)
    atac.uns["rna_ground_truth_file"] = "rna.h5ad"

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_rna = output_dir / ".rna.h5ad.tmp"
    temporary_atac = output_dir / ".atac.h5ad.tmp"
    try:
        rna.write_h5ad(temporary_rna, compression="gzip")
        atac.write_h5ad(temporary_atac, compression="gzip")
        temporary_rna.replace(rna_path)
        temporary_atac.replace(atac_path)
    finally:
        temporary_rna.unlink(missing_ok=True)
        temporary_atac.unlink(missing_ok=True)

    return {
        "dataset_id": f"{scenario}_seed{seed}",
        "scenario": scenario,
        "seed": int(seed),
        "noise_level": config["noise_level"],
        "n_cells": int(rna.n_obs),
        "n_genes": int(rna.n_vars),
        "n_atac_regions": int(atac.n_vars),
        "n_cell_types": int(obs["cell_type"].nunique()),
        "n_cell_states": int(obs["cell_state"].nunique()),
        "n_active_lr": int(len(active)),
        "n_decoy_lr": 0,
        "n_truth_edges": int(sum(rna.obsp[key].nnz for key in truth_keys.values())),
        "rna_path": str(rna_path),
        "atac_path": str(atac_path),
        "status": "generated",
    }


def generate_one(
    *,
    rscript: Path,
    output_root: Path,
    scenario: str,
    seed: int,
    num_cells: int,
    num_genes: int,
    active_lr: int,
    r_threads: int,
    overwrite: bool,
) -> dict[str, Any]:
    output_dir = output_root / f"{scenario}_{seed}"
    if not overwrite and (output_dir / "rna.h5ad").is_file() and (
        output_dir / "atac.h5ad"
    ).is_file():
        return _assemble_h5ad(
            Path(),
            output_dir,
            scenario=scenario,
            seed=seed,
            expected_active_lr=active_lr,
            overwrite=False,
        )

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=f"ccc-sim-{scenario}-{seed}-") as name:
        temp_dir = Path(name)
        r_file = temp_dir / "generate.R"
        export_dir = temp_dir / "export"
        r_file.write_text(R_GENERATOR, encoding="utf-8")
        command = [
            str(rscript),
            "--vanilla",
            str(r_file),
            str(export_dir),
            scenario,
            str(seed),
            str(num_cells),
            str(num_genes),
            str(active_lr),
            str(r_threads),
        ]
        log_path = output_dir / "generation.log"
        output_dir.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        log_path.write_text(
            "$ " + " ".join(command) + "\n\n[stdout]\n" + completed.stdout
            + "\n[stderr]\n" + completed.stderr,
            encoding="utf-8",
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"scMultiSim failed for {scenario}_{seed}; see {log_path}\n"
                + "\n".join(completed.stderr.splitlines()[-20:])
            )
        record = _assemble_h5ad(
            export_dir,
            output_dir,
            scenario=scenario,
            seed=seed,
            expected_active_lr=active_lr,
            overwrite=True,
        )
    record["elapsed_minutes"] = round((time.monotonic() - started) / 60.0, 3)
    return record


def _validate_args(args: argparse.Namespace) -> None:
    if not args.rscript.is_file() or not os.access(args.rscript, os.X_OK):
        raise FileNotFoundError(f"Executable Rscript not found: {args.rscript}")
    if args.num_cells < 20:
        raise ValueError("--num-cells must be at least 20")
    lr_genes = 2 * args.active_lr
    if args.num_genes < lr_genes + 200:
        raise ValueError(
            "--num-genes must accommodate every LR gene plus 200 background genes"
        )
    if args.active_lr < 1:
        raise ValueError("--active-lr must be positive")
    if args.jobs < 1 or args.r_threads < 1:
        raise ValueError("--jobs and --r-threads must be positive")


def _write_design(output_root: Path, args: argparse.Namespace) -> None:
    design = {
        "scenarios": list(args.scenarios),
        "seeds": list(args.seeds),
        "number_of_datasets": len(args.scenarios) * len(args.seeds),
        "num_cells": args.num_cells,
        "num_genes": args.num_genes,
        "num_parent_cell_types": 10,
        "states_per_cell_type": 2,
        "active_lr_pairs": args.active_lr,
        "dedicated_inactive_lr_pairs": 0,
        "lr_design": LR_DESIGN,
        "receptor_design": "single_gene",
        "ligand_receptor_gene_rule": "disjoint_single_gene_pairs",
        "enabled_state_contexts_per_pair": {
            "state_specific": 2,
            "type_wide": 8,
        },
        "negative_definition": "disabled sender/receiver state contexts within each LR",
        "scenario_metadata": {
            name: SCENARIO_METADATA[name] for name in args.scenarios
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "design.json").write_text(
        json.dumps(design, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--rscript", type=Path, default=DEFAULT_RSCRIPT)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=DEFAULT_SCENARIOS,
        default=list(DEFAULT_SCENARIOS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--num-cells", type=int, default=2000)
    parser.add_argument("--num-genes", type=int, default=2000)
    parser.add_argument("--active-lr", type=int, default=25)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of datasets generated concurrently (default: 1).",
    )
    parser.add_argument(
        "--r-threads",
        type=int,
        default=1,
        help="scMultiSim threads per dataset (default: 1).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.rscript = args.rscript.expanduser().resolve()
    args.scenarios = list(dict.fromkeys(args.scenarios))
    args.seeds = list(dict.fromkeys(args.seeds))
    _validate_args(args)
    tasks = [
        {"scenario": scenario, "seed": seed}
        for scenario in args.scenarios
        for seed in args.seeds
    ]
    print(
        f"Generating {len(tasks)} datasets in {args.output_dir} "
        f"with {args.jobs} concurrent job(s)"
    )
    common = {
        "rscript": args.rscript,
        "output_root": args.output_dir,
        "num_cells": args.num_cells,
        "num_genes": args.num_genes,
        "active_lr": args.active_lr,
        "r_threads": args.r_threads,
        "overwrite": args.overwrite,
    }
    records: list[dict[str, Any]] = []
    if args.jobs == 1:
        for index, task in enumerate(tasks, start=1):
            print(f"[{index}/{len(tasks)}] {task['scenario']} seed={task['seed']}")
            record = generate_one(**common, **task)
            records.append(record)
            print(
                f"  {record['status']}: {record['n_cells']} cells, "
                f"{record['n_genes']} genes, {record['n_active_lr']} active LR"
            )
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(generate_one, **common, **task): task for task in tasks
            }
            for future in as_completed(futures):
                task = futures[future]
                record = future.result()
                records.append(record)
                print(
                    f"[done] {task['scenario']} seed={task['seed']}: "
                    f"{record['status']}"
                )

    manifest = pd.DataFrame(records).sort_values(["scenario", "seed"])
    _write_design(args.output_dir, args)
    manifest_path = args.output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
