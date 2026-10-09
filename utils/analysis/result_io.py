"""Read public CellSTIC result metadata and sparse communication matrices."""
from __future__ import annotations

import json
import re
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
from scipy import sparse

try:
    from anndata.io import read_elem
except ImportError:
    from anndata.experimental import read_elem


def channel_keys(keys, metadata, *, leaf_only=True):
    """Resolve declared channels without counting raw/calibration arrays as LR pairs."""
    levels = json.loads(str(metadata.get("tree_levels_json", "[]")))
    if levels:
        levels = sorted(levels, key=lambda item: int(item["level_num"]))
        selected = levels[-1:] if leaf_only else levels
        result = {}
        for level in selected:
            for name in level["edge_type_map"]:
                safe = str(name).replace(":", "_").replace("/", "_").replace(" ", "_").strip() or "channel"
                key = f"cellstic_L{level['level_num']}_lr_{safe}"
                if key not in keys:
                    raise KeyError(f"Declared channel is absent: {key}")
                identity = name if leaf_only else (int(level["level_num"]), name)
                if key in result.values():
                    raise ValueError(f"Channel names collide at {key}")
                result[identity] = key
        return result
    matches = []
    for key in keys:
        match = re.fullmatch(r"cellstic_L(\d+)_lr_(.+)", key)
        if match and not key.endswith(("_raw", "_null_tail", "_stability")):
            matches.append((int(match[1]), match[2], key))
    if not matches:
        return {}
    last = max(level for level, _, _ in matches)
    return {(name if leaf_only else (level, name)): key
            for level, name, key in matches if not leaf_only or level == last}


def load_result(path, *, leaf_only=False, truth=False, embeddings=True):
    """Load analysis fields, leaving expression, images and auxiliary graphs on disk.

    ``X`` is intentionally absent. Use the original RNA input for gene expression.
    Each HDF5 handle is closed before returning, including on a read error.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing result {path}. Supply the result or enable the training cells.")
    with h5py.File(path, "r") as handle:
        obs = read_elem(handle["obs"])
        var = read_elem(handle["var"])
        metadata = read_elem(handle["uns/cellstic"]) if "cellstic" in handle["uns"] else {}
        uns = {"cellstic": metadata}
        for name in ("generation_config_json", "lr_candidates", "ground_truth_obsp_keys_json",
                     "ligand_receptor_map", "pla_pairs", "dataset_id", "pair_type_constraints_json"):
            if name in handle["uns"]:
                uns[name] = read_elem(handle["uns"][name])
        wanted = ("spatial", "feat", "out", "spfac", "nsfac") if embeddings else ("spatial",)
        obsm = {name: read_elem(handle["obsm"][name]) for name in wanted if name in handle["obsm"]}
        keys = channel_keys(handle["obsp"], metadata, leaf_only=leaf_only)
        obsp = {key: read_elem(handle["obsp"][key]) for key in keys.values()}
        if truth:
            for key in json.loads(str(uns.get("ground_truth_obsp_keys_json", "{}"))).values():
                obsp[key] = read_elem(handle["obsp"][key])
    if not obs.index.is_unique:
        raise ValueError(f"Duplicate observation identifiers in {path}")
    if "spatial" not in obsm or not np.isfinite(obsm["spatial"]).all():
        raise ValueError(f"Missing or nonfinite spatial coordinates in {path}")
    return ad.AnnData(obs=obs, var=var, obsm=obsm, obsp=obsp, uns=uns)


def clean_matrix(matrix, *, threshold=0.0, remove_self=False):
    """Return finite, nonnegative CSR weights with explicit zeros removed."""
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    result = sparse.csr_matrix(matrix, dtype=np.float64).copy()
    result.sum_duplicates()
    if not np.isfinite(result.data).all() or (result.data < 0).any():
        raise ValueError("Communication weights must be finite and nonnegative")
    result.data[result.data < threshold] = 0
    if remove_self:
        # Change only stored entries; setdiag(0) inserts missing diagonal slots
        # in older SciPy versions and raises SparseEfficiencyWarning.
        rows = np.repeat(np.arange(result.shape[0]), np.diff(result.indptr))
        result.data[result.indices == rows] = 0
    result.eliminate_zeros()
    return result


def lr_matrices(result, *, threshold=0.0):
    """Return sender-by-receiver matrices at the leaf level only."""
    keys = channel_keys(result.obsp, result.uns.get("cellstic", {}))
    if not keys:
        raise ValueError("No predicted LR channels in this result")
    return {name: clean_matrix(result.obsp[key], threshold=threshold) for name, key in keys.items()}
