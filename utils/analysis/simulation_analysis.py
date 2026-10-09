"""Sparse truth comparisons and declared state constraints for simulated CCC."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import average_precision_score, roc_auc_score

from .result_io import channel_keys, clean_matrix, load_result, lr_matrices


def constrain_simulated_result(result):
    """Apply the simulation's declared state-pair prior and update ancestor scores.

    This uses known simulation states. Scores evaluated afterwards are conditional
    on this prior and must not be described as a blind recovery benchmark.
    The supplied AnnData is modified in memory; the caller decides where to save it.
    """
    constraints = json.loads(str(result.uns["pair_type_constraints_json"]))
    states = result.obs["cell_state"].astype(str).to_numpy()
    metadata = result.uns["cellstic"]
    levels = sorted(json.loads(str(metadata["tree_levels_json"])), key=lambda v: int(v["level_num"]))
    hierarchy = json.loads(str(metadata["hierarchy_dict_json"]))
    keys = channel_keys(result.obsp, metadata)
    for row in result.uns["lr_candidates"].itertuples(index=False):
        pair = f"{row.ligand}:{row.receptor}"
        matrix = clean_matrix(result.obsp[keys[pair]]).tocoo()
        keep = np.zeros(matrix.nnz, dtype=bool)
        for sender, receiver in constraints[str(row.pair_id)]:
            keep |= (states[matrix.row] == str(sender)) & (states[matrix.col] == str(receiver))
        result.obsp[keys[pair]] = sparse.csr_matrix(
            (matrix.data[keep], (matrix.row[keep], matrix.col[keep])), shape=matrix.shape)
    all_keys = channel_keys(result.obsp, metadata, leaf_only=False)
    for level in levels[:-1]:
        for group in level["edge_type_map"]:
            members = hierarchy[str(level["level_key"])][group]
            members = members if isinstance(members, list) else [members]
            descendants = {m["edge_type_name"] for m in members if isinstance(m, dict) and "edge_type_name" in m}
            if not descendants or not descendants.issubset(keys):
                raise ValueError(f"Invalid leaf membership for {group}")
            total = sum((result.obsp[keys[name]] for name in descendants),
                        sparse.csr_matrix((result.n_obs, result.n_obs)))
            result.obsp[all_keys[(int(level["level_num"]), group)]] = total / len(keys)
    metadata["pair_type_constraints_postprocessed"] = True
    metadata["pair_type_constraints_annotation_key"] = "cell_state"
    return result


def pair_metrics(prediction, truth, *, threshold=.5):
    """Evaluate all ordered, non-self pairs using an exact compressed zero class.

    Unstored pairs are score-zero negatives, not dropped observations. Truth is
    binarized before conversion so fractional positive weights remain positive.
    """
    prediction = clean_matrix(prediction, remove_self=True)
    truth = clean_matrix(truth, remove_self=True)
    if prediction.shape != truth.shape or prediction.shape[0] != prediction.shape[1]:
        raise ValueError("Prediction and truth must have equal square shapes")
    truth.data[:] = 1
    n = prediction.shape[0]
    if n < 2:
        raise ValueError("At least two observations are required")
    a, b = prediction.tocoo(), truth.tocoo()
    union = np.union1d(a.row.astype(np.int64) * n + a.col, b.row.astype(np.int64) * n + b.col)
    rows, cols = union // n, union % n
    scores = np.asarray(prediction[rows, cols]).ravel()
    labels = np.asarray(truth[rows, cols]).ravel().astype(bool)
    weights = np.ones(len(union))
    remainder = n * (n - 1) - len(union)
    if remainder:
        scores, labels, weights = np.append(scores, 0), np.append(labels, False), np.append(weights, remainder)
    if not 0 < threshold <= 1:
        raise ValueError("Classification threshold must be in (0, 1]")
    predicted = scores >= threshold
    tp, fp = weights[predicted & labels].sum(), weights[predicted & ~labels].sum()
    fn = weights[~predicted & labels].sum()
    return dict(auroc=roc_auc_score(labels, scores, sample_weight=weights) if len(np.unique(labels)) == 2 else np.nan,
                auprc=average_precision_score(labels, scores, sample_weight=weights) if labels.any() else np.nan,
                f1=2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.,
                n_positive=int(weights[labels].sum()), n_evaluated=int(weights.sum()))


def evaluate_simulations(paths, *, threshold=.5):
    """Return one measured row per dataset and LR channel, including its design."""
    rows = []
    for path in paths:
        result = load_result(path, leaf_only=True, truth=True, embeddings=False)
        design = json.loads(str(result.uns["generation_config_json"]))
        truth_keys = json.loads(str(result.uns["ground_truth_obsp_keys_json"]))
        matrices = lr_matrices(result)
        for candidate in result.uns["lr_candidates"].itertuples(index=False):
            pair = f"{candidate.ligand}:{candidate.receptor}"
            truth = result.obsp[truth_keys[str(candidate.pair_id)]]
            rows.append(dict(dataset=result.uns["dataset_id"], lr_pair=pair,
                             lr_design=design["lr_design"],
                             state_prior=bool(result.uns["cellstic"].get("pair_type_constraints_postprocessed", False)),
                             **pair_metrics(matrices[pair], truth, threshold=threshold)))
    return pd.DataFrame(rows)
