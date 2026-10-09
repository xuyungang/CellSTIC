"""Descriptive calibration of leaf-level CCC interaction scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
from scipy import sparse

from model.graph.sparse_graph import SparseTypedGraph


@dataclass(frozen=True)
class CCCPostprocessResult:
    """Sparse descriptive calibration outputs for leaf-level CCC scores."""

    null_tail_probabilities: List[sparse.csr_matrix]
    stability: Optional[List[sparse.csr_matrix]]
    selected_scores: List[sparse.csr_matrix]
    null_cutoffs: np.ndarray
    summary: List[Dict[str, object]]


class CCCPostprocessor:
    """Compare CCC scores with in-sample expression-negative candidates.

    The resulting tail proportions are descriptive calibration quantities, not
    hypothesis-test p-values, and no frequentist FDR guarantee is claimed.
    """

    @staticmethod
    def _validate_inputs(
        score_matrices: Sequence[sparse.spmatrix],
        channel_names: Sequence[str],
        candidate_pairs: np.ndarray,
        n_cells: int,
        *,
        validate_pairs: bool = True,
    ) -> np.ndarray:
        if len(score_matrices) != len(channel_names):
            raise ValueError("score_matrices and channel_names must have equal length")
        if len(set(channel_names)) != len(channel_names):
            raise ValueError("channel_names must be unique")

        candidates = np.asarray(candidate_pairs, dtype=np.int64)
        if candidates.size == 0:
            candidates = np.empty((0, 2), dtype=np.int64)
        elif candidates.ndim != 2 or candidates.shape[1] != 2:
            raise ValueError(
                "candidate_pairs must have shape (n_candidates, 2), got "
                f"{candidates.shape}"
            )
        if validate_pairs:
            if candidates.size and (
                candidates.min() < 0 or candidates.max() >= n_cells
            ):
                raise ValueError(
                    f"candidate_pairs contains an index outside [0, {n_cells})"
                )
            if candidates.size and np.any(candidates[:, 0] == candidates[:, 1]):
                raise ValueError("candidate_pairs must not contain self-edges")
            if candidates.size and len(np.unique(candidates, axis=0)) != len(
                candidates
            ):
                raise ValueError("candidate_pairs must not contain duplicate edges")

        expected_shape = (n_cells, n_cells)
        for index, matrix in enumerate(score_matrices):
            if not sparse.issparse(matrix):
                raise TypeError(f"score_matrices[{index}] must be sparse")
            if matrix.shape != expected_shape:
                raise ValueError(
                    f"score_matrices[{index}] must have shape {expected_shape}, "
                    f"got {matrix.shape}"
                )
            if not np.all(np.isfinite(matrix.data)):
                raise ValueError(
                    f"score_matrices[{index}] contains non-finite candidate scores"
                )
        return candidates

    @staticmethod
    def _candidate_values(
        matrix: sparse.spmatrix, candidate_pairs: np.ndarray
    ) -> np.ndarray:
        if len(candidate_pairs) == 0:
            return np.empty(0, dtype=np.float64)
        return (
            np.asarray(matrix.tocsr()[candidate_pairs[:, 0], candidate_pairs[:, 1]])
            .reshape(-1)
            .astype(np.float64, copy=False)
        )

    @staticmethod
    def _as_candidate_matrix(
        values: np.ndarray, candidate_pairs: np.ndarray, n_cells: int
    ) -> sparse.csr_matrix:
        matrix = sparse.csr_matrix(
            (
                np.asarray(values, dtype=np.float32),
                (candidate_pairs[:, 0], candidate_pairs[:, 1]),
            ),
            shape=(n_cells, n_cells),
        )
        matrix.eliminate_zeros()
        return matrix

    @staticmethod
    def _null_tail_probabilities(
        scores: np.ndarray, null_scores: np.ndarray
    ) -> np.ndarray:
        """Return finite-sample upper-tail proportions in the reference scores."""
        ordered = np.sort(np.asarray(null_scores, dtype=np.float64))
        exceedances = len(ordered) - np.searchsorted(ordered, scores, side="left")
        return (exceedances + 1.0) / (len(ordered) + 1.0)

    @staticmethod
    def null_cutoffs(
        score_matrices: Sequence[sparse.spmatrix],
        channel_names: Sequence[str],
        base_graph: SparseTypedGraph,
        base_edge_type_map: Mapping[str, int],
        candidate_pairs: np.ndarray,
        *,
        tail_fraction: float,
        min_null_edges: int,
    ) -> np.ndarray:
        """Get per-LR descriptive reference cutoffs for stability estimation."""
        if not 0.0 < tail_fraction < 1.0:
            raise ValueError("tail_fraction must be in (0, 1)")
        if not isinstance(min_null_edges, (int, np.integer)) or min_null_edges < 1:
            raise ValueError("min_null_edges must be a positive integer")
        candidates = CCCPostprocessor._validate_inputs(
            score_matrices,
            channel_names,
            candidate_pairs,
            base_graph.n_nodes,
        )
        cutoffs = np.full(len(channel_names), np.inf, dtype=np.float64)
        for channel_index, name in enumerate(channel_names):
            base_index = base_edge_type_map.get(name)
            if base_index is None or not 0 <= base_index < len(base_graph):
                continue
            scores = CCCPostprocessor._candidate_values(
                score_matrices[channel_index], candidates
            )
            expression_events = (
                CCCPostprocessor._candidate_values(base_graph[base_index], candidates)
                > 0
            )
            null_scores = scores[~expression_events]
            if len(null_scores) >= min_null_edges:
                cutoffs[channel_index] = float(
                    np.quantile(null_scores, 1.0 - tail_fraction)
                )
        return cutoffs

    @staticmethod
    def calibrate(
        score_matrices: Sequence[sparse.spmatrix],
        channel_names: Sequence[str],
        base_graph: SparseTypedGraph,
        base_edge_type_map: Mapping[str, int],
        candidate_pairs: np.ndarray,
        *,
        tail_fraction: float,
        min_null_edges: int,
        stability_matrices: Optional[Sequence[sparse.spmatrix]] = None,
        stability_threshold: float = 0.8,
        precomputed_null_cutoffs: Optional[np.ndarray] = None,
    ) -> CCCPostprocessResult:
        """Compute descriptive null tails, reference cutoffs, and selected edges."""
        if not 0.0 < tail_fraction < 1.0:
            raise ValueError("tail_fraction must be in (0, 1)")
        if not isinstance(min_null_edges, (int, np.integer)) or min_null_edges < 1:
            raise ValueError("min_null_edges must be a positive integer")
        if not 0.0 <= stability_threshold <= 1.0:
            raise ValueError("stability_threshold must be in [0, 1]")
        n_cells = base_graph.n_nodes
        candidates = CCCPostprocessor._validate_inputs(
            score_matrices, channel_names, candidate_pairs, n_cells
        )
        if stability_matrices is not None:
            CCCPostprocessor._validate_inputs(
                stability_matrices,
                channel_names,
                candidates,
                n_cells,
                validate_pairs=False,
            )
        per_channel_scores: List[np.ndarray] = []
        per_channel_tail: List[np.ndarray] = []
        null_counts: List[int] = []
        computed_cutoffs = np.full(len(channel_names), np.inf, dtype=np.float64)

        for channel_index, name in enumerate(channel_names):
            scores = CCCPostprocessor._candidate_values(
                score_matrices[channel_index], candidates
            )
            base_index = base_edge_type_map.get(name)
            expression_events = (
                CCCPostprocessor._candidate_values(base_graph[base_index], candidates)
                > 0
                if base_index is not None and 0 <= base_index < len(base_graph)
                else np.zeros(len(candidates), dtype=bool)
            )
            null_scores = scores[~expression_events]
            null_counts.append(len(null_scores))
            tail_probabilities = np.ones(len(scores), dtype=np.float64)
            if len(null_scores) >= min_null_edges:
                tail_probabilities = CCCPostprocessor._null_tail_probabilities(
                    scores, null_scores
                )
                computed_cutoffs[channel_index] = float(
                    np.quantile(null_scores, 1.0 - tail_fraction)
                )
            per_channel_scores.append(scores)
            per_channel_tail.append(tail_probabilities)
        if precomputed_null_cutoffs is None:
            cutoffs = computed_cutoffs
        else:
            cutoffs = np.asarray(precomputed_null_cutoffs, dtype=np.float64).reshape(-1)
            if len(cutoffs) != len(channel_names):
                raise ValueError(
                    "precomputed_null_cutoffs and channel_names must have "
                    "equal length"
                )
            if np.any(np.isnan(cutoffs)) or np.any(np.isneginf(cutoffs)):
                raise ValueError(
                    "precomputed_null_cutoffs may contain finite values or "
                    "positive infinity only"
                )

        tail_matrices: List[sparse.csr_matrix] = []
        output_stability: List[sparse.csr_matrix] = []
        selected_matrices: List[sparse.csr_matrix] = []
        summary: List[Dict[str, object]] = []
        for channel_index, name in enumerate(channel_names):
            scores = per_channel_scores[channel_index]
            tail_probabilities = per_channel_tail[channel_index]
            passes_tail_threshold = tail_probabilities <= tail_fraction
            if stability_matrices is None:
                stability = None
                selected = passes_tail_threshold
            else:
                stability = CCCPostprocessor._candidate_values(
                    stability_matrices[channel_index], candidates
                )
                selected = passes_tail_threshold & (
                    stability >= stability_threshold
                )
            selected_scores = np.where(selected, scores, 0.0)
            tail_matrices.append(
                CCCPostprocessor._as_candidate_matrix(
                    tail_probabilities, candidates, n_cells
                )
            )
            if stability is not None:
                output_stability.append(
                    CCCPostprocessor._as_candidate_matrix(
                        stability, candidates, n_cells
                    )
                )
            selected_matrices.append(
                CCCPostprocessor._as_candidate_matrix(
                    selected_scores, candidates, n_cells
                )
            )
            summary.append(
                {
                    "lr_pair": str(name),
                    "n_candidates": int(len(candidates)),
                    "n_null": int(null_counts[channel_index]),
                    "null_cutoff": (
                        float(cutoffs[channel_index])
                        if np.isfinite(cutoffs[channel_index])
                        else None
                    ),
                    "n_below_tail_threshold": int(
                        np.count_nonzero(passes_tail_threshold)
                    ),
                    "n_selected": int(np.count_nonzero(selected)),
                }
            )

        return CCCPostprocessResult(
            null_tail_probabilities=tail_matrices,
            stability=(output_stability if stability_matrices is not None else None),
            selected_scores=selected_matrices,
            null_cutoffs=cutoffs,
            summary=summary,
        )


__all__ = ["CCCPostprocessResult", "CCCPostprocessor"]
