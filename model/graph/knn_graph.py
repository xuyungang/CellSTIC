"""KNN-based graph construction."""

import numpy as np
from scipy import sparse


class KNNGraphUtils:
    """Build KNN adjacency from features (cosine similarity)."""

    @staticmethod
    def build_knn_graph(
        features: np.ndarray,
        top_k: int,
        candidate_pairs: np.ndarray,
    ) -> sparse.csr_matrix:
        """Return a sparse cosine-kNN adjacency with self loops."""
        features = np.asarray(features)
        if features.ndim != 2 or len(features) == 0:
            raise ValueError(
                f"features must be a non-empty 2D matrix, got {features.shape}"
            )
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        n_cells = len(features)
        rows, cols, values = [], [], []
        candidates = np.asarray(candidate_pairs, dtype=np.int64).reshape(-1, 2)
        if len(candidates) and (candidates.min() < 0 or candidates.max() >= n_cells):
            raise ValueError("candidate_pairs contains an out-of-range cell index")
        norms = np.linalg.norm(features, axis=1) + 1e-8
        similarities = (
            np.einsum(
                "ij,ij->i", features[candidates[:, 0]], features[candidates[:, 1]]
            )
            / (norms[candidates[:, 0]] * norms[candidates[:, 1]])
            if len(candidates)
            else np.empty(0)
        )
        similarities = (similarities + 1.0) / 2.0
        if len(candidates) == 0:
            return sparse.eye(n_cells, format="csr", dtype=np.float32)
        order = np.argsort(candidates[:, 0], kind="stable")
        sorted_sources = candidates[order, 0]
        boundaries = np.flatnonzero(
            np.r_[True, sorted_sources[1:] != sorted_sources[:-1], True]
        )
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            source = sorted_sources[start]
            positions = order[start:end]
            positions = positions[candidates[positions, 1] != source]
            if len(positions) > top_k:
                selected = np.argpartition(similarities[positions], -top_k)[-top_k:]
                positions = positions[selected]
            rows.extend(candidates[positions, 0].tolist())
            cols.extend(candidates[positions, 1].tolist())
            values.extend(similarities[positions].tolist())
        rows.extend(range(n_cells))
        cols.extend(range(n_cells))
        values.extend(np.ones(n_cells, dtype=np.float32))
        return sparse.csr_matrix((values, (rows, cols)), shape=(n_cells, n_cells))
