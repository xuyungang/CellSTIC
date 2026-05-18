"""Negative sampling for graph-based training."""

import numpy as np
from typing import Optional, Tuple

try:
    from scipy.sparse import issparse
except ImportError:
    def issparse(x):
        return False


class NegativeSampler:
    """Generate negative samples balanced across edge types, optionally within a specified spatial distance."""

    @staticmethod
    def generate_negative_samples(
        base_graph: np.ndarray,
        random_seed: int,
        mask_indicator: Optional[np.ndarray] = None,
        spatial_distances: Optional[np.ndarray] = None,
        max_distance: Optional[float] = None,
        min_distance: Optional[float] = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (neg_samples (N, 2), negative_sample_indicator 3D). Neg count per type matches mask_indicator ones or base_graph positives.
        When spatial_distances and max_distance are provided, negative pairs are restricted to (i,j) with
        min_distance <= spatial_distances[i,j] <= max_distance (hard negatives within the same spatial range)."""
        np.random.seed(random_seed)
        n_types = base_graph.shape[2]
        neg_samples = []
        negative_sample_indicator = np.zeros_like(base_graph)

        no_edge_mask = np.all(base_graph <= 0, axis=2)
        np.fill_diagonal(no_edge_mask, False)
        cand_i, cand_j = np.where(no_edge_mask)
        candidate_coords = np.column_stack((cand_i, cand_j)) if len(cand_i) > 0 else np.array([]).reshape(0, 2)

        if candidate_coords.shape[0] > 0 and spatial_distances is not None and max_distance is not None:
            dist = spatial_distances
            if issparse(dist):
                dist = dist.toarray()
            else:
                dist = np.asarray(dist, dtype=np.float64)
            # Restrict to pairs within [min_distance, max_distance]
            dist_flat = dist[cand_i, cand_j]
            in_range = (dist_flat >= min_distance) & (dist_flat <= max_distance)
            if np.any(in_range):
                candidate_coords = candidate_coords[in_range]
            # else: keep all no-edge pairs as fallback when no pair in range

        for type_idx in range(n_types):
            target_count = int(np.sum(mask_indicator[:, :, type_idx] == 1)) if mask_indicator is not None else int(np.sum(base_graph[:, :, type_idx] > 0))
            if target_count <= 0 or candidate_coords.shape[0] == 0:
                continue
            replace = candidate_coords.shape[0] < target_count
            sel_idx = np.random.choice(candidate_coords.shape[0], size=target_count, replace=replace)
            chosen = candidate_coords[sel_idx]
            neg_samples.extend(chosen.tolist())
            negative_sample_indicator[chosen[:, 0], chosen[:, 1], type_idx] = 1

        neg_arr = np.array(neg_samples, dtype=np.int64).reshape(-1, 2) if neg_samples else np.empty((0, 2), dtype=np.int64)
        return neg_arr, negative_sample_indicator
