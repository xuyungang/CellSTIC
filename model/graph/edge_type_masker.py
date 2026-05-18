"""Balanced edge type masking for graph-based training."""

import numpy as np
from typing import Tuple


class EdgeTypeMasker:
    """Balanced edge type masking with uniform sampling per type."""

    @staticmethod
    def balanced_edge_type_masking(
        base_graph: np.ndarray,
        sampling_rate: float,
        random_seed: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (masked_graph, mask_indicator_3d). 3D base_graph: mask each type independently by sampling_rate."""
        np.random.seed(random_seed)
        masked_graph = base_graph.copy()
        n_virtual_types = base_graph.shape[2]
        mask_indicator_3d = np.zeros_like(base_graph)

        for vt_idx in range(n_virtual_types):
            type_graph = base_graph[:, :, vt_idx]
            edge_positions = np.column_stack(np.where(type_graph > 0))
            non_self = edge_positions[edge_positions[:, 0] != edge_positions[:, 1]] if len(edge_positions) > 0 else np.empty((0, 2), dtype=int)
            if len(non_self) == 0:
                continue
            num_edges = len(non_self)
            num_to_mask = int(num_edges * sampling_rate)
            if num_to_mask <= 0:
                continue
            mask_indices = np.random.choice(num_edges, num_to_mask, replace=False)
            for idx in mask_indices:
                i, j = non_self[idx]
                mask_indicator_3d[i, j, vt_idx] = 1
                masked_graph[i, j, vt_idx] = 0

        return masked_graph, mask_indicator_3d
