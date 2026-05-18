"""KNN-based graph construction."""

import numpy as np
from sklearn.neighbors import NearestNeighbors


class KNNGraphUtils:
    """Build KNN adjacency from features (cosine similarity)."""

    @staticmethod
    def build_knn_graph(features: np.ndarray, top_k: int) -> np.ndarray:
        """Return adjacency (n_samples, n_samples); cosine similarity in [0,1], self-loops 1."""
        n_cells = len(features)
        adj = np.zeros((n_cells, n_cells))
        nn = NearestNeighbors(n_neighbors=top_k + 1, metric='cosine')
        nn.fit(features)
        distances, indices = nn.kneighbors(features)
        for i in range(n_cells):
            for j, ni in enumerate(indices[i]):
                if ni != i:
                    adj[i, ni] = 1 - distances[i, j] / 2
        np.fill_diagonal(adj, 1)
        return adj
