"""Recompute network descriptors from retained CellSTIC communication graphs."""
from __future__ import annotations

import random
import numpy as np
from scipy import sparse
from .result_io import clean_matrix


def network_metrics(matrix, *, seed=42, block_size=128):
    """Return exact topology descriptors without allocating all-pairs distances.

    Remove self edges and symmetrize by maximum reciprocal strength. Efficiency
    is unweighted mean reciprocal shortest-path length over ALL ordered distinct
    node pairs; unreachable pairs contribute zero. Average path length uses only
    reachable distinct pairs. Weighted Louvain modularity uses the symmetrized
    strengths, resolution 1, and a fixed random generator. Isolates remain nodes.
    """
    import igraph as ig

    if block_size < 1:
        raise ValueError("block_size must be positive")
    matrix = clean_matrix(matrix, remove_self=True)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Expected a square communication matrix")
    adjacency = matrix.maximum(matrix.T)
    upper = sparse.triu(adjacency, k=1, format="coo")
    n = adjacency.shape[0]
    graph = ig.Graph(n=n, edges=list(zip(upper.row.tolist(), upper.col.tolist())), directed=False)
    graph.es["weight"] = upper.data.tolist()
    components = graph.connected_components()
    inverse_sum = distance_sum = 0.0
    reachable_pairs = 0
    for members in components:
        if len(members) < 2:
            continue
        component = graph.induced_subgraph(members)
        size = component.vcount()
        reachable_pairs += size * (size - 1)
        for start in range(0, size, block_size):
            sources = list(range(start, min(start + block_size, size)))
            distances = np.asarray(component.distances(source=sources, weights=None), dtype=float)
            positive = distances > 0
            if not np.isfinite(distances).all():
                raise ValueError("Unreachable vertex within a connected component")
            distance_sum += float(distances.sum())
            inverse_sum += float(np.divide(1.0, distances, out=np.zeros_like(distances), where=positive).sum())
    if graph.ecount():
        ig.set_random_number_generator(random.Random(seed))
        try:
            communities = graph.community_multilevel(weights="weight", resolution=1)
            modularity = float(communities.modularity)
            n_communities = len(communities)
        finally:
            ig.set_random_number_generator(None)
    else:
        modularity, n_communities = np.nan, n
    denominator = n * (n - 1)
    return dict(
        global_efficiency=inverse_sum / denominator if denominator else np.nan,
        average_shortest_path=distance_sum / reachable_pairs if reachable_pairs else np.nan,
        modularity=modularity, observations=n, undirected_edges=graph.ecount(),
        reachable_pair_fraction=reachable_pairs / denominator if denominator else np.nan,
        components=len(components), communities=n_communities,
        largest_component_fraction=max(components.sizes(), default=0) / n if n else np.nan,
    )
