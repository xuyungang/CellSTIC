"""Map sparse ligand-receptor channels to hierarchy-level virtual types."""

from __future__ import annotations

from typing import Dict, Literal, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse

from .sparse_graph import SparseTypedGraph, as_sparse_typed_graph


class EdgeTypeMapper:
    @staticmethod
    def _get_group_type_indices(level_data: Dict, edge_type_map: Dict):
        if not level_data:
            raise ValueError("level_data must contain at least one hierarchy group")
        if not edge_type_map:
            raise ValueError("edge_type_map must contain at least one leaf channel")
        result = []
        for group_index, group_name in enumerate(sorted(level_data)):
            group_data = level_data[group_name]
            items = group_data if isinstance(group_data, list) else [group_data]
            leaf_names = {
                item.get("edge_type_name")
                for item in items
                if isinstance(item, dict) and item.get("edge_type_name") is not None
            }
            unknown = sorted(name for name in leaf_names if name not in edge_type_map)
            if unknown:
                raise ValueError(
                    f"Hierarchy group {group_name!r} references unknown leaf "
                    f"channels: {unknown}"
                )
            indices = sorted({edge_type_map[name] for name in leaf_names})
            if not indices:
                raise ValueError(
                    f"Hierarchy group {group_name!r} contains no leaf channels"
                )
            result.append((group_index, indices))
        return result

    @staticmethod
    def reduce_to_virtual_types(
        adj,
        level_data: Dict,
        edge_type_map: Dict,
        reduction: Literal["sum", "max", "avg"] = "sum",
    ) -> SparseTypedGraph:
        graph = as_sparse_typed_graph(adj)
        if reduction not in {"sum", "max", "avg"}:
            raise ValueError(f"Unsupported hierarchy reduction: {reduction!r}")
        indices = sorted(edge_type_map.values())
        if indices != list(range(len(graph))):
            raise ValueError(
                "edge_type_map must map contiguously onto graph channels, got "
                f"{indices} for {len(graph)} channels"
            )
        channels = []
        for _, indices in EdgeTypeMapper._get_group_type_indices(
            level_data, edge_type_map
        ):
            result = sparse.csr_matrix((graph.n_nodes, graph.n_nodes), dtype=np.float32)
            for index in indices:
                result = (
                    result.maximum(graph[index])
                    if reduction == "max"
                    else result + graph[index]
                )
            if reduction == "avg" and indices:
                result = result / len(indices)
            result.eliminate_zeros()
            channels.append(result.tocsr())
        return SparseTypedGraph.from_channels(channels, graph.n_nodes)

    @staticmethod
    def map_path_targets(
        base_graph_adj,
        level_data: Dict,
        edge_type_map: Dict,
        *,
        normalization_leaf_count: Optional[int] = None,
    ) -> SparseTypedGraph:
        """Return hierarchy-consistent path targets for one training level.

        Leaf targets remain binary.  Ancestor targets are the sum of active
        descendant leaf indicators divided by the total number of real leaves,
        matching exported upward aggregation.
        """
        targets = EdgeTypeMapper.reduce_to_virtual_types(
            base_graph_adj,
            level_data,
            edge_type_map,
            reduction="sum",
        )
        if normalization_leaf_count is None:
            return targets
        if normalization_leaf_count < 1:
            raise ValueError("normalization_leaf_count must be >= 1")
        return SparseTypedGraph.from_channels(
            (channel / float(normalization_leaf_count) for channel in targets),
            targets.n_nodes,
        )

    @staticmethod
    def map_level_graph_and_strengths(
        base_graph_adj,
        ligand_strength_adj,
        receptor_strength_adj,
        knn_per_modality: Sequence[sparse.spmatrix],
        level_data: Dict,
        edge_type_map: Dict,
        normalization_leaf_count: Optional[int] = None,
    ) -> Tuple[
        SparseTypedGraph,
        SparseTypedGraph,
        SparseTypedGraph,
        Tuple[sparse.csr_matrix, ...],
    ]:
        """Map leaf channels with one hierarchy-wide strength normalization.

        Binary topology uses union/max semantics.  Continuous ligand and
        receptor evidence is summed over descendant leaves and, when provided,
        divided by the total number of real leaves used by output aggregation.
        """
        if normalization_leaf_count is not None and normalization_leaf_count < 1:
            raise ValueError("normalization_leaf_count must be >= 1")
        base_graph = as_sparse_typed_graph(base_graph_adj)
        ligand_graph = as_sparse_typed_graph(ligand_strength_adj)
        receptor_graph = as_sparse_typed_graph(receptor_strength_adj)
        if ligand_graph.shape != base_graph.shape:
            raise ValueError("ligand_strength_adj must match base_graph_adj shape")
        if receptor_graph.shape != base_graph.shape:
            raise ValueError("receptor_strength_adj must match base_graph_adj shape")
        for index, matrix in enumerate(knn_per_modality):
            if matrix.shape != (base_graph.n_nodes, base_graph.n_nodes):
                raise ValueError(
                    f"knn_per_modality[{index}] must have shape "
                    f"({base_graph.n_nodes}, {base_graph.n_nodes}), got "
                    f"{matrix.shape}"
                )
        # The mapped base graph supplies topology only. Numeric ligand,
        # receptor, and path-target values are constructed independently.
        mapped_base = EdgeTypeMapper.reduce_to_virtual_types(
            base_graph, level_data, edge_type_map, reduction="max"
        )
        strength_reduction = "sum" if normalization_leaf_count is not None else "avg"
        mapped_ligand = EdgeTypeMapper.reduce_to_virtual_types(
            ligand_graph,
            level_data,
            edge_type_map,
            reduction=strength_reduction,
        )
        mapped_receptor = EdgeTypeMapper.reduce_to_virtual_types(
            receptor_graph,
            level_data,
            edge_type_map,
            reduction=strength_reduction,
        )
        if normalization_leaf_count is not None:
            mapped_ligand = SparseTypedGraph.from_channels(
                (
                    channel / float(normalization_leaf_count)
                    for channel in mapped_ligand
                ),
                mapped_ligand.n_nodes,
            )
            mapped_receptor = SparseTypedGraph.from_channels(
                (
                    channel / float(normalization_leaf_count)
                    for channel in mapped_receptor
                ),
                mapped_receptor.n_nodes,
            )
        knn = tuple(sparse.csr_matrix(matrix) for matrix in knn_per_modality)
        return mapped_base, mapped_ligand, mapped_receptor, knn
