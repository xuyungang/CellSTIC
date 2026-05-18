"""Map edge types to virtual types (level groups) for hierarchical graphs."""

import numpy as np
from typing import Dict, Literal, Tuple


class EdgeTypeMapper:
    """Aggregate edge types per level group; output (n_cells, n_cells, n_groups)."""

    @staticmethod
    def _get_group_type_indices(level_data: Dict, edge_type_map: Dict):
        """Return list of (group_index, type_indices) for each group in level_data."""
        group_names = sorted(level_data.keys())
        result = []
        for g_idx, group_name in enumerate(group_names):
            group_data = level_data[group_name]
            items = group_data if isinstance(group_data, list) else [group_data]
            type_indices = sorted(set(
                edge_type_map[item["edge_type_name"]]
                for item in items
                if isinstance(item, dict) and "edge_type_name" in item and item["edge_type_name"] in edge_type_map
            ))
            result.append((g_idx, type_indices))
        return result

    @staticmethod
    def reduce_to_virtual_types(
        adj: np.ndarray,
        level_data: Dict,
        edge_type_map: Dict = None,
        reduction: Literal["sum", "max", "avg"] = "sum",
    ) -> np.ndarray:
        """Return (n_cells, n_cells, n_groups) by reducing adj over type indices per group. No normalization."""
        if adj.ndim != 3:
            raise ValueError("adj must be 3D (n_cells, n_cells, n_types)")
        n_cells = adj.shape[0]
        group_names = sorted(level_data.keys())
        n_groups = len(group_names)
        out = np.zeros((n_cells, n_cells, n_groups), dtype=adj.dtype)

        if not edge_type_map:
            return out

        for g_idx, type_indices in EdgeTypeMapper._get_group_type_indices(level_data, edge_type_map):
            if not type_indices:
                continue
            if reduction == "sum":
                out[:, :, g_idx] = adj[:, :, type_indices].sum(axis=2)
            elif reduction == "avg":
                out[:, :, g_idx] = adj[:, :, type_indices].sum(axis=2) / len(type_indices)
            else:
                out[:, :, g_idx] = adj[:, :, type_indices].max(axis=2)
        return out

    @staticmethod
    def map_edge_types_to_virtual_types(
        base_graph: np.ndarray,
        level_data: Dict,
        edge_type_map: Dict = None,
    ) -> np.ndarray:
        """Return new base graph (n_cells, n_cells, n_groups). Per group: max over type indices (0/1); then bottom 50% to 0, row-normalize, diagonal 1."""
        n_cells = base_graph.shape[0]
        group_names = sorted(level_data.keys())
        n_groups = len(group_names)
        group_graph = np.zeros((n_cells, n_cells, n_groups), dtype=base_graph.dtype)

        if not edge_type_map:
            return group_graph

        for g_idx, type_indices in EdgeTypeMapper._get_group_type_indices(level_data, edge_type_map):
            if type_indices:
                group_graph[:, :, g_idx] = base_graph[:, :, type_indices].max(axis=2)

        for g_idx in range(n_groups):
            tg = group_graph[:, :, g_idx]
            if tg.sum() == 0:
                continue
            flat = tg.flatten()
            bottom_count = len(flat) // 2
            if bottom_count > 0:
                bottom_flat = np.argsort(flat)[:bottom_count]
                group_graph[bottom_flat // n_cells, bottom_flat % n_cells, g_idx] = 0
            row_sums = group_graph[:, :, g_idx].sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            group_graph[:, :, g_idx] = group_graph[:, :, g_idx] / row_sums
            np.fill_diagonal(group_graph[:, :, g_idx], 1)

        return group_graph

    @staticmethod
    def map_level_graph_and_strengths(
        base_graph_adj: np.ndarray,
        ligand_strength_adj: np.ndarray,
        receptor_strength_adj: np.ndarray,
        knn_per_modality: np.ndarray,
        level_data: Dict,
        edge_type_map: Dict = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Map base graph and strength/KNN arrays to virtual types for one level.
        Returns (mapped_base_graph, mapped_ligand_strength, mapped_receptor_strength, knn_strength_adj).
        - base_graph_adj: max over type indices per group, then bottom-50% zero + row-normalize.
        - ligand/receptor_strength_adj: sum over type indices per group.
        - knn_per_modality (n_cells, n_cells, n_modalities): for each modality call reduce_to_virtual_types (broadcast to n_types then sum per group); return (n_cells, n_cells, n_groups, n_modalities) for edge-feature concatenation."""
        mapped_base_graph = EdgeTypeMapper.map_edge_types_to_virtual_types(
            base_graph_adj, level_data, edge_type_map
        )
        mapped_ligand_strength = EdgeTypeMapper.reduce_to_virtual_types(
            ligand_strength_adj, level_data, edge_type_map, reduction="avg"
        )
        mapped_receptor_strength = EdgeTypeMapper.reduce_to_virtual_types(
            receptor_strength_adj, level_data, edge_type_map, reduction="avg"
        )
        knn = np.asarray(knn_per_modality)
        n_types = base_graph_adj.shape[2]
        n_modalities = knn.shape[2]
        knn_mapped_list = []
        for m in range(n_modalities):
            knn_m = knn[:, :, m:m + 1]  # (n_cells, n_cells, 1)
            knn_expanded = np.broadcast_to(knn_m, (knn.shape[0], knn.shape[1], n_types))
            knn_mapped_list.append(
                EdgeTypeMapper.reduce_to_virtual_types(
                    knn_expanded, level_data, edge_type_map, reduction="max"
                )
            )
        knn_strength_adj = np.stack(knn_mapped_list, axis=-1)  # (n_cells, n_cells, n_groups, n_modalities)
        return mapped_base_graph, mapped_ligand_strength, mapped_receptor_strength, knn_strength_adj
