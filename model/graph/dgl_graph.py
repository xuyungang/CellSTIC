"""DGL graph construction and local-batch utilities for CellSTIC."""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import dgl
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.csgraph import reverse_cuthill_mckee

from .sparse_graph import SparseTypedGraph, as_sparse_typed_graph, sparse_values


@dataclass(frozen=True)
class DGLNodeBatchIndex:
    """Cached incoming-neighbour indices shared by aligned feature graphs."""

    incoming_adjacencies: Tuple[sparse.csr_matrix, ...]
    n_nodes: int
    target_order: np.ndarray


class DGLGraphUtils:
    """Utilities for constructing DGL graphs."""

    @staticmethod
    def build_dgl_graph_from_virtual_types_graph(
        spatial_graph,
        node_features: torch.Tensor,
        device: torch.device,
        spatial_coords: torch.Tensor,
        ligand_strength_adj: SparseTypedGraph,
        receptor_strength_adj: SparseTypedGraph,
        knn_strength_adj: Sequence[sparse.spmatrix],
        path_target_adj: Optional[SparseTypedGraph] = None,
    ) -> dgl.DGLGraph:
        """Build a DGL graph from sparse virtual-type channels.

        Edge features are ligand strength, receptor strength, per-modality KNN
        strength, source/target coordinates, and a self-loop indicator.
        """
        spatial_graph = as_sparse_typed_graph(spatial_graph)
        ligand_strength_adj = as_sparse_typed_graph(ligand_strength_adj)
        receptor_strength_adj = as_sparse_typed_graph(receptor_strength_adj)
        if path_target_adj is not None:
            path_target_adj = as_sparse_typed_graph(path_target_adj)
        n_nodes, _, n_virtual_types = spatial_graph.shape
        if ligand_strength_adj.shape != spatial_graph.shape:
            raise ValueError("ligand_strength_adj must match spatial_graph shape")
        if receptor_strength_adj.shape != spatial_graph.shape:
            raise ValueError("receptor_strength_adj must match spatial_graph shape")
        if path_target_adj is not None and path_target_adj.shape != spatial_graph.shape:
            raise ValueError("path_target_adj must match spatial_graph shape")

        n_modalities = len(knn_strength_adj)
        if n_modalities < 1:
            raise ValueError("knn_strength_adj must contain at least one modality")
        for index, matrix in enumerate(knn_strength_adj):
            if matrix.shape != (n_nodes, n_nodes):
                raise ValueError(
                    f"knn_strength_adj[{index}] must have shape "
                    f"({n_nodes}, {n_nodes}), got {matrix.shape}"
                )
        node_features = torch.as_tensor(
            node_features, dtype=torch.float32, device=device
        )
        spatial_coords = torch.as_tensor(
            spatial_coords, dtype=torch.float32, device=device
        )
        if node_features.ndim != 2 or node_features.shape[0] != n_nodes:
            raise ValueError(
                "node_features must be a 2D matrix with one row per node, got "
                f"{tuple(node_features.shape)}"
            )
        if spatial_coords.shape != (n_nodes, 2):
            raise ValueError(
                f"spatial_coords must have shape ({n_nodes}, 2), got "
                f"{tuple(spatial_coords.shape)}"
            )

        src_parts, dst_parts, type_parts = [], [], []
        ligand_parts, receptor_parts, knn_parts, path_target_parts = (
            [],
            [],
            [],
            [],
        )
        for vt_idx in range(n_virtual_types):
            channel = spatial_graph[vt_idx].tocoo()
            src, dst = channel.row, channel.col
            if len(src) == 0:
                continue
            mask = src != dst
            src, dst = src[mask], dst[mask]
            if len(src) == 0:
                continue
            src_parts.append(np.asarray(src, dtype=np.int64))
            dst_parts.append(np.asarray(dst, dtype=np.int64))
            type_parts.append(np.full(len(src), vt_idx, dtype=np.int64))
            ligand_parts.append(
                sparse_values(ligand_strength_adj[vt_idx], src, dst).astype(
                    np.float32, copy=False
                )
            )
            receptor_parts.append(
                sparse_values(receptor_strength_adj[vt_idx], src, dst).astype(
                    np.float32, copy=False
                )
            )
            path_target_parts.append(
                sparse_values(path_target_adj[vt_idx], src, dst).astype(
                    np.float32, copy=False
                )
                if path_target_adj is not None
                else np.ones(len(src), dtype=np.float32)
            )
            knn_parts.append(
                np.column_stack(
                    [sparse_values(matrix, src, dst) for matrix in knn_strength_adj]
                ).astype(np.float32, copy=False)
            )

        # Append one self-loop per node after all typed non-self edges.
        self_nodes = np.arange(n_nodes, dtype=np.int64)
        src_parts.append(self_nodes)
        dst_parts.append(self_nodes)
        type_parts.append(np.zeros(n_nodes, dtype=np.int64))
        ligand_parts.append(np.ones(n_nodes, dtype=np.float32))
        receptor_parts.append(np.ones(n_nodes, dtype=np.float32))
        knn_parts.append(np.ones((n_nodes, n_modalities), dtype=np.float32))
        path_target_parts.append(np.zeros(n_nodes, dtype=np.float32))

        src_array = np.concatenate(src_parts)
        dst_array = np.concatenate(dst_parts)
        type_array = np.concatenate(type_parts)
        ligand_array = np.concatenate(ligand_parts)
        receptor_array = np.concatenate(receptor_parts)
        knn_array = np.concatenate(knn_parts, axis=0)
        path_target_array = np.concatenate(path_target_parts)

        g = dgl.graph((src_array, dst_array), num_nodes=n_nodes).to(device)
        g.ndata["node_features"] = node_features

        n_edges = len(src_array)
        n_self = n_nodes
        n_non_self = n_edges - n_self

        # Store class indices rather than an ``(n_edges, n_virtual_types)``
        # one-hot matrix.  The latter dominates memory for leaf-level graphs
        # with many LR channels and is not required by cross-entropy.
        g.edata["edge_type"] = torch.tensor(type_array, dtype=torch.long, device=device)
        g.edata["path_target"] = torch.tensor(
            path_target_array, dtype=torch.float32, device=device
        )
        is_self = torch.zeros(n_edges, dtype=torch.bool, device=device)
        is_self[n_non_self:] = True
        g.edata["is_self"] = is_self

        src_t = torch.as_tensor(src_array, dtype=torch.long, device=device)
        dst_t = torch.as_tensor(dst_array, dtype=torch.long, device=device)
        coords = torch.cat([spatial_coords[src_t], spatial_coords[dst_t]], dim=1)
        lig = torch.as_tensor(
            ligand_array, dtype=torch.float32, device=device
        ).unsqueeze(1)
        rec = torch.as_tensor(
            receptor_array, dtype=torch.float32, device=device
        ).unsqueeze(1)
        knn = torch.as_tensor(knn_array, dtype=torch.float32, device=device)

        head = torch.cat([lig, rec, knn], dim=1)
        n_knn = head.shape[1] - 2
        self_indicator = torch.zeros(n_edges, 1, dtype=torch.float32, device=device)
        self_indicator[n_non_self:] = 1.0
        raw = torch.cat([head, coords, self_indicator], dim=1)

        # Normalize from non-self edges so self-edge strengths can be set to 1.
        if n_non_self > 0:
            raw_non_self = raw[:n_non_self]
            head_norm_ns = DGLGraphUtils._minmax_normalize(raw_non_self[:, : 2 + n_knn])
            coord_slice = slice(2 + n_knn, 6 + n_knn)
            c_min = raw_non_self[:, coord_slice].min()
            c_max = raw_non_self[:, coord_slice].max()
            c_range = c_max - c_min
            c_range = torch.where(c_range > 0, c_range, torch.ones_like(c_range))
            coords_norm_ns = (raw_non_self[:, coord_slice] - c_min) / c_range
            self_head = torch.ones(
                n_self, 2 + n_knn, dtype=torch.float32, device=device
            )
            self_coords = (raw[n_non_self:, coord_slice] - c_min) / c_range
            self_type = torch.ones(n_self, 1, dtype=torch.float32, device=device)
            g.edata["edge_features"] = torch.cat(
                [
                    torch.cat(
                        [
                            head_norm_ns,
                            coords_norm_ns,
                            raw_non_self[:, -1:],
                        ],
                        dim=1,
                    ),
                    torch.cat([self_head, self_coords, self_type], dim=1),
                ],
                dim=0,
            )
        else:
            self_head = torch.ones(
                n_self, 2 + n_knn, dtype=torch.float32, device=device
            )
            self_coords_raw = raw[:, 2 + n_knn : 6 + n_knn]
            c_min, c_max = self_coords_raw.min(), self_coords_raw.max()
            c_range = torch.where(c_max > c_min, c_max - c_min, torch.ones_like(c_max))
            self_coords = (self_coords_raw - c_min) / c_range
            self_type = torch.ones(n_self, 1, dtype=torch.float32, device=device)
            g.edata["edge_features"] = torch.cat(
                [self_head, self_coords, self_type], dim=1
            )

        return g

    @staticmethod
    def build_dgl_graph(
        spatial_graph: sparse.spmatrix,
        node_features: torch.Tensor,
        device: torch.device,
        spatial_coords: torch.Tensor,
    ) -> dgl.DGLGraph:
        """Build a feature DGL graph with required node and edge features."""
        adjacency = sparse.coo_matrix(spatial_graph)
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError(f"spatial_graph must be square, got {adjacency.shape}")
        n_nodes = adjacency.shape[0]
        src, dst = adjacency.row, adjacency.col
        if len(src) == 0:
            raise ValueError("spatial_graph must contain at least one edge")
        node_features = torch.as_tensor(
            node_features, dtype=torch.float32, device=device
        )
        spatial_coords = torch.as_tensor(
            spatial_coords, dtype=torch.float32, device=device
        )
        if node_features.ndim != 2 or node_features.shape[0] != n_nodes:
            raise ValueError(
                "node_features must be a 2D matrix with one row per node, got "
                f"{tuple(node_features.shape)}"
            )
        if spatial_coords.shape != (n_nodes, 2):
            raise ValueError(
                f"spatial_coords must have shape ({n_nodes}, 2), got "
                f"{tuple(spatial_coords.shape)}"
            )
        g = dgl.graph((src, dst), num_nodes=n_nodes).to(device)
        g.ndata["node_features"] = node_features

        src_tensor = torch.as_tensor(src, dtype=torch.long, device=device)
        dst_tensor = torch.as_tensor(dst, dtype=torch.long, device=device)
        src_coords = spatial_coords[src_tensor]
        dst_coords = spatial_coords[dst_tensor]
        values = sparse_values(spatial_graph, src, dst)
        knn_strength = torch.as_tensor(
            np.asarray(values), dtype=torch.float32, device=device
        ).unsqueeze(1)
        coords = torch.cat([src_coords, dst_coords], dim=1)
        c_min, c_max = coords.min(), coords.max()
        c_range = c_max - c_min
        c_range = torch.where(c_range > 0, c_range, torch.ones_like(c_range))
        coords_norm = (coords - c_min) / c_range
        g.edata["edge_features"] = torch.cat([knn_strength, coords_norm], dim=1)

        return g

    @staticmethod
    def build_node_batch_index(
        graphs: Sequence[dgl.DGLGraph],
    ) -> DGLNodeBatchIndex:
        """Cache incoming sparse adjacency for aligned node mini-batches."""
        if not graphs:
            raise ValueError("graphs must contain at least one DGL graph")
        n_nodes = graphs[0].num_nodes()
        incoming = []
        for index, graph in enumerate(graphs):
            if graph.num_nodes() != n_nodes:
                raise ValueError(
                    "All feature graphs must contain the same aligned nodes; "
                    f"graph 0 has {n_nodes}, graph {index} has {graph.num_nodes()}"
                )
            sources, targets = graph.edges(order="eid")
            sources_np = sources.detach().cpu().numpy().astype(np.int64, copy=False)
            targets_np = targets.detach().cpu().numpy().astype(np.int64, copy=False)
            adjacency = sparse.csr_matrix(
                (
                    np.ones(len(sources_np), dtype=np.uint8),
                    (targets_np, sources_np),
                ),
                shape=(n_nodes, n_nodes),
            )
            adjacency.sum_duplicates()
            adjacency.data[:] = 1
            incoming.append(adjacency)
        connectivity = incoming[0]
        for adjacency in incoming[1:]:
            connectivity = connectivity.maximum(adjacency)
        connectivity = connectivity.maximum(connectivity.transpose()).tocsr()
        target_order = reverse_cuthill_mckee(connectivity, symmetric_mode=True).astype(
            np.int64, copy=False
        )
        if len(target_order) != n_nodes:
            raise RuntimeError("Failed to construct a complete feature-batch order")
        return DGLNodeBatchIndex(tuple(incoming), n_nodes, target_order)

    @staticmethod
    def build_aligned_node_batch(
        graphs: Sequence[dgl.DGLGraph],
        batch_index: DGLNodeBatchIndex,
        target_nodes: np.ndarray,
        *,
        num_message_passing_hops: int,
    ) -> Tuple[Tuple[dgl.DGLGraph, ...], torch.Tensor]:
        """Build aligned induced subgraphs around one target-node batch.

        Incoming neighbourhoods from every modality are unioned at each hop.
        This gives every target node the context required by the configured
        HODGNN depth while keeping graph activations local to the batch.
        """
        if len(graphs) != len(batch_index.incoming_adjacencies):
            raise ValueError("graphs do not match the cached node-batch index")
        if num_message_passing_hops < 0:
            raise ValueError("num_message_passing_hops must be non-negative")
        targets = np.asarray(target_nodes, dtype=np.int64).reshape(-1)
        if len(targets) == 0:
            raise ValueError("target_nodes must contain at least one node")
        if len(np.unique(targets)) != len(targets):
            raise ValueError("target_nodes must not contain duplicates")
        if targets.min() < 0 or targets.max() >= batch_index.n_nodes:
            raise ValueError("target_nodes contains an out-of-range node")

        local_nodes = np.sort(targets)
        frontier = local_nodes
        for _ in range(num_message_passing_hops):
            neighbour_parts = [
                adjacency[frontier].indices
                for adjacency in batch_index.incoming_adjacencies
            ]
            neighbours = (
                np.concatenate(neighbour_parts)
                if neighbour_parts
                else np.empty(0, dtype=np.int64)
            )
            expanded = np.union1d(local_nodes, neighbours).astype(np.int64, copy=False)
            frontier = np.setdiff1d(expanded, local_nodes, assume_unique=True)
            local_nodes = expanded
            if frontier.size == 0:
                break

        node_tensor = torch.as_tensor(
            local_nodes, dtype=torch.long, device=graphs[0].device
        )
        local_graphs = tuple(
            dgl.node_subgraph(
                graph,
                node_tensor.to(graph.device),
                relabel_nodes=True,
                store_ids=True,
            )
            for graph in graphs
        )
        reference_nodes = local_graphs[0].ndata[dgl.NID]
        reference_np = (
            reference_nodes.detach().cpu().numpy().astype(np.int64, copy=False)
        )
        for index, graph in enumerate(local_graphs[1:], start=1):
            graph_nodes = graph.ndata[dgl.NID].detach().cpu().numpy()
            if not np.array_equal(reference_np, graph_nodes):
                raise RuntimeError(
                    f"Feature graph {index} produced a different local node order"
                )

        order = np.argsort(reference_np, kind="stable")
        sorted_nodes = reference_np[order]
        insertion = np.searchsorted(sorted_nodes, targets)
        if np.any(insertion >= len(sorted_nodes)) or not np.array_equal(
            sorted_nodes[insertion], targets
        ):
            raise RuntimeError("Failed to map feature targets to local node IDs")
        local_targets = torch.as_tensor(
            order[insertion], dtype=torch.long, device=graphs[0].device
        )
        return local_graphs, local_targets

    @staticmethod
    def _minmax_normalize(x: torch.Tensor) -> torch.Tensor:
        """Min-max normalization with zero-division protection."""
        x_min = x.min(dim=0, keepdim=True)[0]
        x_max = x.max(dim=0, keepdim=True)[0]
        x_range = x_max - x_min
        x_range = torch.where(x_range == 0, torch.ones_like(x_range), x_range)
        return (x - x_min) / x_range
