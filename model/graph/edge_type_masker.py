"""Balanced masking for sparse typed graphs and cached DGL graphs."""

from dataclasses import dataclass
from typing import Optional, Tuple

import dgl
import numpy as np
import torch
from scipy import sparse


@dataclass(frozen=True)
class DGLChannelEdgeIndex:
    """CPU graph index cached once for channel sampling and local batches."""

    sources: np.ndarray
    targets: np.ndarray
    path_targets: np.ndarray
    channel_edge_ids: Tuple[np.ndarray, ...]
    incoming_adjacency: sparse.csr_matrix


class EdgeTypeMasker:
    @staticmethod
    def build_dgl_channel_index(
        graph: dgl.DGLGraph,
        n_channels: int,
    ) -> DGLChannelEdgeIndex:
        """Index non-self DGL edge IDs by channel without a dense one-hot mask."""
        if n_channels < 1:
            raise ValueError("n_channels must be >= 1")
        if "edge_type" not in graph.edata or "is_self" not in graph.edata:
            raise KeyError("graph must contain edge_type and is_self edge data")
        edge_types = graph.edata["edge_type"]
        if edge_types.ndim == 2:
            edge_types = edge_types.argmax(dim=1)
        if edge_types.ndim != 1 or edge_types.shape[0] != graph.num_edges():
            raise ValueError("graph.edata['edge_type'] must contain one class per edge")
        sources, targets = graph.edges(order="eid")
        sources_np = sources.detach().cpu().numpy().astype(np.int64, copy=False)
        targets_np = targets.detach().cpu().numpy().astype(np.int64, copy=False)
        types_np = edge_types.detach().cpu().numpy().astype(np.int64, copy=False)
        path_targets = (
            graph.edata["path_target"] if "path_target" in graph.edata else None
        )
        if path_targets is None:
            path_targets_np = np.ones(graph.num_edges(), dtype=np.float32)
        else:
            path_targets_np = (
                path_targets.detach().cpu().numpy().astype(np.float32, copy=False)
            )
        non_self = ~graph.edata["is_self"].detach().cpu().numpy().astype(bool)
        if np.any(types_np < 0) or np.any(types_np >= n_channels):
            raise ValueError("graph contains an out-of-range edge type")
        if (
            np.any(~np.isfinite(path_targets_np))
            or np.any(path_targets_np < 0)
            or np.any(path_targets_np > 1)
        ):
            raise ValueError("graph path targets must be finite values in [0, 1]")
        if np.any(path_targets_np[non_self] <= 0):
            raise ValueError(
                "every non-self training edge must have a positive path target"
            )
        non_self_eids = np.flatnonzero(non_self).astype(np.int64, copy=False)
        edge_order = non_self_eids[np.argsort(types_np[non_self_eids], kind="stable")]
        ordered_types = types_np[edge_order]
        boundaries = np.searchsorted(
            ordered_types, np.arange(n_channels + 1), side="left"
        )
        channel_edge_ids = tuple(
            edge_order[boundaries[channel] : boundaries[channel + 1]]
            for channel in range(n_channels)
        )
        incoming_adjacency = sparse.csr_matrix(
            (
                np.ones(int(non_self.sum()), dtype=np.uint8),
                (targets_np[non_self], sources_np[non_self]),
            ),
            shape=(graph.num_nodes(), graph.num_nodes()),
        )
        incoming_adjacency.sum_duplicates()
        incoming_adjacency.data[:] = 1
        return DGLChannelEdgeIndex(
            sources=sources_np,
            targets=targets_np,
            path_targets=path_targets_np,
            channel_edge_ids=channel_edge_ids,
            incoming_adjacency=incoming_adjacency,
        )

    @staticmethod
    def sample_balanced_dgl_positive_edges(
        graph: dgl.DGLGraph,
        edge_index: DGLChannelEdgeIndex,
        sampling_rate: float,
        *,
        min_edges_per_channel: int,
        max_edges_per_channel: int,
        random_seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Select bounded positive edges per channel for local graph batches.

        Positions, channel IDs, targets, and original DGL edge IDs are aligned
        one-to-one. The caller removes the selected edge IDs from the local
        batch graph in which those samples are reconstructed.
        """
        if not 0.0 < sampling_rate <= 1.0:
            raise ValueError("sampling_rate must be in (0, 1]")
        if min_edges_per_channel < 1:
            raise ValueError("min_edges_per_channel must be >= 1")
        if max_edges_per_channel < min_edges_per_channel:
            raise ValueError("max_edges_per_channel must be >= min_edges_per_channel")
        if len(edge_index.sources) != graph.num_edges():
            raise ValueError("edge_index does not match graph edge count")

        rng = np.random.default_rng(random_seed)
        selected_parts = []
        channel_parts = []
        counts = np.zeros(len(edge_index.channel_edge_ids), dtype=np.int64)
        for channel, eligible in enumerate(edge_index.channel_edge_ids):
            if len(eligible) == 0:
                continue
            requested = max(
                min_edges_per_channel,
                int(np.ceil(len(eligible) * sampling_rate)),
            )
            count = min(len(eligible), max_edges_per_channel, requested)
            selected = rng.choice(eligible, count, replace=False)
            selected_parts.append(np.asarray(selected, dtype=np.int64))
            channel_parts.append(np.full(count, channel, dtype=np.int64))
            counts[channel] = count

        if not selected_parts:
            empty_positions = np.empty((0, 2), dtype=np.int64)
            return (
                empty_positions,
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int64),
                counts,
            )

        selected_eids = np.concatenate(selected_parts)
        channels = np.concatenate(channel_parts)
        positions = np.column_stack(
            (edge_index.sources[selected_eids], edge_index.targets[selected_eids])
        )
        path_targets = edge_index.path_targets[selected_eids]
        permutation = rng.permutation(len(selected_eids))
        selected_eids = selected_eids[permutation]
        positions = positions[permutation]
        channels = channels[permutation]
        path_targets = path_targets[permutation]
        return positions, channels, path_targets, selected_eids, counts

    @staticmethod
    def build_dgl_training_batch(
        graph: dgl.DGLGraph,
        edge_index: DGLChannelEdgeIndex,
        sample_positions: np.ndarray,
        masked_positive_edge_ids: np.ndarray,
        *,
        num_message_passing_hops: int,
    ) -> Tuple[dgl.DGLGraph, np.ndarray, torch.Tensor]:
        """Construct the local graph induced by one positive/negative batch.

        Sample endpoints are seed nodes. Incoming neighbours are expanded for
        the number of HODGNN layers so the batch includes the message-passing
        context needed by the scored nodes. The positive target edges of this
        batch are then removed to prevent reconstruction leakage.

        Returns the local graph, sample positions remapped to local node IDs,
        and local-to-global node IDs for slicing non-graph feature tensors.
        """
        positions = np.asarray(sample_positions, dtype=np.int64).reshape(-1, 2)
        positive_eids = np.asarray(masked_positive_edge_ids, dtype=np.int64).reshape(-1)
        if len(positions) == 0:
            raise ValueError("sample_positions must contain at least one edge")
        if positions.min() < 0 or positions.max() >= graph.num_nodes():
            raise ValueError("sample_positions contains an out-of-range node")
        if num_message_passing_hops < 0:
            raise ValueError("num_message_passing_hops must be non-negative")
        if positive_eids.size and (
            positive_eids.min() < 0 or positive_eids.max() >= graph.num_edges()
        ):
            raise ValueError("masked_positive_edge_ids contains an invalid edge ID")

        local_nodes = np.unique(positions.reshape(-1))
        frontier = local_nodes
        for _ in range(num_message_passing_hops):
            neighbours = edge_index.incoming_adjacency[frontier].indices
            if neighbours.size == 0:
                break
            expanded = np.union1d(local_nodes, neighbours).astype(np.int64, copy=False)
            frontier = np.setdiff1d(expanded, local_nodes, assume_unique=True)
            local_nodes = expanded
            if frontier.size == 0:
                break

        node_tensor = torch.as_tensor(
            local_nodes, dtype=torch.long, device=graph.device
        )
        local_graph = dgl.node_subgraph(
            graph,
            node_tensor,
            relabel_nodes=True,
            store_ids=True,
        )
        global_nodes = local_graph.ndata[dgl.NID]
        global_nodes_np = (
            global_nodes.detach().cpu().numpy().astype(np.int64, copy=False)
        )
        order = np.argsort(global_nodes_np, kind="stable")
        sorted_global_nodes = global_nodes_np[order]
        insertion = np.searchsorted(sorted_global_nodes, positions)
        if np.any(insertion >= len(sorted_global_nodes)) or not np.array_equal(
            sorted_global_nodes[insertion], positions
        ):
            raise RuntimeError("Failed to map global sample endpoints to batch nodes")
        local_positions = order[insertion].astype(np.int64, copy=False)

        if positive_eids.size:
            original_eids = (
                local_graph.edata[dgl.EID]
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64, copy=False)
            )
            sorted_positive_eids = np.sort(positive_eids)
            matches = np.searchsorted(sorted_positive_eids, original_eids)
            is_masked = matches < len(sorted_positive_eids)
            candidate_ids = np.flatnonzero(is_masked)
            is_masked[candidate_ids] = (
                sorted_positive_eids[matches[candidate_ids]]
                == original_eids[candidate_ids]
            )
            kept_eids = np.flatnonzero(~is_masked).astype(np.int64, copy=False)
            local_graph = dgl.edge_subgraph(
                local_graph,
                torch.as_tensor(kept_eids, dtype=torch.long, device=graph.device),
                relabel_nodes=False,
                store_ids=False,
            )

        return local_graph, local_positions, global_nodes


__all__ = ["DGLChannelEdgeIndex", "EdgeTypeMasker"]
