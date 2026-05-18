"""
DGL graph construction utilities for CellSTIC.
"""

import numpy as np
import torch
from typing import Optional, Any
import dgl


class DGLGraphUtils:
    """Utilities for constructing DGL graphs."""

    @staticmethod
    def build_dgl_graph_from_virtual_types_graph(
        spatial_graph: np.ndarray,
        node_features: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        spatial_coords: Optional[torch.Tensor] = None,
        ligand_strength_adj: Optional[np.ndarray] = None,
        receptor_strength_adj: Optional[np.ndarray] = None,
        knn_strength_adj: Optional[np.ndarray] = None,
    ) -> Any:
        """Build DGL graph from 3D spatial graph (n_nodes, n_nodes, n_virtual_types).
        Requires ligand_strength_adj, receptor_strength_adj (n,n,n_virtual_types), knn_strength_adj (n,n,n_virtual_types,n_modalities).
        Edge features: [ligand, receptor, knn_mod0..M, x1, y1, x2, y2, type] (minmax normalized)."""
        n_nodes, _, n_virtual_types = spatial_graph.shape

        src_list, dst_list, vt_list, w_list = [], [], [], []
        lig_list, rec_list, knn_list = [], [], []
        for vt_idx in range(n_virtual_types):
            src, dst = np.where(spatial_graph[:, :, vt_idx] > 0)
            if len(src) == 0:
                continue
            mask = src != dst
            src, dst = src[mask], dst[mask]
            if len(src) == 0:
                continue
            src_list.extend(src)
            dst_list.extend(dst)
            vt_list.extend([vt_idx] * len(src))
            w_list.extend(spatial_graph[src, dst, vt_idx])
            lig_list.extend(ligand_strength_adj[src, dst, vt_idx])
            rec_list.extend(receptor_strength_adj[src, dst, vt_idx])
            for k in range(len(src)):
                knn_list.append(knn_strength_adj[src[k], dst[k], vt_idx, :])

        # Append self-loops so graph is built in one shot (one path)
        n_modalities = knn_strength_adj.shape[3]
        for i in range(n_nodes):
            for vt_idx in range(n_virtual_types):
                src_list.append(i)
                dst_list.append(i)
                vt_list.append(vt_idx)
                w_list.append(1.0)
                lig_list.append(1.0)
                rec_list.append(1.0)
                knn_list.append(np.ones(n_modalities, dtype=np.float32))

        g = dgl.graph((src_list, dst_list), num_nodes=n_nodes).to(device)  # type: ignore
        if node_features is not None:
            g.ndata["node_features"] = (
                torch.tensor(node_features, dtype=torch.float32) if isinstance(node_features, np.ndarray) else node_features
            ).to(device)

        n_edges = len(src_list)
        n_self = n_nodes * n_virtual_types
        n_non_self = n_edges - n_self

        edge_type_indices = torch.tensor(vt_list, dtype=torch.long, device=device)
        g.edata["edge_type"] = torch.nn.functional.one_hot(edge_type_indices, num_classes=n_virtual_types).float()
        g.edata["edge_weight"] = torch.tensor(w_list, dtype=torch.float32, device=device)

        if spatial_coords is not None:
            dev = spatial_coords.device
            src_t = torch.tensor(src_list, dtype=torch.long, device=dev)
            dst_t = torch.tensor(dst_list, dtype=torch.long, device=dev)
            coords = torch.cat([spatial_coords[src_t], spatial_coords[dst_t]], dim=1)

            lig = torch.tensor(lig_list, dtype=torch.float32, device=dev).unsqueeze(1)
            rec = torch.tensor(rec_list, dtype=torch.float32, device=dev).unsqueeze(1)
            knn = torch.tensor(np.stack([k.ravel() for k in knn_list], axis=0), dtype=torch.float32, device=dev)

            head = torch.cat([lig, rec, knn], dim=1)
            n_knn = head.shape[1] - 2
            tail = torch.ones(n_edges, 1, dtype=torch.float32, device=dev)
            raw = torch.cat([head, coords, tail], dim=1)

            # Normalize from non-self edges so self-edges can be set to 1 after
            if n_non_self > 0:
                raw_non_self = raw[:n_non_self]
                head_norm_ns = DGLGraphUtils._minmax_normalize(raw_non_self[:, : 2 + n_knn])
                c_min = raw_non_self[:, 2 + n_knn : 6 + n_knn].min()
                c_max = raw_non_self[:, 2 + n_knn : 6 + n_knn].max()
                c_r = c_max - c_min
                c_r = torch.where(c_r > 0, c_r, torch.ones_like(c_r))
                coords_norm_ns = (raw_non_self[:, 2 + n_knn : 6 + n_knn] - c_min) / c_r
                type_norm_ns = DGLGraphUtils._minmax_normalize(raw_non_self[:, 6 + n_knn : 7 + n_knn])
                self_head = torch.ones(n_self, 2 + n_knn, dtype=torch.float32, device=dev)
                self_coords = (raw[n_non_self:, 2 + n_knn : 6 + n_knn] - c_min) / c_r
                self_type = torch.ones(n_self, 1, dtype=torch.float32, device=dev)
                g.edata["edge_features"] = torch.cat([
                    torch.cat([head_norm_ns, coords_norm_ns, type_norm_ns], dim=1),
                    torch.cat([self_head, self_coords, self_type], dim=1),
                ], dim=0).to(device)
            else:
                self_head = torch.ones(n_self, 2 + n_knn, dtype=torch.float32, device=dev)
                self_coords_raw = raw[:, 2 + n_knn : 6 + n_knn]
                c_min, c_max = self_coords_raw.min(), self_coords_raw.max()
                c_r = torch.where(c_max > c_min, c_max - c_min, torch.ones_like(c_max))
                self_coords = (self_coords_raw - c_min) / c_r
                self_type = torch.ones(n_self, 1, dtype=torch.float32, device=dev)
                g.edata["edge_features"] = torch.cat([self_head, self_coords, self_type], dim=1).to(device)

        return g

    @staticmethod
    def build_dgl_graph(
        spatial_graph: np.ndarray,
        node_features: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        spatial_coords: Optional[torch.Tensor] = None,
    ) -> Any:
        """Build DGL graph from 2D adjacency; optional node features and spatial coords for edge features."""
        src, dst = np.where(spatial_graph > 0)
        g = dgl.graph((src, dst)).to(device)  # type: ignore
        
        # Add node features
        if node_features is not None:
            if isinstance(node_features, np.ndarray):
                node_features = torch.tensor(node_features, dtype=torch.float32)
            g.ndata['node_features'] = node_features.to(device)
        
        # Add edge features [ligand=1, receptor=1, KNN_strength, x1, y1, x2, y2, type=1], normalized per dimension
        if spatial_coords is not None and len(src) > 0:
            device = spatial_coords.device
            src_tensor = torch.from_numpy(src).long().to(device)
            dst_tensor = torch.from_numpy(dst).long().to(device)
            src_coords = spatial_coords[src_tensor]  # (n_edges, 2)
            dst_coords = spatial_coords[dst_tensor]  # (n_edges, 2)
            knn_strength = torch.from_numpy(spatial_graph[src, dst]).float().unsqueeze(1).to(device)
            edge_features = torch.cat([knn_strength, src_coords, dst_coords], dim=1)
            coords = edge_features[:, 1:5]
            c_min, c_max = coords.min(), coords.max()
            c_range = c_max - c_min
            c_range = torch.where(c_range > 0, c_range, torch.ones_like(c_range))
            coords_norm = (coords - c_min) / c_range
            g.edata['edge_features'] = torch.cat([knn_strength, coords_norm], dim=1).to(device)

        return g

    @staticmethod
    def _minmax_normalize(x: torch.Tensor) -> torch.Tensor:
        """Min-max normalization with zero-division protection."""
        x_min = x.min(dim=0, keepdim=True)[0]
        x_max = x.max(dim=0, keepdim=True)[0]
        x_range = x_max - x_min
        x_range = torch.where(x_range == 0, torch.ones_like(x_range), x_range)
        return (x - x_min) / x_range
