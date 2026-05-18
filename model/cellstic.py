from typing import Dict, List, Optional, Tuple

import dgl
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist

from utils.train import ModelConfig, EdgeFilterUtils

from .hodgnn import HODGNN
from .tree import BalancedHierarchyBuilder, BiologicalHierarchyBuilder, LLMHierarchyBuilder
from .graph import (
    BaseGraphUtils,
    ClusterGraphUtils,
    KNNGraphUtils,
    DGLGraphUtils,
)

class CellSTIC(nn.Module):
    def __init__(self, config: ModelConfig, device: torch.device):
        """
        Initialize CellSTIC model.

        Args:
            config: Model configuration
            device: Device
        """
        super().__init__()
        self.config = config
        self.device = device
        feat_config = getattr(config, "feat", None) or getattr(config, "feature", None)
        enc_dims = getattr(feat_config, "encoder_dims", None) if feat_config else None
        if enc_dims and isinstance(enc_dims, list) and len(enc_dims) > 0 and isinstance(enc_dims[0], list):
            self.num_modality = len(enc_dims)
        else:
            self.num_modality = 2

        # Get common dropout from model config
        dropout = getattr(config, 'dropout', 0.1)

        # Initialize components (config.ccc, config.feat, config.graph, config.tree)
        self.ccc_predictor = CCC_Predicor(getattr(config, 'ccc', None), self.device, dropout, self.num_modality)
        self.feat_integrator = Feat_Integrator(getattr(config, 'feat', None), self.device, dropout)
        self.graph_generator = Graph_Generator(getattr(config, 'graph', None), self.device)
        self.tree_builder = Tree_Builder(getattr(config, 'tree', None), self.device)

    def forward(self, g, h, e):
        pass


class CCC_Predicor(nn.Module):
    def __init__(self, config: ModelConfig, device: torch.device, dropout: float, n_modalities: int):
        super().__init__()
        self.config = config
        self.device = device
        self.dropout = dropout

        # Get config parameters
        self.encoder_dims = getattr(config, 'encoder_dims')
        self.decoder_dims = getattr(config, 'decoder_dims')
        self.decoder_head = getattr(config, 'decoder_head')
        # Temperature for sigmoid: <1 sharper probs; >1 flattens toward 0.5
        t = getattr(config, 'temperature', 0.4) if config else 0.4
        self.temperature = 0.4 if t is None else float(t)
        self.n_modalities = n_modalities
        self.encoder_edge_dim = 5 + self.n_modalities
        # EGNN layers (internal edge dim = encoder_edge_dim)
        self.egnn_layers = nn.ModuleList()
        for i in range(len(self.encoder_dims) - 1):
            in_dim = self.encoder_dims[i]
            out_dim = self.encoder_dims[i + 1]
            self.egnn_layers.append(HODGNN(in_dim, out_dim, self.dropout, edge_dim=self.encoder_edge_dim))

        # Structure decoder
        self.structure_decoder = _build_adaptive_mlp(self.decoder_dims, self.dropout)

        # Structure decoder head
        self.structure_decoder_head_layers = nn.ModuleList()

        # Edge type decoder head layers
        self.edge_type_decoder_head_layers = nn.ModuleList()

        # Lazy edge projection: variable input edge dim -> encoder_edge_dim (created on first forward)
        self._edge_projection = None

    def init_head_layers(self, hierarchy_dict: Dict[str, List[Dict]]):
        """
        Initialize head layers based on hierarchy dictionary (only when training).
        Keys: level_1, level_2, ...; head_num = number of groups per level.
        """
        level_keys = sorted(hierarchy_dict.keys(), key=lambda x: int(x.split('_')[1]))
        struct_dim = self.decoder_dims[-1]
        edge_dim = self.encoder_edge_dim * (len(self.egnn_layers) + 1)
        for level_key in level_keys:
            head_num = len(hierarchy_dict[level_key])
            self.structure_decoder_head_layers.append(
                nn.Linear(struct_dim, head_num).to(self.device)
            )
            self.edge_type_decoder_head_layers.append(
                nn.Linear(edge_dim, head_num).to(self.device)
            )

    def _compute_edge_probs_batch(
        self,
        positions: torch.Tensor,
        fused_features: torch.Tensor,
        batch_size: int,
        head_layer: nn.Module,
    ) -> torch.Tensor:
        """Compute edge probabilities in batches. Returns (num_edges, n_types)."""
        num_edges = positions.size(0)
        n_types = getattr(head_layer, "out_features", None)
        if n_types is None and num_edges > 0:
            with torch.no_grad():
                idx = positions[:1]
                pairs = torch.cat([fused_features[idx[:, 0]], fused_features[idx[:, 1]]], dim=-1)
                n_types = head_layer(self.structure_decoder(pairs)).shape[1]
        n_types = n_types or 0
        edge_probs = torch.empty((num_edges, n_types), device=fused_features.device, dtype=torch.float32)
        for start in range(0, num_edges, batch_size):
            end = min(start + batch_size, num_edges)
            pos = positions[start:end]
            pairs = torch.cat([fused_features[pos[:, 0]], fused_features[pos[:, 1]]], dim=-1)
            logits = head_layer(self.structure_decoder(pairs))
            edge_probs[start:end] = torch.sigmoid(logits / self.temperature)
        return edge_probs

    def forward(
        self,
        g_dgl,
        pos_positions: torch.Tensor,
        neg_positions: torch.Tensor,
        level_idx: int,
        mutiple_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward: EGNN encode -> fused features -> pos/neg edge probs + edge type logits.
        Returns (pos_edge_probs, neg_edge_probs, edge_type_predictions, fused_features).
        """
        # Node and edge features from graph or defaults
        ndata = getattr(g_dgl, 'ndata', None)
        edata = getattr(g_dgl, 'edata', None)
        if ndata and 'node_features' in ndata:
            x = ndata['node_features']
        else:
            x = torch.eye(g_dgl.num_nodes(), device=g_dgl.device, dtype=torch.float32)
        if edata and 'edge_features' in edata:
            e = edata['edge_features']
        else:
            e = torch.ones(g_dgl.num_edges(), self.encoder_edge_dim, device=g_dgl.device, dtype=torch.float32)
        if e.shape[1] != self.encoder_edge_dim:
            if self._edge_projection is None:
                self._edge_projection = nn.Linear(e.shape[1], self.encoder_edge_dim).to(self.device)
            e = self._edge_projection(e)

        # EGNN encoding and concatenate edge features from all layers
        all_edge_features = [e]
        for egnn_layer in self.egnn_layers:
            x, e = egnn_layer(g_dgl, x, e)
            all_edge_features.append(e)
        e = torch.cat(all_edge_features, dim=-1)
        fused_features = torch.cat([mutiple_features, x], dim=-1)

        # Pos/neg edge probabilities (batched)
        batch_size = getattr(self, "edge_prob_batch_size", 10000)
        head = self.structure_decoder_head_layers[level_idx]
        pos_edge_probs = (
            self._compute_edge_probs_batch(pos_positions, fused_features, batch_size, head)
            if pos_positions.size(0) > 0
            else torch.empty(0, head.out_features, device=g_dgl.device)
        )
        neg_edge_probs = (
            self._compute_edge_probs_batch(neg_positions, fused_features, batch_size, head)
            if neg_positions.size(0) > 0
            else torch.empty(0, head.out_features, device=g_dgl.device)
        )

        # Edge type logits (level > 0 only)
        edge_type_layer = self.edge_type_decoder_head_layers[level_idx]
        edge_type_predictions = edge_type_layer(e) if level_idx > 0 else None

        return pos_edge_probs, neg_edge_probs, edge_type_predictions, fused_features

class Feat_Integrator(nn.Module):
    def __init__(self, config: ModelConfig, device: torch.device, dropout: float):
        super().__init__()
        self.config = config
        self.device = device
        self.dropout = dropout
        self.encoder_dims = getattr(config, 'encoder_dims')
        self.decoder_dims = getattr(config, 'decoder_dims', None)
        self.output_dims = getattr(config, 'output_dims')
        enc_dims = self.encoder_dims
        dec_dims = self.decoder_dims
        self.num_modality = len(enc_dims)
        self.encoder_edge_dim = 5  # fixed: feature graph edge dim (knn + 4 coords)

        # GNN encoders (one ModuleList of HODGNN per modality)
        self.modality_gnn_encoders = nn.ModuleList()
        for i in range(self.num_modality):
            dims = enc_dims[i]
            layers = nn.ModuleList([
                HODGNN(dims[j], dims[j + 1], self.dropout, edge_dim=self.encoder_edge_dim)
                for j in range(len(dims) - 1)
            ])
            self.modality_gnn_encoders.append(layers)

        # Attribute decoders (one MLP per modality)
        self.attribute_decoders = nn.ModuleList([
            _build_adaptive_mlp(dec_dims[i], self.dropout)
            for i in range(self.num_modality)
        ])
        self.output_layer = _build_adaptive_mlp(self.output_dims, self.dropout)

        # Lazy edge projection: variable input edge dim -> encoder_edge_dim
        self._edge_projection = None

    def forward(
        self, modality_g_dgls: List[dgl.DGLGraph]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Encode each modality graph -> fuse -> output_layer -> fused_features;
        decode each encoded feature -> decoded_features.
        Returns (fused_features, decoded_features).
        """
        encoded_features = [
            self._process_graph(modality_g_dgls[i], self.modality_gnn_encoders[i])
            for i in range(self.num_modality)
        ]
        fused_features = self.output_layer(torch.cat(encoded_features, dim=-1))
        decoded_features = [
            self.attribute_decoders[i](encoded_features[i])
            for i in range(self.num_modality)
        ]
        return fused_features, decoded_features

    def _process_graph(self, g: dgl.DGLGraph, egnn_layers: nn.ModuleList) -> torch.Tensor:
        """Get node/edge features from g, run egnn_layers, return final node features."""
        ndata = getattr(g, 'ndata', None)
        edata = getattr(g, 'edata', None)
        if ndata and 'node_features' in ndata:
            x = ndata['node_features']
        else:
            x = torch.eye(g.num_nodes(), device=g.device, dtype=torch.float32)
        if edata and 'edge_features' in edata:
            e = edata['edge_features']
        else:
            e = torch.ones(g.num_edges(), self.encoder_edge_dim, device=g.device, dtype=torch.float32)
        if e.shape[1] != self.encoder_edge_dim:
            if self._edge_projection is None:
                self._edge_projection = nn.Linear(e.shape[1], self.encoder_edge_dim).to(self.device)
            e = self._edge_projection(e)
        for layer in egnn_layers:
            x, e = layer(g, x, e)
        return x

class Graph_Generator(nn.Module):
    def __init__(self, config: ModelConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device

        # Get config parameters (distance_threshold computed from spatial data; high expression = ≥expression_percentile)
        self.cluster_size = getattr(config, 'cluster_size')
        self.clustering_top_k = getattr(config, 'cluster_top_k')
        self.knn_top_k = getattr(config, 'knn_top_k')
        self.expression_percentile = getattr(config, 'expression_percentile', 75)  # default 75th (upper quartile)
        self.n_spots = getattr(config, 'n_spots', 10)

    def forward(self) -> None:
        """
        Forward pass of Adaptive graph generator.
        """
        pass

    def build_ccc_graph(
        self,
        modality_features,
        spatial_distances,
        gene_expression,
        genes,
        ligand_receptor_map,
        cell_types: Optional[np.ndarray] = None,
        lr_pair_type_constraints: Optional[Dict[str, List[Tuple[int, int]]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build CCC graph.

        Returns:
            base_graph_adj, edge_type_map, ligand_strength_adj, receptor_strength_adj, knn_per_modality
        where knn_per_modality is (n_cells, n_cells, n_modalities), each modality's KNN graph
        for edge features. If cell_types and lr_pair_type_constraints are provided (scmultisim),
        apply per-ligand–receptor (sender_type, receiver_type) constraints to all adjacency tensors.
        """
        distance_threshold = EdgeFilterUtils.compute_spot_distance_threshold(spatial_distances, n_spots=self.n_spots)
        print(f"Distance threshold: {distance_threshold}")
        base_graph_adj, edge_type_map, ligand_strength_adj, receptor_strength_adj = BaseGraphUtils.build_spatial_graph(
            spatial_distances=spatial_distances,
            gene_expression=gene_expression,
            genes=genes,
            distance_threshold=distance_threshold,
            expression_percentile=self.expression_percentile,
            ligand_receptor_map=ligand_receptor_map,
        )
        non_zero_count = np.count_nonzero(base_graph_adj)
        print(f"Base graph non-zero count: {non_zero_count}")
        modality_merged_graphs = [
            self._build_feature_graph(f, spatial_distances, distance_threshold)
            for f in modality_features
        ]
        # Stack all modalities' KNN values (n_cells, n_cells, n_modalities) for edge features, no merging
        knn_per_modality = np.stack([np.asarray(g) for g in modality_merged_graphs], axis=-1)

        # Optional (scmultisim): apply cell-type / LR pair constraints at graph-building time
        base_graph_adj, ligand_strength_adj, receptor_strength_adj, knn_per_modality = EdgeFilterUtils.apply_cell_type_constraints(
            base_graph_adj=base_graph_adj,
            ligand_strength_adj=ligand_strength_adj,
            receptor_strength_adj=receptor_strength_adj,
            knn_per_modality=knn_per_modality,
            edge_type_map=edge_type_map,
            cell_types=cell_types,
            lr_pair_type_constraints=lr_pair_type_constraints,
        )

        return base_graph_adj, edge_type_map, ligand_strength_adj, receptor_strength_adj, knn_per_modality

    def build_feature_dgl_graph(self, modality_features: List[torch.Tensor], spatial_coords: torch.Tensor) -> List:
        """Build DGL graphs per modality from spatial coords. Returns list of DGL graphs."""
        coords_np = spatial_coords.detach().cpu().numpy()
        spatial_distances = cdist(coords_np, coords_np)
        distance_threshold = EdgeFilterUtils.compute_spot_distance_threshold(spatial_distances, n_spots=self.n_spots)
        print(f"Distance threshold: {distance_threshold}")
        return [
            DGLGraphUtils.build_dgl_graph(
                self._build_feature_graph(f, spatial_distances, distance_threshold),
                f, self.device, spatial_coords,
            )
            for f in modality_features
        ]

    def _build_feature_graph(self, modality_features, spatial_distances: Optional[np.ndarray] = None, distance_threshold: Optional[float] = None) -> np.ndarray:
        """Build merged cluster+knn graph; optionally mask by spatial distance."""
        feat_np = modality_features.detach().cpu().numpy()
        cluster_g = ClusterGraphUtils.cluster_and_build_graph_balanced(
            features=feat_np, min_cluster_size=self.cluster_size,
            top_k=self.clustering_top_k, device=self.device,
        )
        knn_g = KNNGraphUtils.build_knn_graph(features=feat_np, top_k=self.knn_top_k)
        out = np.maximum(cluster_g, knn_g)
        if spatial_distances is not None and distance_threshold is not None:
            from scipy.sparse import issparse
            dist = spatial_distances.toarray() if issparse(spatial_distances) else np.asarray(spatial_distances)
            out = out * (dist < distance_threshold).astype(np.float64)
        non_zero_count = np.count_nonzero(out)
        print(f"Non-zero count: {non_zero_count}")
        return out

class Tree_Builder(nn.Module):
    def __init__(self, config: ModelConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device

        # Get config parameters
        self.hierarchy_method = getattr(config, 'hierarchy_method')

    def forward(self, edge_type_map: dict, method: str, cell_chat_db: Optional[Dict] = None) -> Dict[str, List[Dict]]:
        """
        Forward pass of CellSTIC label tree builder.

        Args:
            edge_type_map: Edge type map
            method: Hierarchy method
            cell_chat_db: CellChatDB
        Returns:
            HierarchyTree: Hierarchy tree
        """
        if method == 'biological':
            hierarchy_tree = BiologicalHierarchyBuilder.build(edge_type_map, cell_chat_db=cell_chat_db)
        elif method == 'balanced':
            hierarchy_tree = BalancedHierarchyBuilder.build(edge_type_map)
        elif method == 'llm':
            hierarchy_tree = LLMHierarchyBuilder.build(edge_type_map)
        else:
            raise ValueError(f"Invalid hierarchy method: {method}")

        hierarchy_dict = hierarchy_tree.to_dict()
        return hierarchy_dict

def _build_adaptive_mlp(dims: List[int], dropout: float) -> nn.Module:
    """
    Build adaptive MLP layers based on dimension list.

    Args:
        dims: List of dimensions for each layer
        dropout: Dropout rate (currently not used but kept for compatibility)

    Returns:
        Sequential MLP module
    """
    layers = []
    for i in range(len(dims) - 1):
        linear = nn.Linear(dims[i], dims[i + 1])
        # Initialize with xavier uniform for better stability
        nn.init.xavier_uniform_(linear.weight.data, gain=1.0)
        if linear.bias is not None:
            nn.init.constant_(linear.bias.data, 0.0)
        layers.append(linear)
        if i < len(dims) - 2:  # Don't add activation/dropout after the last layer
            layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())

    return nn.Sequential(*layers)
