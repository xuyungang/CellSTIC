import logging
from typing import Dict, List, Optional, Sequence, Tuple

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse

from model.train import (
    CellSTICCCCModelConfig,
    CellSTICFeatureModelConfig,
    CellSTICGraphConfig,
    CellSTICModelConfig,
    CellSTICTreeConfig,
    EdgeFilterUtils,
)

from .graph import (
    BaseGraphUtils,
    ClusterGraphUtils,
    DGLGraphUtils,
    KNNGraphUtils,
)
from .hodgnn import HODGNN
from .tree import (
    BalancedHierarchyBuilder,
    BiologicalHierarchyBuilder,
    LLMHierarchyBuilder,
)


LOGGER = logging.getLogger(__name__)


def _required_graph_features(
    graph: dgl.DGLGraph,
    *,
    node_width: int,
    edge_width: int,
    context: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return validated graph features without inventing silent defaults."""
    missing = [
        name
        for name, data in (
            ("node_features", graph.ndata),
            ("edge_features", graph.edata),
        )
        if name not in data
    ]
    if missing:
        raise KeyError(f"{context} graph is missing required data: {missing}")
    node_features = graph.ndata["node_features"]
    edge_features = graph.edata["edge_features"]
    expected_node_shape = (graph.num_nodes(), node_width)
    expected_edge_shape = (graph.num_edges(), edge_width)
    if tuple(node_features.shape) != expected_node_shape:
        raise ValueError(
            f"{context} node features must have shape {expected_node_shape}, "
            f"got {tuple(node_features.shape)}"
        )
    if tuple(edge_features.shape) != expected_edge_shape:
        raise ValueError(
            f"{context} edge features must have shape {expected_edge_shape}, "
            f"got {tuple(edge_features.shape)}"
        )
    return node_features, edge_features


class CellSTIC(nn.Module):
    """CellSTIC neural network and its trainable/modeling components."""

    def __init__(
        self,
        config: CellSTICModelConfig,
        device: torch.device,
        random_seed: int = 42,
    ):
        """Initialize the CellSTIC model.

        Args:
            config: Model configuration
            device: Device
        """
        super().__init__()
        self.config = config
        self.device = device
        self.random_seed = random_seed
        self.num_modality = len(config.feat.encoder_dims)

        # Get common dropout from model config
        dropout = config.dropout

        # Initialize components (config.ccc, config.feat, config.graph, config.tree)
        self.ccc_predictor = CCCPredictor(
            config.ccc, self.device, dropout, self.num_modality
        )
        self.feat_integrator = FeatureIntegrator(config.feat, self.device, dropout)
        self.graph_generator = GraphGenerator(
            config.graph, self.device, random_seed=random_seed
        )
        self.tree_builder = HierarchyBuilder(config.tree, self.device)


class CCCPredictor(nn.Module):
    def __init__(
        self,
        config: CellSTICCCCModelConfig,
        device: torch.device,
        dropout: float,
        n_modalities: int,
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.dropout = dropout

        # Get config parameters
        self.encoder_dims = config.encoder_dims
        self.decoder_dims = config.decoder_dims
        # Temperature for sigmoid: <1 sharper probs; >1 flattens toward 0.5.
        self.temperature = config.temperature
        self.edge_prob_batch_size = config.edge_prob_batch_size
        self.n_modalities = n_modalities
        # [ligand, receptor, KNN per modality, src/dst XY, type indicator]
        self.encoder_edge_dim = 7 + self.n_modalities
        # EGNN layers (internal edge dim = encoder_edge_dim)
        self.egnn_layers = nn.ModuleList()
        for i in range(len(self.encoder_dims) - 1):
            in_dim = self.encoder_dims[i]
            out_dim = self.encoder_dims[i + 1]
            self.egnn_layers.append(
                HODGNN(
                    in_dim,
                    out_dim,
                    self.dropout,
                    edge_dim=self.encoder_edge_dim,
                )
            )

        # Structure decoder
        self.structure_decoder = _build_adaptive_mlp(self.decoder_dims, self.dropout)

        # Structure decoder head
        self.structure_decoder_head_layers = nn.ModuleList()

        # Edge type decoder head layers
        self.edge_type_decoder_head_layers = nn.ModuleList()

    def init_head_layers(self, hierarchy_dict: Dict[str, List[Dict]]) -> None:
        """
        Initialize head layers based on hierarchy dictionary (only when training).
        Keys: level_1, level_2, ...; head_num = number of groups per level.
        """
        if self.structure_decoder_head_layers or self.edge_type_decoder_head_layers:
            raise RuntimeError("CCC hierarchy heads have already been initialized")
        if not hierarchy_dict:
            raise ValueError("hierarchy_dict must contain at least one level")
        try:
            level_keys = sorted(hierarchy_dict, key=lambda key: int(key.split("_")[1]))
        except (IndexError, ValueError) as error:
            raise ValueError(
                "Hierarchy level keys must use the form 'level_<integer>'"
            ) from error
        struct_dim = self.decoder_dims[-1]
        edge_dim = self.encoder_edge_dim * (len(self.egnn_layers) + 1)
        for level_key in level_keys:
            head_num = len(hierarchy_dict[level_key])
            if head_num < 1:
                raise ValueError(f"Hierarchy level {level_key!r} is empty")
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
        head_layer: nn.Linear,
        channel_indices: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """Compute edge probabilities in batches. Returns (num_edges, n_types)."""
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("positions must have shape (n_edges, 2)")
        if positions.numel() and (
            positions.min() < 0 or positions.max() >= fused_features.shape[0]
        ):
            raise ValueError("positions contains an out-of-range node index")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if channel_indices is None:
            selected_indices = None
            n_output_channels = head_layer.out_features
        else:
            selected_indices = torch.as_tensor(
                tuple(int(index) for index in channel_indices),
                dtype=torch.long,
                device=head_layer.weight.device,
            )
            if selected_indices.numel() and (
                selected_indices.min() < 0
                or selected_indices.max() >= head_layer.out_features
            ):
                raise ValueError("channel_indices contains an invalid channel")
            n_output_channels = len(selected_indices)
        num_edges = positions.size(0)
        edge_probs = fused_features.new_empty((num_edges, n_output_channels))
        for start in range(0, num_edges, batch_size):
            end = min(start + batch_size, num_edges)
            pos = positions[start:end]
            pairs = torch.cat(
                [
                    fused_features[pos[:, 0]],
                    fused_features[pos[:, 1]],
                ],
                dim=-1,
            )
            hidden = self.structure_decoder(pairs)
            if selected_indices is None:
                logits = head_layer(hidden)
            else:
                logits = F.linear(
                    hidden,
                    head_layer.weight.index_select(0, selected_indices),
                    (
                        head_layer.bias.index_select(0, selected_indices)
                        if head_layer.bias is not None
                        else None
                    ),
                )
            edge_probs[start:end] = torch.sigmoid(logits / self.temperature)
        return edge_probs

    def encode_graph(
        self,
        g_dgl: dgl.DGLGraph,
        multimodal_features: torch.Tensor,
        *,
        collect_edge_features: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode one masked graph and return node and edge representations."""
        x, e = _required_graph_features(
            g_dgl,
            node_width=self.encoder_dims[0],
            edge_width=self.encoder_edge_dim,
            context="CCC",
        )
        if (
            multimodal_features.ndim != 2
            or multimodal_features.shape[0] != g_dgl.num_nodes()
        ):
            raise ValueError(
                "multimodal_features must have one row per CCC graph node, got "
                f"{tuple(multimodal_features.shape)} for {g_dgl.num_nodes()} nodes"
            )

        all_edge_features = [e] if collect_edge_features else None
        for egnn_layer in self.egnn_layers:
            x, e = egnn_layer(g_dgl, x, e)
            if all_edge_features is not None:
                all_edge_features.append(e)
        edge_features = (
            torch.cat(all_edge_features, dim=-1)
            if all_edge_features is not None
            else None
        )
        node_features = torch.cat([multimodal_features, x], dim=-1)
        return node_features, edge_features

    def edge_probabilities(
        self,
        positions: torch.Tensor,
        node_features: torch.Tensor,
        level_idx: int,
        channel_indices: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """Score a caller-bounded edge batch for selected hierarchy channels."""
        if not 0 <= level_idx < len(self.structure_decoder_head_layers):
            raise ValueError(f"Invalid hierarchy level index: {level_idx}")
        return self._compute_edge_probs_batch(
            positions,
            node_features,
            self.edge_prob_batch_size,
            self.structure_decoder_head_layers[level_idx],
            channel_indices,
        )

    def selected_edge_logits(
        self,
        positions: torch.Tensor,
        channel_indices: torch.Tensor,
        node_features: torch.Tensor,
        level_idx: int,
    ) -> torch.Tensor:
        """Score aligned ``(source, target, channel)`` samples without E x T."""
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("positions must have shape (n_samples, 2)")
        if channel_indices.ndim != 1 or len(channel_indices) != len(positions):
            raise ValueError("channel_indices must be a vector aligned with positions")
        head = self.structure_decoder_head_layers[level_idx]
        if channel_indices.numel() and (
            channel_indices.min() < 0 or channel_indices.max() >= head.out_features
        ):
            raise ValueError("channel_indices contains an invalid channel")
        if len(positions) == 0:
            return node_features.new_empty(0)
        pair_features = torch.cat(
            [
                node_features[positions[:, 0]],
                node_features[positions[:, 1]],
            ],
            dim=-1,
        )
        hidden = self.structure_decoder(pair_features)
        selected_weight = head.weight[channel_indices]
        logits = torch.sum(hidden * selected_weight, dim=1)
        if head.bias is not None:
            logits = logits + head.bias[channel_indices]
        return logits / self.temperature

    def edge_type_logits(
        self,
        edge_features: torch.Tensor,
        level_idx: int,
    ) -> torch.Tensor:
        """Return virtual edge-type logits for a caller-provided edge batch."""
        return self.edge_type_decoder_head_layers[level_idx](edge_features)

    def forward(
        self,
        g_dgl: dgl.DGLGraph,
        positions: torch.Tensor,
        level_idx: int,
        multimodal_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return all hierarchy-channel probabilities for selected node pairs."""
        fused_features, _ = self.encode_graph(
            g_dgl, multimodal_features, collect_edge_features=False
        )
        return self.edge_probabilities(
            positions,
            fused_features,
            level_idx,
        )


class FeatureIntegrator(nn.Module):
    def __init__(
        self,
        config: CellSTICFeatureModelConfig,
        device: torch.device,
        dropout: float,
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.dropout = dropout
        self.encoder_dims = config.encoder_dims
        self.decoder_dims = config.decoder_dims
        self.output_dims = config.output_dims
        enc_dims = self.encoder_dims
        dec_dims = self.decoder_dims
        self.num_modality = len(enc_dims)
        self.encoder_edge_dim = 5  # fixed: feature graph edge dim (knn + 4 coords)

        # GNN encoders (one ModuleList of HODGNN per modality)
        self.modality_gnn_encoders = nn.ModuleList()
        for i in range(self.num_modality):
            dims = enc_dims[i]
            layers = nn.ModuleList(
                [
                    HODGNN(
                        dims[j],
                        dims[j + 1],
                        self.dropout,
                        edge_dim=self.encoder_edge_dim,
                    )
                    for j in range(len(dims) - 1)
                ]
            )
            self.modality_gnn_encoders.append(layers)

        # Attribute decoders (one MLP per modality)
        self.attribute_decoders = nn.ModuleList(
            [
                _build_adaptive_mlp(dec_dims[i], self.dropout)
                for i in range(self.num_modality)
            ]
        )
        self.output_layer = _build_adaptive_mlp(self.output_dims, self.dropout)

    def forward(
        self, modality_g_dgls: List[dgl.DGLGraph]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Encode each modality graph -> fuse -> output_layer -> fused_features;
        decode each encoded feature -> decoded_features.
        Returns (fused_features, decoded_features).
        """
        encoded_features = self._encode_modalities(modality_g_dgls)
        fused_features = self.output_layer(torch.cat(encoded_features, dim=-1))
        decoded_features = [
            self.attribute_decoders[i](encoded_features[i])
            for i in range(self.num_modality)
        ]
        return fused_features, decoded_features

    def encode(self, modality_g_dgls: Sequence[dgl.DGLGraph]) -> torch.Tensor:
        """Encode and fuse modalities without running reconstruction decoders."""
        encoded_features = self._encode_modalities(modality_g_dgls)
        return self.output_layer(torch.cat(encoded_features, dim=-1))

    def _encode_modalities(
        self, modality_g_dgls: Sequence[dgl.DGLGraph]
    ) -> List[torch.Tensor]:
        if len(modality_g_dgls) != self.num_modality:
            raise ValueError(
                f"Expected {self.num_modality} modality graphs, got "
                f"{len(modality_g_dgls)}"
            )
        return [
            self._process_graph(modality_g_dgls[i], self.modality_gnn_encoders[i])
            for i in range(self.num_modality)
        ]

    def _process_graph(
        self, g: dgl.DGLGraph, egnn_layers: nn.ModuleList
    ) -> torch.Tensor:
        """Get node/edge features from g, run egnn_layers, return final node features."""
        if not egnn_layers:
            raise ValueError("Each modality encoder must contain at least one layer")
        x, e = _required_graph_features(
            g,
            node_width=egnn_layers[0].in_features,
            edge_width=self.encoder_edge_dim,
            context="Feature",
        )
        for layer in egnn_layers:
            x, e = layer(g, x, e)
        return x


class GraphGenerator(nn.Module):
    def __init__(
        self,
        config: CellSTICGraphConfig,
        device: torch.device,
        random_seed: int = 42,
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.random_seed = random_seed

        self.cluster_size = config.cluster_size
        self.clustering_top_k = config.cluster_top_k
        self.knn_top_k = config.knn_top_k
        self.expression_percentile = config.expression_percentile
        self.n_spots = config.n_spots
        self.max_spatial_neighbors = config.max_spatial_neighbors
        self._feature_graph_cache = None

    @staticmethod
    def _feature_cache_key(modality_features, spatial_coords) -> Tuple:
        feature_key = tuple(
            (
                int(feature.data_ptr()),
                int(getattr(feature, "_version", 0)),
                tuple(feature.shape),
            )
            for feature in modality_features
        )
        spatial_key = (
            int(spatial_coords.data_ptr()),
            int(getattr(spatial_coords, "_version", 0)),
            tuple(spatial_coords.shape),
        )
        return feature_key, spatial_key

    def build_ccc_graph(
        self,
        modality_features,
        spatial_coords,
        gene_expression,
        genes,
        ligand_receptor_map,
    ):
        """Build CCC graph.

        Returns:
            base_graph_adj, edge_type_map, ligand_strength_adj, receptor_strength_adj, knn_per_modality
        Typed adjacency/strength values are stored as CSR channels, and
        ``knn_per_modality`` is a tuple of CSR matrices. Candidate edges are
        determined only from spatial proximity, expression evidence, and the
        supplied ligand-receptor event map.
        """
        coords_np = (
            spatial_coords.detach().cpu().numpy()
            if isinstance(spatial_coords, torch.Tensor)
            else np.asarray(spatial_coords)
        )
        distance_threshold = EdgeFilterUtils.compute_spot_distance_threshold(
            coords_np, n_spots=self.n_spots
        )
        LOGGER.info("CCC spatial distance threshold: %s", distance_threshold)
        candidate_pairs = EdgeFilterUtils.spatial_candidate_pairs(
            coords_np,
            distance_threshold,
            max_neighbors=self.max_spatial_neighbors,
        )
        (
            base_graph_adj,
            edge_type_map,
            ligand_strength_adj,
            receptor_strength_adj,
        ) = BaseGraphUtils.build_spatial_graph(
            spatial_coords=coords_np,
            gene_expression=gene_expression,
            genes=genes,
            distance_threshold=distance_threshold,
            expression_percentile=self.expression_percentile,
            ligand_receptor_map=ligand_receptor_map,
            candidate_pairs=candidate_pairs,
        )
        LOGGER.info("CCC base graph stored edges: %d", base_graph_adj.nnz)
        cache_key = self._feature_cache_key(modality_features, spatial_coords)
        if (
            self._feature_graph_cache is not None
            and self._feature_graph_cache[0] == cache_key
        ):
            modality_merged_graphs = self._feature_graph_cache[1]
        else:
            modality_merged_graphs = tuple(
                self._build_feature_graph(feature, candidate_pairs)
                for feature in modality_features
            )
        # Retain one sparse fused cluster+KNN matrix per modality for edge features.
        knn_per_modality = tuple(modality_merged_graphs)

        return (
            base_graph_adj,
            edge_type_map,
            ligand_strength_adj,
            receptor_strength_adj,
            knn_per_modality,
        )

    def build_feature_dgl_graph(
        self,
        modality_features: List[torch.Tensor],
        spatial_coords: torch.Tensor,
    ) -> List:
        """Build DGL graphs per modality from spatial coords. Returns list of DGL graphs."""
        coords_np = spatial_coords.detach().cpu().numpy()
        distance_threshold = EdgeFilterUtils.compute_spot_distance_threshold(
            coords_np, n_spots=self.n_spots
        )
        LOGGER.info("Feature spatial distance threshold: %s", distance_threshold)
        candidate_pairs = EdgeFilterUtils.spatial_candidate_pairs(
            coords_np,
            distance_threshold,
            max_neighbors=self.max_spatial_neighbors,
        )
        feature_adjacencies = tuple(
            self._build_feature_graph(feature, candidate_pairs)
            for feature in modality_features
        )
        self._feature_graph_cache = (
            self._feature_cache_key(modality_features, spatial_coords),
            feature_adjacencies,
        )
        return [
            DGLGraphUtils.build_dgl_graph(
                adjacency,
                feature,
                self.device,
                spatial_coords,
            )
            for feature, adjacency in zip(modality_features, feature_adjacencies)
        ]

    def _build_feature_graph(
        self,
        modality_features: torch.Tensor,
        candidate_pairs: np.ndarray,
    ) -> sparse.csr_matrix:
        """Build and fuse balanced-cluster and KNN feature graphs."""
        feat_np = modality_features.detach().cpu().numpy()
        cluster_graph = ClusterGraphUtils.cluster_and_build_graph_balanced(
            features=feat_np,
            min_cluster_size=self.cluster_size,
            top_k=self.clustering_top_k,
            candidate_pairs=candidate_pairs,
            device=self.device,
            random_seed=self.random_seed,
        )
        knn_graph = KNNGraphUtils.build_knn_graph(
            features=feat_np,
            top_k=self.knn_top_k,
            candidate_pairs=candidate_pairs,
        )
        out = cluster_graph.maximum(knn_graph).tocsr()
        LOGGER.info("Feature graph stored edges: %d", out.nnz)
        return out


class HierarchyBuilder(nn.Module):
    def __init__(self, config: CellSTICTreeConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device

        # Get config parameters
        self.hierarchy_method = config.hierarchy_method
        self.height = config.height

    def forward(
        self,
        edge_type_map: dict,
        cell_chat_db: Optional[Dict] = None,
    ) -> Dict[str, List[Dict]]:
        """
        Build the CellSTIC ligand-receptor hierarchy.

        Args:
            edge_type_map: Edge type map
            cell_chat_db: CellChatDB
        Returns:
            Serialized hierarchy levels.
        """
        if self.hierarchy_method == "biological":
            hierarchy_tree = BiologicalHierarchyBuilder.build(
                edge_type_map, cell_chat_db=cell_chat_db, height=self.height
            )
        elif self.hierarchy_method == "balanced":
            hierarchy_tree = BalancedHierarchyBuilder.build(edge_type_map, height=self.height)
        elif self.hierarchy_method == "llm":
            hierarchy_tree = LLMHierarchyBuilder.build(
                edge_type_map, cell_chat_db=cell_chat_db, height=self.height
            )
        else:
            raise ValueError(f"Invalid hierarchy method: {self.hierarchy_method}")

        hierarchy_dict = hierarchy_tree.to_dict()
        return hierarchy_dict


def _build_adaptive_mlp(dims: Sequence[int], dropout: float) -> nn.Module:
    """
    Build adaptive MLP layers based on dimension list.

    Args:
        dims: List of dimensions for each layer
        dropout: Dropout rate between hidden layers

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


__all__ = [
    "CCCPredictor",
    "CellSTIC",
    "FeatureIntegrator",
    "GraphGenerator",
    "HierarchyBuilder",
]
