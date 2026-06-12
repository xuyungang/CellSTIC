"""CellSTIC evaluation: CCC precision (tree levels), multiple feature (clustering + viz)."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from anndata import AnnData

from model import CellSTIC
from model.graph import DGLGraphUtils, EdgeTypeMapper
from model.train import CellSTICConfig, CellSTICTrainArtifacts, EdgeFilterUtils
from utils.tools import ClusteringUtils


class CellSTICEvaluator:
    def __init__(
        self,
        model: CellSTIC,
        config: CellSTICConfig,
        ligand_receptor_map: Dict,
        train_artifacts: CellSTICTrainArtifacts,
        output_path: Union[str, Path],
        lr_pair_type_constraints: Optional[Dict[str, List[tuple]]] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
        self.model.to(self.device)
        self.feat_integrator = model.feat_integrator
        self.graph_generator = model.graph_generator
        self.precision_model = model.ccc_predictor
        self.ligand_receptor_map = ligand_receptor_map
        self.lr_pair_type_constraints = lr_pair_type_constraints
        self.train_artifacts = train_artifacts
        self.output_path = str(Path(output_path))

    def _modality_g_dgls_on_device(self) -> List:
        return [g.to(self.device) for g in self.train_artifacts.modality_g_dgls]

    def evaluate_ccc_precision_tree(
        self,
        primary_adata: AnnData,
        modality1_features: torch.Tensor,
        spatial_coords: torch.Tensor,
        annotation_key: str = "cell_type",
    ) -> List[Dict[str, Any]]:
        """Per-level CCC evaluation using in-memory graph artifacts from training."""
        base_output_dir = Path(self.output_path)
        art = self.train_artifacts

        enhanced_base_graph = art.base_graph_adj
        model_edge_type_map = art.edge_type_map
        hierarchy_dict = art.hierarchy_dict
        ligand_adj = art.ligand_strength_adj
        receptor_adj = art.receptor_strength_adj
        knn_per_mod = art.knn_per_modality

        hierarchy_levels = sorted(
            [k for k in hierarchy_dict if k.startswith("level_")],
            key=lambda x: int(x.split("_")[1]),
        )

        self.precision_model.eval()
        self.feat_integrator.eval()
        with torch.no_grad():
            modality_g_dgls = self._modality_g_dgls_on_device()
            fused_features, _ = self.feat_integrator(modality_g_dgls)

        results = []
        for level_idx, level_key in enumerate(hierarchy_levels):
            level_num = int(level_key.split("_")[1])
            level_output_dir = base_output_dir / f"tree_level_{level_num}"
            level_data = hierarchy_dict[level_key]
            pos_edge_probs = self._get_frozen_ccc_model_outputs_for_level(
                level_idx,
                level_data,
                enhanced_base_graph,
                model_edge_type_map,
                modality1_features,
                spatial_coords,
                fused_features,
                ligand_strength_adj=ligand_adj,
                receptor_strength_adj=receptor_adj,
                knn_per_modality=knn_per_mod,
            )
            if isinstance(pos_edge_probs, torch.Tensor):
                for k in range(pos_edge_probs.shape[2]):
                    pos_edge_probs[:, :, k].fill_diagonal_(0)
            else:
                for k in range(pos_edge_probs.shape[2]):
                    np.fill_diagonal(pos_edge_probs[:, :, k], 0)
            pos_edge_probs = EdgeFilterUtils.apply_distance_constraint(
                pos_edge_probs, spatial_coords, n_spots=self.graph_generator.n_spots
            )
            pos_probs_np = (
                pos_edge_probs.detach().cpu().numpy()
                if isinstance(pos_edge_probs, torch.Tensor)
                else np.asarray(pos_edge_probs)
            )
            group_names = sorted(level_data.keys())
            edge_type_map = {name: i for i, name in enumerate(group_names)}
            if (
                self.lr_pair_type_constraints
                and annotation_key in primary_adata.obs
                and level_key == hierarchy_levels[-1]
            ):
                pos_probs_np = EdgeFilterUtils._apply_pair_type_constraints_core(
                    pos_probs_np,
                    edge_type_map=edge_type_map,
                    cell_types=primary_adata.obs[annotation_key].to_numpy(),
                    lr_pair_type_constraints=self.lr_pair_type_constraints,
                    log=True,
                )
            results.append(
                {
                    "level_key": level_key,
                    "level_num": level_num,
                    "pos_edge_probs_np": pos_probs_np,
                    "edge_type_map": edge_type_map,
                    "output_dir": level_output_dir,
                    "adata": primary_adata,
                    "hierarchy_dict": hierarchy_dict,
                }
            )

        return results

    def _get_frozen_ccc_model_outputs_for_level(
        self,
        level_idx: int,
        level_data: Dict[str, Any],
        enhanced_base_graph: np.ndarray,
        model_edge_type_map: Dict[str, int],
        modality1_features: torch.Tensor,
        spatial_coords: torch.Tensor,
        fused_features: torch.Tensor,
        ligand_strength_adj: Optional[np.ndarray] = None,
        receptor_strength_adj: Optional[np.ndarray] = None,
        knn_per_modality: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Run precision model for one level; return dense (n_cells, n_cells, n_types) tensor."""
        if ligand_strength_adj is not None and receptor_strength_adj is not None and knn_per_modality is not None:
            mapped_base_graph, mapped_ligand, mapped_receptor, knn_strength_adj = (
                EdgeTypeMapper.map_level_graph_and_strengths(
                    enhanced_base_graph,
                    ligand_strength_adj,
                    receptor_strength_adj,
                    np.asarray(knn_per_modality),
                    level_data,
                    model_edge_type_map,
                )
            )
            g_dgl = DGLGraphUtils.build_dgl_graph_from_virtual_types_graph(
                mapped_base_graph,
                node_features=modality1_features,
                device=self.device,
                spatial_coords=spatial_coords,
                ligand_strength_adj=mapped_ligand,
                receptor_strength_adj=mapped_receptor,
                knn_strength_adj=knn_strength_adj,
            )
        else:
            mapped_base_graph = EdgeTypeMapper.map_edge_types_to_virtual_types(
                enhanced_base_graph, level_data, model_edge_type_map
            )
            g_dgl = DGLGraphUtils.build_dgl_graph_from_virtual_types_graph(
                mapped_base_graph,
                node_features=modality1_features,
                device=self.device,
                spatial_coords=spatial_coords,
            )
        n_cells = mapped_base_graph.shape[0]
        i_indices = torch.arange(n_cells, device=self.device)
        j_indices = torch.arange(n_cells, device=self.device)
        pos_positions = torch.cartesian_prod(i_indices, j_indices)
        neg_empty = torch.empty((0, 2), dtype=torch.long, device=self.device)
        with torch.no_grad():
            pos_edge_probs, _, _, _ = self.precision_model(
                g_dgl, pos_positions, neg_empty, level_idx, fused_features
            )
        head_layer = self.precision_model.structure_decoder_head_layers[level_idx]
        n_types = getattr(head_layer, "out_features", pos_edge_probs.shape[1] if pos_edge_probs.dim() >= 2 else 0)
        return pos_edge_probs.reshape(n_cells, n_cells, n_types).detach().cpu()

    def evaluate_mutiple_feature(
        self,
        primary_adata: AnnData,
        true_labels: Optional[np.ndarray] = None,
        auto_n_clusters: Optional[int] = None,
    ) -> None:
        """Fused features -> Louvain clustering; update adata in memory."""
        self.feat_integrator.eval()
        with torch.no_grad():
            modality_g_dgls = self._modality_g_dgls_on_device()
            fused_features, _ = self.feat_integrator(modality_g_dgls)

        fused_np = fused_features.detach().cpu().numpy()
        n_clusters = auto_n_clusters or ClusteringUtils.auto_n_clusters(fused_np.shape[0])
        cluster_labels = ClusteringUtils.cluster_louvain(fused_np, primary_adata.obs_names, n_clusters=n_clusters)
        if true_labels is not None:
            cluster_labels = ClusteringUtils.align_labels(cluster_labels, true_labels)

        unique_clusters = sorted(np.unique(cluster_labels.astype(str)))
        primary_adata.obs["cluster"] = pd.Categorical(cluster_labels.astype(str), categories=unique_clusters)
        primary_adata.obsm["out"] = fused_np
