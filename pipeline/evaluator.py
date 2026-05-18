"""CellSTIC evaluation: CCC precision (single level or tree), multiple feature (clustering + viz)."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pickle
import torch
from anndata import AnnData

from model import CellSTIC
from model.graph import DGLGraphUtils, EdgeTypeMapper
from utils.metrics import MetricsComputer
from utils.train import ClusteringUtils, EdgeFilterUtils, ExperimentConfig, GraphIOUtils


class CellSTICEvaluator:
    def __init__(
        self,
        model: CellSTIC,
        config: ExperimentConfig,
        ligand_receptor_map: Dict,
        model_path: Union[str, Path],
        output_path: Union[str, Path],
        lr_pair_type_constraints: Optional[Dict[str, List[Tuple[int, int]]]] = None,
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
        self.tree_builder = model.tree_builder
        self.ligand_receptor_map = ligand_receptor_map
        # Optional (scmultisim): per-ligand–receptor allowed (sender_type, receiver_type) pairs
        self.lr_pair_type_constraints = lr_pair_type_constraints
        self.model_path = str(Path(model_path))
        self.output_path = str(Path(output_path))

    @staticmethod
    def _channel_label(s: str) -> str:
        return str(s).replace(":", "_").replace("/", "_").replace(" ", "_").strip() or "channel"

    def evaluate_ccc_precision(
        self,
        modality_datas: List[AnnData],
        ccc_label: Optional[np.ndarray] = None,
        label_edge_type_map: Optional[Dict[str, int]] = None,
        annotation_key: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict[str, int], AnnData]:
        """Get CCC model outputs (last level), apply distance/annotation filters, optional ROC/PR; return probs, edge_type_map, adata with obsm['out']."""
        m0 = modality_datas[0]
        n_cells = m0.n_obs
        out_dir = Path(self.output_path)
        csv_dir = out_dir / "pos_probs_dense"
        cell_names = m0.obs_names.tolist()

        modality1_features = torch.tensor(m0.obsm["feat"], dtype=torch.float32).to(self.device)
        spatial_coords = torch.tensor(m0.obsm["spatial"], dtype=torch.float32).to(self.device)
        pos_edge_probs, edge_type_map, fused_features = self._get_frozen_ccc_model_outputs(
            modality1_features, spatial_coords, label_edge_type_map
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
        pos_edge_probs_np = pos_edge_probs.detach().cpu().numpy() if isinstance(pos_edge_probs, torch.Tensor) else np.asarray(pos_edge_probs)
        if edge_type_map is not None:
            csv_dir.mkdir(parents=True, exist_ok=True)
            for name, ch_idx in sorted(edge_type_map.items(), key=lambda x: x[1]):
                mat = np.asarray(pos_edge_probs_np[:, :, ch_idx], dtype=float)
                mat[~np.isfinite(mat)] = np.nan
                pd.DataFrame(mat, index=cell_names, columns=cell_names).to_csv(csv_dir / f"{self._channel_label(name)}.csv", index=True)

        if self.lr_pair_type_constraints and annotation_key and annotation_key in m0.obs and edge_type_map is not None:
            pos_edge_probs_np = EdgeFilterUtils._apply_pair_type_constraints_core(
                pos_edge_probs_np, edge_type_map=edge_type_map, cell_types=m0.obs[annotation_key].to_numpy(),
                lr_pair_type_constraints=self.lr_pair_type_constraints, log=True,
            )
        n_cells, _, n_types = pos_edge_probs_np.shape
        eval_mask = np.ones((n_cells, n_cells, n_types), dtype=bool)
        for k in range(n_types):
            np.fill_diagonal(eval_mask[:, :, k], False)
        MetricsComputer.plot_roc_pr_and_save_csv(
            pos_edge_probs_np, ccc_label, save_dir=out_dir,
            lr_pair_names=[k for k, _ in sorted(edge_type_map.items(), key=lambda x: x[1])] if edge_type_map else None,
            cell_names=cell_names, eval_mask=eval_mask, export_matrix_csv=False,
        )
        m0.obsm["out"] = (fused_features.detach().cpu().numpy() if isinstance(fused_features, torch.Tensor) else np.asarray(fused_features)) if fused_features is not None else m0.obsm.get("out", m0.obsm["feat"].copy())
        return pos_edge_probs_np, edge_type_map, m0

    def evaluate_ccc_precision_tree(
        self,
        modality_datas: List[AnnData],
    ) -> List[Dict[str, Any]]:
        """Per-level CCC evaluation: load graph+hierarchy, run precision model per level, apply distance filter, save CSV per channel."""
        m0 = modality_datas[0]
        n_cells = m0.n_obs
        cell_names = m0.obs_names.tolist()
        base_output_dir = Path(self.output_path)

        enhanced_base_graph, model_edge_type_map, hierarchy_dict = self._load_saved_graph_data()
        hierarchy_levels = sorted([k for k in hierarchy_dict if k.startswith("level_")], key=lambda x: int(x.split("_")[1]))

        modality1_features = torch.tensor(m0.obsm["feat"], dtype=torch.float32).to(self.device)
        spatial_coords = torch.tensor(m0.obsm["spatial"], dtype=torch.float32).to(self.device)
        self.precision_model.eval()
        self.feat_integrator.eval()
        with torch.no_grad():
            modality_g_dgls = GraphIOUtils.load_feature_graphs(self.model_path, self.device)
            fused_features, _ = self.feat_integrator(modality_g_dgls)
        ligand_adj, receptor_adj, knn_per_mod = self._load_strength_arrays()

        results = []
        for level_idx, level_key in enumerate(hierarchy_levels):
            level_num = int(level_key.split("_")[1])
            level_output_dir = base_output_dir / f"tree_level_{level_num}"
            level_output_dir.mkdir(parents=True, exist_ok=True)
            level_data = hierarchy_dict[level_key]
            pos_edge_probs_level = self._get_frozen_ccc_model_outputs_for_level(
                level_idx, level_data, enhanced_base_graph, model_edge_type_map,
                modality1_features, spatial_coords, fused_features,
                ligand_strength_adj=ligand_adj, receptor_strength_adj=receptor_adj, knn_per_modality=knn_per_mod,
            )
            pos_edge_probs_level = EdgeFilterUtils.apply_distance_constraint(
                pos_edge_probs_level, spatial_coords, n_spots=self.graph_generator.n_spots
            )
            pos_probs_np = pos_edge_probs_level.detach().cpu().numpy() if isinstance(pos_edge_probs_level, torch.Tensor) else np.asarray(pos_edge_probs_level)
            for k in range(pos_probs_np.shape[2]):
                np.fill_diagonal(pos_probs_np[:, :, k], 0)
            group_names = sorted(level_data.keys())
            edge_type_map = {name: i for i, name in enumerate(group_names)}
            csv_dir = level_output_dir / "pos_probs_dense"
            csv_dir.mkdir(parents=True, exist_ok=True)
            for name, ch_idx in edge_type_map.items():
                mat = np.asarray(pos_probs_np[:, :, ch_idx], dtype=float)
                mat[~np.isfinite(mat)] = np.nan
                pd.DataFrame(mat, index=cell_names, columns=cell_names).to_csv(csv_dir / f"{self._channel_label(name)}.csv", index=True)
            results.append({
                "level_key": level_key,
                "level_num": level_num,
                "pos_edge_probs_np": pos_probs_np,
                "edge_type_map": edge_type_map,
                "output_dir": level_output_dir,
                "adata": m0,
                "hierarchy_dict": hierarchy_dict,
            })
        return results

    def _load_saved_graph_data(self) -> Tuple[np.ndarray, Dict[str, int], Dict[str, Any]]:
        """Load enhanced_base_graph.npy, edge_type_map.pkl, hierarchy_dict.pkl from model path."""
        save_dir = Path(self.model_path)
        for name in ("enhanced_base_graph.npy", "edge_type_map.pkl", "hierarchy_dict.pkl"):
            if not (save_dir / name).exists():
                raise FileNotFoundError(f"Saved graph data missing under {save_dir}: need {name}")
        enhanced_base_graph = np.load(save_dir / "enhanced_base_graph.npy", mmap_mode="r")
        with open(save_dir / "edge_type_map.pkl", "rb") as f:
            model_edge_type_map = pickle.load(f)
        with open(save_dir / "hierarchy_dict.pkl", "rb") as f:
            hierarchy_dict = pickle.load(f)
        return enhanced_base_graph, model_edge_type_map, hierarchy_dict

    def _load_strength_arrays(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Load ligand_strength_adj, receptor_strength_adj, knn_per_modality if present. Return (lig, rec, knn) or (None, None, None)."""
        save_dir = Path(self.model_path)
        lig = np.load(save_dir / "ligand_strength_adj.npy") if (save_dir / "ligand_strength_adj.npy").exists() else None
        rec = np.load(save_dir / "receptor_strength_adj.npy") if (save_dir / "receptor_strength_adj.npy").exists() else None
        knn = np.load(save_dir / "knn_per_modality.npy") if (save_dir / "knn_per_modality.npy").exists() else None
        return lig, rec, knn

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
        if (ligand_strength_adj is not None and receptor_strength_adj is not None and knn_per_modality is not None):
            mapped_base_graph, mapped_ligand, mapped_receptor, knn_strength_adj = EdgeTypeMapper.map_level_graph_and_strengths(
                enhanced_base_graph, ligand_strength_adj, receptor_strength_adj,
                np.asarray(knn_per_modality), level_data, model_edge_type_map,
            )
            g_dgl = DGLGraphUtils.build_dgl_graph_from_virtual_types_graph(
                mapped_base_graph, node_features=modality1_features, device=self.device, spatial_coords=spatial_coords,
                ligand_strength_adj=mapped_ligand, receptor_strength_adj=mapped_receptor, knn_strength_adj=knn_strength_adj,
            )
        else:
            mapped_base_graph = EdgeTypeMapper.map_edge_types_to_virtual_types(
                enhanced_base_graph, level_data, model_edge_type_map
            )
            g_dgl = DGLGraphUtils.build_dgl_graph_from_virtual_types_graph(
                mapped_base_graph, node_features=modality1_features, device=self.device, spatial_coords=spatial_coords
            )
        n_cells = mapped_base_graph.shape[0]
        rows, cols = np.where(~np.eye(n_cells, dtype=bool))
        pos_positions = torch.tensor(np.column_stack([rows, cols]), dtype=torch.long, device=self.device)
        neg_empty = torch.empty((0, 2), dtype=torch.long, device=self.device)
        with torch.no_grad():
            pos_edge_probs, _, _, _ = self.precision_model(
                g_dgl, pos_positions, neg_empty, level_idx, fused_features
            )
        head_layer = self.precision_model.structure_decoder_head_layers[level_idx]
        n_types = getattr(head_layer, "out_features", pos_edge_probs.shape[1] if pos_edge_probs.dim() >= 2 else 0)
        pos_probs_dense = torch.zeros((n_cells, n_cells, n_types), dtype=torch.float32, device="cpu")
        if pos_positions.numel() > 0 and n_types > 0:
            probs_cpu = pos_edge_probs.detach().cpu()
            E = probs_cpu.shape[0]
            i_idx = pos_positions[:, 0].cpu().unsqueeze(1).expand(E, n_types)
            j_idx = pos_positions[:, 1].cpu().unsqueeze(1).expand(E, n_types)
            k_idx = torch.arange(n_types, dtype=torch.long, device="cpu").unsqueeze(0).expand(E, n_types)
            pos_probs_dense.index_put_((i_idx, j_idx, k_idx), probs_cpu)
        return pos_probs_dense

    def evaluate_mutiple_feature(
        self,
        modality_datas: List[AnnData],
        true_labels: Optional[np.ndarray] = None,
        auto_n_clusters: Optional[int] = None,
    ) -> None:
        """Fused features -> Louvain clustering, optional label alignment, UMAP + spatial viz, clustering metrics."""
        m0 = modality_datas[0]
        self.feat_integrator.eval()
        with torch.no_grad():
            modality_g_dgls = GraphIOUtils.load_feature_graphs(self.model_path, self.device)
            fused_features, _ = self.feat_integrator(modality_g_dgls)

        fused_np = fused_features.detach().cpu().numpy()
        n_clusters = auto_n_clusters or ClusteringUtils.auto_n_clusters(fused_np.shape[0])
        cluster_labels = ClusteringUtils.cluster_louvain(
            fused_np, m0.obs_names, n_clusters=n_clusters
        )
        if true_labels is not None:
            cluster_labels = ClusteringUtils.align_labels(cluster_labels, true_labels)

        unique_clusters = sorted(np.unique(cluster_labels.astype(str)))
        m0.obs["cluster"] = pd.Categorical(cluster_labels.astype(str), categories=unique_clusters)
        m0.obsm["out"] = fused_np
        MetricsComputer.run_region_umap_metrics_export(
            m0, save_dir=self.output_path, feature_key="out", cluster_key="cluster", true_labels=true_labels
        )

    def _get_frozen_ccc_model_outputs(
        self,
        modality1_features: torch.Tensor,
        spatial_coords: torch.Tensor,
        label_edge_type_map: Optional[Dict[str, int]] = None,
    ) -> Tuple[Any, Dict[str, int], torch.Tensor]:
        """Load saved graph+hierarchy, run precision model on last level, return (pos_edge_probs, edge_type_map, fused_features)."""
        self.precision_model.eval()
        self.feat_integrator.eval()
        with torch.no_grad():
            modality_g_dgls = GraphIOUtils.load_feature_graphs(self.model_path, self.device)
            fused_features, _ = self.feat_integrator(modality_g_dgls)

            save_dir = Path(self.model_path)
            enhanced_base_graph = np.load(save_dir / "enhanced_base_graph.npy")
            with open(save_dir / "edge_type_map.pkl", "rb") as f:
                model_edge_type_map = pickle.load(f)
            with open(save_dir / "hierarchy_dict.pkl", "rb") as f:
                hierarchy_dict = pickle.load(f)

            hierarchy_levels = sorted([k for k in hierarchy_dict if k.startswith("level_")])
            level_key = hierarchy_levels[-1]
            level_data = hierarchy_dict[level_key]
            level_idx = len(hierarchy_levels) - 1

            lig_adj, rec_adj, knn_per_mod = self._load_strength_arrays()
            if lig_adj is not None and rec_adj is not None and knn_per_mod is not None:
                mapped_base_graph, mapped_ligand, mapped_receptor, knn_strength_adj = EdgeTypeMapper.map_level_graph_and_strengths(
                    enhanced_base_graph, lig_adj, rec_adj, np.asarray(knn_per_mod), level_data, model_edge_type_map,
                )
                g_dgl = DGLGraphUtils.build_dgl_graph_from_virtual_types_graph(
                    mapped_base_graph, node_features=modality1_features, device=self.device, spatial_coords=spatial_coords,
                    ligand_strength_adj=mapped_ligand, receptor_strength_adj=mapped_receptor, knn_strength_adj=knn_strength_adj,
                )
            else:
                mapped_base_graph = EdgeTypeMapper.map_edge_types_to_virtual_types(
                    enhanced_base_graph, level_data, model_edge_type_map
                )
                g_dgl = DGLGraphUtils.build_dgl_graph_from_virtual_types_graph(
                    mapped_base_graph, node_features=modality1_features, device=self.device, spatial_coords=spatial_coords
                )

            n_cells = mapped_base_graph.shape[0]
            i_indices = torch.arange(n_cells, device=self.device)
            j_indices = torch.arange(n_cells, device=self.device)
            pos_positions = torch.cartesian_prod(i_indices, j_indices)
            neg_empty = torch.empty((0, 2), dtype=torch.long, device=self.device)
            pos_edge_probs, _, _, fused_features = self.precision_model(
                g_dgl, pos_positions, neg_empty, level_idx, fused_features
            )
            pos_edge_probs = pos_edge_probs.reshape(n_cells, n_cells, pos_edge_probs.shape[1])

            if label_edge_type_map is not None:
                pos_edge_probs_reordered = pos_edge_probs.clone() if isinstance(pos_edge_probs, torch.Tensor) else pos_edge_probs.copy()
                for target_idx, edge_type_name in enumerate(sorted(label_edge_type_map.keys(), key=lambda x: label_edge_type_map[x])):
                    if edge_type_name in model_edge_type_map:
                        model_idx = model_edge_type_map[edge_type_name]
                        pos_edge_probs_reordered[:, :, target_idx] = pos_edge_probs[:, :, model_idx]
                return pos_edge_probs_reordered, label_edge_type_map, fused_features
            return pos_edge_probs, model_edge_type_map, fused_features
