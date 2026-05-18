"""Two-stage training: multiple feature pretrain, then CCC precision (hierarchical)."""

from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import dgl
import numpy as np
import pickle
import torch
from anndata import AnnData
from torch.optim import Adam
from tqdm import tqdm

from model import CellSTIC
from model.graph import DGLGraphUtils, EdgeTypeMasker, EdgeTypeMapper, NegativeSampler
from utils.tools.seed_utils import active_base_seed
from utils.train import ExperimentConfig, GraphIOUtils, LossUtils
from utils.train.model_utils import ModelUtils
from utils.train.edge_filter_utils import EdgeFilterUtils


class CellSTICTrainer:
    """Train CellSTIC: feature pretrain (recon + clustering), then CCC precision (hierarchical)."""

    def __init__(
        self,
        model: CellSTIC,
        config: ExperimentConfig,
        model_path: Optional[Union[str, Path]] = None,
        cell_chat_db: Optional[Dict] = None,
        ligand_receptor_map: Optional[Dict] = None,
        lr_pair_type_constraints: Optional[Dict[str, List[Tuple[int, int]]]] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.feat_integrator = model.feat_integrator
        self.graph_generator = model.graph_generator
        self.precision_model = model.ccc_predictor
        self.tree_builder = model.tree_builder
        self.ligand_receptor_map = ligand_receptor_map
        # Optional (scmultisim only): per-ligand–receptor cell-type pair constraints
        self.lr_pair_type_constraints = lr_pair_type_constraints
        self.cell_chat_db = cell_chat_db
        self.config = config
        # Model save/load path: use explicit model_path, default to current directory if None
        self.model_path = str(Path(model_path)) if model_path is not None else "."
        self.train_config = getattr(config, "train", None)
        self.feat_train_cfg = getattr(self.train_config, "feat", None)
        self.ccc_train_cfg = getattr(self.train_config, "ccc", None)

    def train(
        self,
        modality_datas: List[AnnData],
        is_train_ccc: bool = True,
        is_train_feature: bool = True,
    ) -> None:
        """Extract data, build feature graphs, optionally run feature pretrain and CCC precision train, save model."""
        m0 = modality_datas[0]
        modality1_features = torch.tensor(m0.obsm["feat"], dtype=torch.float32).to(self.device)
        gene_expression = m0.X
        genes = m0.var_names.tolist()
        spatial_coords = torch.tensor(m0.obsm["spatial"], dtype=torch.float32).to(self.device)
        spatial_distances = m0.obsp.get("spatial_distances")
        if spatial_distances is not None:
            sd_arr = spatial_distances.toarray() if hasattr(spatial_distances, "toarray") else np.asarray(spatial_distances)
            if sd_arr.ndim < 2 or sd_arr.shape[0] < 2:
                spatial_distances = None
        modality_features = [
            torch.tensor(d.obsm["feat"], dtype=torch.float32).to(self.device) for d in modality_datas
        ]

        modality_g_dgls = self.graph_generator.build_feature_dgl_graph(modality_features, spatial_coords)
        save_dir = self.model_path
        GraphIOUtils.save_feature_graphs(modality_g_dgls, save_dir)

        if is_train_feature and self.feat_train_cfg:
            self._feature_train(modality_features)

        # Optional (scmultisim): cell types for CCC graph filtering
        cell_types = None
        if "cell_type" in m0.obs and self.lr_pair_type_constraints is not None:
            try:
                cell_types = m0.obs["cell_type"].to_numpy()
            except Exception:
                cell_types = None

        if is_train_ccc and self.ccc_train_cfg:
            self._ccc_precision_train(
                gene_expression, genes, spatial_coords, spatial_distances,
                modality1_features, modality_features,
                cell_types=cell_types,
            )

        ModelUtils.save_model(self.model, save_dir)

    def _ccc_precision_train(
        self,
        gene_expression,
        genes,
        spatial_coords: torch.Tensor,
        spatial_distances,
        modality1_features: torch.Tensor,
        modality_features: List[torch.Tensor],
        cell_types: Optional[np.ndarray] = None,
    ) -> None:
        """Build CCC graph, load/build hierarchy, init heads, train per level."""
        cfg = self.ccc_train_cfg

        base_graph_adj, edge_type_map, ligand_strength_adj, receptor_strength_adj, knn_per_modality = self.graph_generator.build_ccc_graph(
            modality_features=modality_features,
            spatial_distances=spatial_distances,
            gene_expression=gene_expression,
            genes=genes,
            ligand_receptor_map=self.ligand_receptor_map,
            cell_types=cell_types,
            lr_pair_type_constraints=self.lr_pair_type_constraints,
        )
        # Use same distance threshold as positive edges for negative sampling (hard negatives within spatial range)
        spatial_distances_np = spatial_distances.toarray() if hasattr(spatial_distances, "toarray") else np.asarray(spatial_distances)
        neg_sample_max_distance = EdgeFilterUtils.compute_spot_distance_threshold(spatial_distances_np, n_spots=4)

        modality_g_dgls = GraphIOUtils.load_feature_graphs(self.model_path, self.device)

        save_dir = Path(self.model_path)
        hierarchy_dict = None
        if (save_dir / "hierarchy_dict.pkl").exists():
            try:
                with open(save_dir / "hierarchy_dict.pkl", "rb") as f:
                    hierarchy_dict = pickle.load(f)
            except Exception:
                pass
        if hierarchy_dict is None:
            hierarchy_dict = self.tree_builder.forward(
                edge_type_map, self.tree_builder.hierarchy_method, self.cell_chat_db
            )

        hierarchy_levels = sorted(k for k in hierarchy_dict if k.startswith("level_"))
        self.precision_model.init_head_layers(hierarchy_dict)
        GraphIOUtils.save_graph_data(
            base_graph_adj, edge_type_map, hierarchy_dict, str(save_dir),
            ligand_strength_adj=ligand_strength_adj,
            receptor_strength_adj=receptor_strength_adj,
            knn_per_modality=knn_per_modality,
        )

        optimizer = Adam(
            list(self.precision_model.parameters()) + list(self.feat_integrator.parameters()),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        self.precision_model.train()
        self.feat_integrator.train()
        for level_idx, level_key in enumerate(hierarchy_levels):
            level_data = hierarchy_dict[level_key]
            mapped_base_graph, mapped_ligand_strength, mapped_receptor_strength, knn_strength_adj = (
                EdgeTypeMapper.map_level_graph_and_strengths(
                    base_graph_adj, ligand_strength_adj, receptor_strength_adj,
                    np.asarray(knn_per_modality), level_data, edge_type_map,
                )
            )
            pbar = tqdm(range(cfg.epochs), desc=f"Level {level_idx + 1}, type num {len(level_data)}", mininterval=0.1)
            for epoch_idx, _ in enumerate(pbar):
                _, masked_loss, edge_type_loss = self._train_ccc_epoch(
                    optimizer, mapped_base_graph, modality1_features, modality_g_dgls,
                    spatial_coords, level_idx, epoch_idx,
                    spatial_distances=spatial_distances_np,
                    neg_sample_max_distance=neg_sample_max_distance,
                    mapped_ligand_strength=mapped_ligand_strength,
                    mapped_receptor_strength=mapped_receptor_strength,
                    knn_strength_adj=knn_strength_adj,
                )
                pbar.set_postfix(path=masked_loss.item(), edge_type=edge_type_loss.item(), refresh=True)

    def _feature_train(self, modality_features: List[torch.Tensor]) -> None:
        """Pretrain feature model (recon + clustering loss)."""
        cfg = self.feat_train_cfg
        optimizer = Adam(self.feat_integrator.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        modality_g_dgls = GraphIOUtils.load_feature_graphs(self.model_path, self.device)
        self.feat_integrator.train()
        self.graph_generator.train()
        pbar = tqdm(range(cfg.epochs), desc="Feature", mininterval=0.1)
        for _ in pbar:
            _, recon_loss, cluster_loss = self._train_feature_epoch(
                optimizer, modality_g_dgls, modality_features
            )
            pbar.set_postfix(recon=recon_loss.item(), cluster=cluster_loss.item(), refresh=True)

    def _train_feature_epoch(
        self,
        optimizer: Adam,
        modality_g_dgls: List[dgl.DGLGraph],
        modality_features: List[torch.Tensor],
    ) -> tuple:
        """One epoch: forward, recon loss + clustering loss, backward."""
        optimizer.zero_grad()
        g_copy = [g.clone() for g in modality_g_dgls]
        fused, decoded = self.feat_integrator(g_copy)
        recon_loss = sum(
            LossUtils.reconstruction_loss(decoded[i], modality_features[i])
            for i in range(len(modality_features))
        )
        cluster_loss = LossUtils.unsupervised_clustering_loss(
            fused,
            n_clusters=getattr(self.feat_train_cfg, "n_clusters", None),
            entropy_weight=getattr(self.feat_train_cfg, "entropy_weight", 0.2),
        )
        total = recon_loss * getattr(self.feat_train_cfg, "weight_modality", 1.0) + cluster_loss
        total.backward()
        optimizer.step()
        return total, recon_loss, cluster_loss

    def _train_ccc_epoch(
        self,
        optimizer: Adam,
        mapped_base_graph: np.ndarray,
        modality1_features: torch.Tensor,
        modality_g_dgls: List[dgl.DGLGraph],
        spatial_coords: torch.Tensor,
        level_idx: int,
        epoch: int,
        spatial_distances: Optional[np.ndarray] = None,
        neg_sample_max_distance: Optional[float] = None,
        mapped_ligand_strength: Optional[np.ndarray] = None,
        mapped_receptor_strength: Optional[np.ndarray] = None,
        knn_strength_adj: Optional[np.ndarray] = None,
    ) -> tuple:
        """One CCC epoch: mask edges, build DGL from masked graph, sample negs, precision forward, GAEs + edge_type loss."""
        optimizer.zero_grad()
        g_copy = [g.clone() for g in modality_g_dgls]
        fused_features, _ = self.feat_integrator(g_copy)

        ccc_cfg = getattr(self.config.train, "ccc", None) or getattr(self.config.train, "CCC_precision", None)
        mask_seed = active_base_seed() + epoch

        # Balanced edge-type masking: sample a fraction of edges to predict (pos_mask_indicator marks them)
        masked_graph, pos_mask_indicator = EdgeTypeMasker.balanced_edge_type_masking(
            mapped_base_graph, ccc_cfg.sampling_rate, random_seed=mask_seed
        )
        g_dgl = DGLGraphUtils.build_dgl_graph_from_virtual_types_graph(
            masked_graph,
            node_features=modality1_features,
            device=self.device,
            spatial_coords=spatial_coords,
            ligand_strength_adj=mapped_ligand_strength,
            receptor_strength_adj=mapped_receptor_strength,
            knn_strength_adj=knn_strength_adj,
        )

        # Negative samples: one neg per masked pos per type (neg_mask_indicator marks them);
        neg_positions, neg_mask_indicator = NegativeSampler.generate_negative_samples(
            mapped_base_graph,
            random_seed=mask_seed,
            mask_indicator=pos_mask_indicator,
            spatial_distances=spatial_distances,
            max_distance=neg_sample_max_distance,
            min_distance=0.0,
        )
        neg_positions_tensor = torch.tensor(neg_positions, dtype=torch.long, device=self.device)
        pos_positions_raw = np.argwhere(pos_mask_indicator == 1)[:, :2]
        pos_positions_unique = (
            np.unique(pos_positions_raw, axis=0) if len(pos_positions_raw) else np.empty((0, 2), dtype=np.int64)
        )
        pos_positions_tensor = torch.tensor(pos_positions_unique.tolist(), dtype=torch.long, device=self.device)

        pos_edge_probs, neg_edge_probs, edge_type_pred, _ = self.precision_model(
            g_dgl, pos_positions_tensor, neg_positions_tensor, level_idx, fused_features
        )
        pos_indicator_tensor = torch.tensor(pos_mask_indicator, dtype=torch.float32, device=self.device)
        neg_indicator_tensor = torch.tensor(neg_mask_indicator, dtype=torch.float32, device=self.device)

        edge_type_loss = LossUtils.edge_type_reconstruction_loss(edge_type_pred, g_dgl.edata["edge_type"])
        masked_path_loss = LossUtils.gaes_loss(
            pos_indicator_tensor,
            pos_edge_probs,
            neg_edge_probs,
            neg_indicator_tensor,
            pos_positions_tensor,
            neg_positions_tensor,
        )
        total_loss = masked_path_loss + ccc_cfg.edge_type_loss_weight * edge_type_loss
        if total_loss.requires_grad:
            total_loss.backward()
            optimizer.step()
        return total_loss, masked_path_loss, edge_type_loss
