"""CellSTIC end-to-end train, infer, and ``out.h5ad`` export pipeline."""

from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from anndata import AnnData
from scipy import sparse
from torch.optim import Adam
from tqdm import tqdm

from model import CellSTIC
from model.graph import (
    DGLChannelEdgeIndex,
    DGLGraphUtils,
    DGLNodeBatchIndex,
    EdgeTypeMapper,
    EdgeTypeMasker,
    NegativeSampler,
    SparseTypedGraph,
)
from model.train import (
    CCC_CHECKPOINT_NAME,
    CCCPostprocessor,
    FEATURE_CHECKPOINT_NAME,
    ClusteringUtils,
    EdgeFilterUtils,
    CellSTICConfig,
    LossUtils,
    save_ccc_checkpoint,
    save_feature_checkpoint,
)
from utils.data.processor_utils import SpatialPreprocessorUtils


PathLike = Union[str, Path]
CELLSTIC_SCHEMA_VERSION = 5
_OBSP_PREFIX = "cellstic_L"

_CHROMATIN_TYPES = {"atac", "h3k27ac", "h3k27me3", "chromatin", "peaks"}
_PROTEIN_TYPES = {"adt", "protein", "proteomics"}
_STANDARDIZED_PROTEIN_TYPES = {"adt_standardized", "protein_standardized"}


def _pad_features(features: np.ndarray, n_components: int) -> np.ndarray:
    """Keep feature dimensions compatible with the configured encoders."""
    features = np.asarray(features, dtype=np.float32)
    if features.shape[1] < n_components:
        features = np.pad(features, ((0, 0), (0, n_components - features.shape[1])))
    return features[:, :n_components]


def _input_dimensions(config: CellSTICConfig, n_modalities: int) -> List[int]:
    encoder_dims = config.model.feat.encoder_dims
    if len(encoder_dims) != n_modalities:
        raise ValueError(
            "config.model.feat.encoder_dims must contain one dimension list per modality"
        )
    return [int(dimensions[0]) for dimensions in encoder_dims]


def _validate_runtime_config(
    config: CellSTICConfig,
    *,
    n_observations: int,
    n_modalities: int,
    n_clusters: Optional[int] = None,
) -> None:
    """Validate constraints that depend on the assembled input data."""
    if n_observations < 2:
        raise ValueError(
            f"CellSTIC requires at least two shared observations, got {n_observations}"
        )
    if len(config.model.feat.encoder_dims) != n_modalities:
        raise ValueError(
            "config.model.feat.encoder_dims must contain one dimension list per "
            f"modality, got {len(config.model.feat.encoder_dims)} for {n_modalities}"
        )
    for parameter in ("knn_top_k", "cluster_top_k"):
        value = getattr(config.model.graph, parameter)
        if value >= n_observations:
            raise ValueError(
                f"config.model.graph.{parameter} must be smaller than the "
                f"number of shared observations ({n_observations}), got {value}"
            )
    effective_clusters = (
        config.train.feat.n_clusters if n_clusters is None else n_clusters
    )
    if isinstance(effective_clusters, bool) or not isinstance(
        effective_clusters, (int, np.integer)
    ):
        raise ValueError(f"n_clusters must be an integer, got {effective_clusters!r}")
    if not 2 <= int(effective_clusters) <= n_observations:
        raise ValueError(
            f"n_clusters must be between 2 and n_observations ({n_observations}), "
            f"got {effective_clusters}"
        )


def _infer_modality_type(adata: AnnData, index: int) -> str:
    explicit = adata.uns.get("modality") or adata.uns.get("feature_type")
    if explicit:
        value = str(explicit).lower()
        return "chromatin" if "peak" in value else value
    return "rna" if index == 0 else "generic"


def _prepare_rna(adata: AnnData, n_components: int, random_seed: int) -> None:
    import scanpy as sc

    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.astype(str).str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    n_top_genes = min(3000, adata.n_vars)
    try:
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)
    except (ImportError, ValueError, np.linalg.LinAlgError):
        try:
            sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=n_top_genes)
        except (ValueError, IndexError, np.linalg.LinAlgError):
            matrix = adata.layers["counts"]
            if sparse.issparse(matrix):
                values = matrix.astype(np.float64, copy=False)
                means = np.asarray(values.mean(axis=0)).reshape(-1)
                squared_means = np.asarray(values.power(2).mean(axis=0)).reshape(-1)
                variances = np.maximum(squared_means - means**2, 0.0)
            else:
                variances = np.asarray(matrix).var(axis=0)
            top_indices = np.argsort(variances)[-n_top_genes:]
            mask = np.zeros(adata.n_vars, dtype=bool)
            mask[top_indices] = True
            adata.var["highly_variable"] = mask
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    feature_mask = np.asarray(adata.var.get("highly_variable"), dtype=bool)
    feature_data = adata[:, feature_mask] if feature_mask.any() else adata
    adata.obsm["feat"] = SpatialPreprocessorUtils.pca(
        feature_data, n_comps=n_components, random_state=random_seed
    )


def _prepare_modality(
    adata: AnnData,
    modality_type: str,
    n_components: int,
    random_seed: int,
) -> None:
    kind = modality_type.lower()
    if "rna" in kind or "transcript" in kind:
        _prepare_rna(adata, n_components, random_seed)
    elif kind in _STANDARDIZED_PROTEIN_TYPES:
        # Explicitly pre-standardized protein measurements may be negative;
        # count-based CLR is not defined for these inputs. Preserve their scale
        # and reject invalid values before PCA's generic cleaning fallback.
        values = adata.X.data if sparse.issparse(adata.X) else np.asarray(adata.X)
        if not np.all(np.isfinite(values)):
            raise ValueError(
                "Standardized protein input must contain only finite values; "
                "found NaN or inf"
            )
        adata.obsm["feat"] = SpatialPreprocessorUtils.pca(
            adata, n_comps=n_components, random_state=random_seed
        )
    elif kind in _PROTEIN_TYPES:
        SpatialPreprocessorUtils.clr_normalize_each_cell(adata)
        adata.obsm["feat"] = SpatialPreprocessorUtils.pca(
            adata, n_comps=n_components, random_state=random_seed
        )
    elif kind in _CHROMATIN_TYPES:
        # ``lsi`` stores the non-trivial components in ``X_lsi``.
        if min(adata.shape) < 2:
            adata.obsm["feat"] = SpatialPreprocessorUtils.pca(
                adata, n_comps=n_components, random_state=random_seed
            )
        else:
            requested = min(n_components + 1, min(adata.shape))
            SpatialPreprocessorUtils.lsi(
                adata, n_components=requested, random_state=random_seed
            )
            adata.obsm["feat"] = _pad_features(adata.obsm["X_lsi"], n_components)
    else:
        adata.obsm["feat"] = SpatialPreprocessorUtils.pca(
            adata, n_comps=n_components, random_state=random_seed
        )
    adata.obsm["feat"] = _pad_features(adata.obsm["feat"], n_components)


def prepare_modalities(
    modality_datas: Sequence[AnnData],
    config: CellSTICConfig,
    *,
    modality_types: Optional[Sequence[str]] = None,
    n_clusters: Optional[int] = None,
) -> List[AnnData]:
    """Align and preprocess assembled modalities for a CellSTIC run in memory."""
    if not isinstance(config, CellSTICConfig):
        raise TypeError(f"config must be CellSTICConfig, got {type(config).__name__}")
    if not modality_datas:
        raise ValueError("modality_datas must contain at least one AnnData object")
    if modality_types is not None and len(modality_types) != len(modality_datas):
        raise ValueError("modality_types must match modality_datas length")

    common = modality_datas[0].obs_names
    for adata in modality_datas[1:]:
        common = common.intersection(adata.obs_names, sort=False)
    if len(common) == 0:
        raise ValueError("Modalities do not share any observation identifiers")
    _validate_runtime_config(
        config,
        n_observations=len(common),
        n_modalities=len(modality_datas),
        n_clusters=n_clusters,
    )
    prepared = [adata[common].copy() for adata in modality_datas]
    if "spatial" not in prepared[0].obsm:
        raise KeyError("The first modality must provide adata.obsm['spatial']")
    spatial = np.asarray(prepared[0].obsm["spatial"], dtype=np.float32)
    if spatial.ndim != 2 or spatial.shape != (len(common), 2):
        raise ValueError(
            "adata.obsm['spatial'] must have shape (n_obs, 2), got " f"{spatial.shape}"
        )
    if not np.all(np.isfinite(spatial)):
        raise ValueError("adata.obsm['spatial'] contains non-finite coordinates")

    dimensions = _input_dimensions(config, len(prepared))
    kinds = (
        list(modality_types)
        if modality_types is not None
        else [
            _infer_modality_type(adata, index) for index, adata in enumerate(prepared)
        ]
    )
    for index, (adata, kind, n_components) in enumerate(
        zip(prepared, kinds, dimensions)
    ):
        adata.var_names_make_unique()
        adata.obsm["spatial"] = spatial.copy()
        _prepare_modality(adata, kind, n_components, config.seed)
        features = np.asarray(adata.obsm["feat"])
        if features.shape != (len(common), n_components):
            raise RuntimeError(
                f"Prepared modality {index} has feature shape {features.shape}; "
                f"expected ({len(common)}, {n_components})"
            )
        if not np.all(np.isfinite(features)):
            raise ValueError(f"Prepared modality {index} contains non-finite features")
        adata.uns["cellstic_preprocessing"] = {
            "modality": str(kind),
            "feature_dimension": int(n_components),
            "pipeline_managed": True,
        }
        if index > 0:
            for column in prepared[0].obs.columns:
                if column not in adata.obs:
                    adata.obs[column] = prepared[0].obs[column].copy()
    return prepared


def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


def _set_global_seed(config: CellSTICConfig, device: torch.device) -> None:
    seed = int(config.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    deterministic = config.deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(deterministic, warn_only=True)


def _hierarchy_levels(hierarchy_dict: Mapping[str, Any]) -> List[str]:
    return sorted(
        (key for key in hierarchy_dict if key.startswith("level_")),
        key=lambda key: int(key.split("_")[1]),
    )


def _sanitize_key(value: str) -> str:
    return (
        str(value).replace(":", "_").replace("/", "_").replace(" ", "_").strip()
        or "channel"
    )


def _obsp_key(level_num: int, channel_name: str) -> str:
    return f"{_OBSP_PREFIX}{level_num}_lr_{_sanitize_key(channel_name)}"


def _postprocess_obsp_key(level_num: int, channel_name: str, statistic: str) -> str:
    return f"{_obsp_key(level_num, channel_name)}_{statistic}"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def _config_dict(config: Any) -> Dict[str, Any]:
    def convert(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Mapping):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        if hasattr(value, "__dict__"):
            return {
                key: convert(item)
                for key, item in vars(value).items()
                if not key.startswith("_")
            }
        return str(value)

    return convert(config)


class CellSTICPipeline:
    """Run the complete CellSTIC lifecycle once, entirely in memory.

    Assembled AnnData inputs are aligned and preprocessed, the requested
    trainable stages are optimized, and predictions are attached to one
    ``out.h5ad``. ``feature.pt`` is always written; ``ccc.pt`` is written when
    ``run_ccc=True``. Spatial domain prediction can be skipped. There is no
    preprocessed-h5ad cache and no checkpoint-loading path.
    """

    def __init__(
        self,
        modality_datas: Sequence[AnnData],
        config: CellSTICConfig,
        *,
        output_path: PathLike,
        ligand_receptor_map: Optional[Dict] = None,
        cell_chat_db: Optional[Dict] = None,
        modality_types: Optional[Sequence[str]] = None,
        auto_n_clusters: Optional[int] = None,
        run_ccc: bool = True,
        skip_spatial_domain_prediction: bool = False,
        postprocess_before_aggregation: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.modality_datas = list(modality_datas)
        if not self.modality_datas:
            raise ValueError("modality_datas must contain at least one AnnData object")
        if not isinstance(config, CellSTICConfig):
            raise TypeError(
                f"config must be CellSTICConfig, got {type(config).__name__}"
            )

        self.config = config
        self.output_path = Path(output_path)
        primary_uns = self.modality_datas[0].uns
        if ligand_receptor_map is None:
            ligand_receptor_map = primary_uns.get("ligand_receptor_map")
        if not isinstance(run_ccc, bool):
            raise TypeError("run_ccc must be a bool")
        if run_ccc and not ligand_receptor_map:
            raise ValueError("run_ccc=True requires a non-empty ligand_receptor_map")
        if ligand_receptor_map is not None and not isinstance(
            ligand_receptor_map, Mapping
        ):
            raise TypeError("ligand_receptor_map must be a mapping")
        self.ligand_receptor_map = ligand_receptor_map
        self.cell_chat_db = cell_chat_db
        self.modality_types = (
            list(modality_types) if modality_types is not None else None
        )
        self.auto_n_clusters = auto_n_clusters
        self.run_ccc = run_ccc
        if not isinstance(skip_spatial_domain_prediction, bool):
            raise TypeError("skip_spatial_domain_prediction must be a bool")
        self.skip_spatial_domain_prediction = skip_spatial_domain_prediction
        if not isinstance(postprocess_before_aggregation, bool):
            raise TypeError("postprocess_before_aggregation must be a bool")
        self.postprocess_before_aggregation = postprocess_before_aggregation
        self.device = _resolve_device(device)
        if config.deterministic and self.device.type == "cuda":
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        self._feature_cluster_generator = torch.Generator(device=str(self.device))
        self._feature_cluster_generator.manual_seed(config.seed + 1)

        self.model: Optional[CellSTIC] = None
        self.feat_integrator = None
        self.graph_generator = None
        self.precision_model = None
        self.tree_builder = None
        self.train_config = config.train
        self.feat_train_cfg = self.train_config.feat
        self.ccc_train_cfg = self.train_config.ccc

        self._feature_graphs: Optional[List[dgl.DGLGraph]] = None
        self._base_graph: Optional[SparseTypedGraph] = None
        self._edge_type_map: Dict[str, int] = {}
        self._hierarchy_dict: Dict[str, Any] = {}
        self._ligand_strength_adj: Optional[SparseTypedGraph] = None
        self._receptor_strength_adj: Optional[SparseTypedGraph] = None
        self._knn_per_modality: Optional[Tuple[sparse.csr_matrix, ...]] = None
        self._prepared = False
        self._trained = False
        self._ccc_trained = False

        self.output_path.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.output_path / "checkpoint"
        self.result_path = self.output_path / "out"
        self.checkpoint_path.mkdir(exist_ok=True)
        self.result_path.mkdir(exist_ok=True)
        _set_global_seed(config, self.device)

    def _prepare(self) -> List[AnnData]:
        """Preprocess assembled modalities in memory without cache reuse."""
        if not self._prepared:
            self.modality_datas = prepare_modalities(
                self.modality_datas,
                self.config,
                modality_types=self.modality_types,
                n_clusters=self.auto_n_clusters,
            )
            self._prepared = True
        return self.modality_datas

    def _bind_model(self) -> CellSTIC:
        self.model = CellSTIC(
            self.config.model, self.device, random_seed=self.config.seed
        ).to(self.device)
        self.feat_integrator = self.model.feat_integrator
        self.graph_generator = self.model.graph_generator
        self.precision_model = self.model.ccc_predictor
        self.tree_builder = self.model.tree_builder
        return self.model

    def _train(self) -> CellSTIC:
        """Train feature integration and, when configured, CCC prediction."""
        self._prepare()
        model = self._bind_model()
        primary = self.modality_datas[0]
        modality1_features = torch.as_tensor(
            primary.obsm["feat"], dtype=torch.float32, device=self.device
        )
        spatial_coords = torch.as_tensor(
            primary.obsm["spatial"], dtype=torch.float32, device=self.device
        )
        modality_features = [
            torch.as_tensor(data.obsm["feat"], dtype=torch.float32, device=self.device)
            for data in self.modality_datas
        ]
        self._feature_graphs = self.graph_generator.build_feature_dgl_graph(
            modality_features, spatial_coords
        )

        self._train_features()

        if self.run_ccc:
            self._train_ccc(
                gene_expression=primary.X,
                genes=primary.var_names.tolist(),
                spatial_coords=spatial_coords,
                modality1_features=modality1_features,
                modality_features=modality_features,
            )
            self._ccc_trained = True

        config_snapshot = _config_dict(self.config)
        config_snapshot["postprocess_before_aggregation"] = (
            self.postprocess_before_aggregation
        )
        config_snapshot["skip_spatial_domain_prediction"] = (
            self.skip_spatial_domain_prediction
        )
        save_feature_checkpoint(
            self.feat_integrator,
            self.checkpoint_path,
            config=config_snapshot,
        )
        if self.run_ccc:
            save_ccc_checkpoint(
                self.precision_model,
                self.checkpoint_path,
                config=config_snapshot,
                trained=self._ccc_trained,
                edge_type_map=self._edge_type_map,
                hierarchy_dict=self._hierarchy_dict,
            )
        self._trained = True
        return model

    def _train_features(self) -> None:
        if self._feature_graphs is None:
            raise RuntimeError("Feature graphs have not been built")
        cfg = self.feat_train_cfg
        n_cells = self._feature_graphs[0].num_nodes()
        if cfg.n_clusters > n_cells:
            raise ValueError(
                f"train.feat.n_clusters ({cfg.n_clusters}) must not exceed "
                f"the number of cells ({n_cells})"
            )
        optimizer = Adam(
            self.feat_integrator.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        self.feat_integrator.train()
        progress = tqdm(range(cfg.epochs), desc="Feature", mininterval=0.1)
        for _ in progress:
            _, reconstruction_loss, cluster_loss = self._train_feature_epoch(
                optimizer,
                self._feature_graphs,
            )
            progress.set_postfix(
                recon=reconstruction_loss.item(),
                cluster=cluster_loss.item(),
                refresh=True,
            )

    def _train_feature_epoch(
        self,
        optimizer: Adam,
        feature_graphs: List[dgl.DGLGraph],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Optimize one feature epoch on the complete aligned graphs.

        Reconstruction and clustering are both evaluated over all cells in a
        single forward pass.  In particular, soft K-means therefore estimates
        one dataset-wide set of centers instead of independently forcing every
        spatially local mini-batch to contain all configured clusters.
        """
        cfg = self.feat_train_cfg
        optimizer.zero_grad(set_to_none=True)
        fused, decoded = self.feat_integrator(feature_graphs)
        reconstruction_loss = sum(
            LossUtils.reconstruction_loss(
                decoded[index], feature_graphs[index].ndata["node_features"]
            )
            for index in range(len(feature_graphs))
        )
        cluster_loss = LossUtils.unsupervised_clustering_loss(
            fused,
            n_clusters=cfg.n_clusters,
            entropy_weight=cfg.entropy_weight,
            temperature=cfg.clustering_temperature,
            n_iterations=cfg.clustering_iterations,
            generator=self._feature_cluster_generator,
        )
        total_loss = (
            reconstruction_loss * cfg.reconstruction_loss_weight + cluster_loss
        )
        total_loss.backward()
        optimizer.step()
        return total_loss.detach(), reconstruction_loss.detach(), cluster_loss.detach()

    def _train_ccc(
        self,
        *,
        gene_expression: Any,
        genes: List[str],
        spatial_coords: torch.Tensor,
        modality1_features: torch.Tensor,
        modality_features: List[torch.Tensor],
    ) -> None:
        if self._feature_graphs is None:
            raise RuntimeError("Feature graphs have not been built")
        cfg = self.ccc_train_cfg
        (
            self._base_graph,
            self._edge_type_map,
            self._ligand_strength_adj,
            self._receptor_strength_adj,
            self._knn_per_modality,
        ) = self.graph_generator.build_ccc_graph(
            modality_features=modality_features,
            spatial_coords=spatial_coords,
            gene_expression=gene_expression,
            genes=genes,
            ligand_receptor_map=self.ligand_receptor_map,
        )
        if not self._edge_type_map:
            raise ValueError(
                "No ligand-receptor channels remain after graph construction"
            )

        self._hierarchy_dict = self.tree_builder.forward(
            self._edge_type_map,
            self.cell_chat_db,
        )
        hierarchy_levels = _hierarchy_levels(self._hierarchy_dict)
        self.precision_model.init_head_layers(self._hierarchy_dict)

        max_negative_distance = EdgeFilterUtils.compute_spot_distance_threshold(
            spatial_coords, n_spots=self.graph_generator.n_spots
        )
        spatial_candidates = EdgeFilterUtils.spatial_candidate_pairs(
            spatial_coords,
            max_negative_distance,
            max_neighbors=self.graph_generator.max_spatial_neighbors,
        )
        optimizer = Adam(
            [
                {"params": self.precision_model.parameters()},
                {"params": self.feat_integrator.parameters()},
            ],
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        self.precision_model.train()
        self.feat_integrator.train()
        feature_batch_index = DGLGraphUtils.build_node_batch_index(
            self._feature_graphs
        )
        feature_message_passing_hops = max(
            len(layers) for layers in self.feat_integrator.modality_gnn_encoders
        )
        total_leaf_count = len(self._edge_type_map)
        # ``HierarchyTree.to_dict`` numbers the root as ``level_1``. Train all
        # heads root-to-leaf as multiscale auxiliary supervision for the shared
        # encoder. Only the leaf head is used directly at inference; exported
        # ancestor scores are aggregated from leaf predictions.
        for level_idx, level_key in enumerate(hierarchy_levels):
            level_data = self._hierarchy_dict[level_key]
            (
                mapped_graph,
                mapped_ligand_strength,
                mapped_receptor_strength,
                knn_strength_adj,
            ) = EdgeTypeMapper.map_level_graph_and_strengths(
                self._base_graph,
                self._ligand_strength_adj,
                self._receptor_strength_adj,
                self._knn_per_modality,
                level_data,
                self._edge_type_map,
                normalization_leaf_count=total_leaf_count,
            )
            path_targets = EdgeTypeMapper.map_path_targets(
                self._base_graph,
                level_data,
                self._edge_type_map,
                normalization_leaf_count=(
                    None if level_idx == len(hierarchy_levels) - 1 else total_leaf_count
                ),
            )
            # Construct and normalize the complete level graph once.  Epoch
            # masks are DGL edge subgraphs, so retained edge features remain on
            # this stable scale.
            level_graph = DGLGraphUtils.build_dgl_graph_from_virtual_types_graph(
                mapped_graph,
                node_features=modality1_features,
                device=self.device,
                spatial_coords=spatial_coords,
                ligand_strength_adj=mapped_ligand_strength,
                receptor_strength_adj=mapped_receptor_strength,
                knn_strength_adj=knn_strength_adj,
                path_target_adj=path_targets,
            )
            channel_edge_index = EdgeTypeMasker.build_dgl_channel_index(
                level_graph, len(level_data)
            )
            progress = tqdm(
                range(cfg.epochs),
                desc=f"{level_key} (root-to-leaf), type num {len(level_data)}",
                mininterval=0.1,
            )
            for epoch in progress:
                _, path_loss, edge_type_loss = self._train_ccc_epoch(
                    optimizer=optimizer,
                    level_graph=level_graph,
                    level_sparse_graph=mapped_graph,
                    channel_edge_index=channel_edge_index,
                    level_idx=level_idx,
                    epoch=epoch,
                    negative_candidate_pairs=spatial_candidates,
                    feature_batch_index=feature_batch_index,
                    feature_message_passing_hops=feature_message_passing_hops,
                )
                progress.set_postfix(
                    path=path_loss.item(),
                    edge_type=edge_type_loss.item(),
                    refresh=True,
                )

    def _train_ccc_epoch(
        self,
        *,
        optimizer: Adam,
        level_graph: dgl.DGLGraph,
        level_sparse_graph: SparseTypedGraph,
        channel_edge_index: DGLChannelEdgeIndex,
        level_idx: int,
        epoch: int,
        negative_candidate_pairs: np.ndarray,
        feature_batch_index: DGLNodeBatchIndex,
        feature_message_passing_hops: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Optimize one level epoch with channel-balanced local graph batches."""
        cfg = self.ccc_train_cfg
        mask_seed = self.config.seed + level_idx * cfg.epochs + epoch
        (
            positive_positions,
            positive_channels,
            positive_targets,
            positive_edge_ids,
            channel_counts,
        ) = EdgeTypeMasker.sample_balanced_dgl_positive_edges(
            level_graph,
            channel_edge_index,
            cfg.sampling_rate,
            min_edges_per_channel=cfg.min_masked_edges_per_channel,
            max_edges_per_channel=cfg.max_masked_edges_per_channel,
            random_seed=mask_seed,
        )
        negative_positions, negative_channels = NegativeSampler.sample_channel_balanced(
            level_sparse_graph,
            negative_candidate_pairs,
            channel_counts,
            random_seed=mask_seed + 1,
        )
        epoch_sample_count = len(positive_positions) + len(negative_positions)
        if epoch_sample_count == 0:
            raise RuntimeError("CCC epoch produced no positive/negative samples")
        path_loss_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        edge_type_loss_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        batch_count = 0
        batches = self._iter_channel_balanced_graph_batches(
            positive_positions,
            positive_channels,
            positive_targets,
            positive_edge_ids,
            negative_positions,
            negative_channels,
            batch_size=cfg.graph_batch_size,
            random_seed=mask_seed + 2,
        )
        for (
            sample_positions,
            sample_channels,
            sample_labels,
            sample_classes,
            masked_positive_edge_ids,
        ) in batches:
            graph, local_positions, global_nodes = (
                EdgeTypeMasker.build_dgl_training_batch(
                    level_graph,
                    channel_edge_index,
                    sample_positions,
                    masked_positive_edge_ids,
                    num_message_passing_hops=len(self.precision_model.egnn_layers),
                )
            )
            weights = self._channel_balanced_binary_weights(
                sample_channels,
                sample_classes,
                channel_counts,
            )
            positions_tensor = torch.as_tensor(
                local_positions, dtype=torch.long, device=self.device
            )
            channels_tensor = torch.as_tensor(
                sample_channels, dtype=torch.long, device=self.device
            )
            labels_tensor = torch.as_tensor(
                sample_labels, dtype=torch.float32, device=self.device
            )
            weights_tensor = torch.as_tensor(
                weights, dtype=torch.float32, device=self.device
            )

            optimizer.zero_grad(set_to_none=True)
            feature_graphs, feature_targets = DGLGraphUtils.build_aligned_node_batch(
                self._feature_graphs,
                feature_batch_index,
                global_nodes.detach().cpu().numpy(),
                num_message_passing_hops=feature_message_passing_hops,
            )
            fused_features = self.feat_integrator.encode(feature_graphs).index_select(
                0, feature_targets
            )
            encoded_nodes, encoded_edges = self.precision_model.encode_graph(
                graph, fused_features
            )
            path_logits = self.precision_model.selected_edge_logits(
                positions_tensor,
                channels_tensor,
                encoded_nodes,
                level_idx,
            )
            path_loss = LossUtils.sampled_gaes_loss(
                path_logits, labels_tensor, weights_tensor
            )

            edge_type_loss = torch.zeros((), dtype=torch.float32, device=self.device)
            if level_idx > 0 and cfg.edge_type_loss_weight > 0:
                if encoded_edges is None:
                    raise RuntimeError(
                        "CCC training requires collected edge representations"
                    )
                edge_ids, edge_targets, edge_weights = self._balanced_edge_type_samples(
                    graph,
                    n_channels=len(channel_counts),
                    max_per_channel=cfg.max_masked_edges_per_channel,
                    max_total=cfg.graph_batch_size,
                    target_node_ids=np.unique(local_positions.reshape(-1)),
                    random_seed=mask_seed + 3 + batch_count,
                )
                if len(edge_ids):
                    edge_ids_tensor = torch.as_tensor(
                        edge_ids, dtype=torch.long, device=self.device
                    )
                    edge_targets_tensor = torch.as_tensor(
                        edge_targets, dtype=torch.long, device=self.device
                    )
                    edge_weights_tensor = torch.as_tensor(
                        edge_weights, dtype=torch.float32, device=self.device
                    )
                    edge_logits = self.precision_model.edge_type_logits(
                        encoded_edges[edge_ids_tensor], level_idx
                    )
                    edge_type_loss = torch.sum(
                        F.cross_entropy(
                            edge_logits,
                            edge_targets_tensor,
                            reduction="none",
                        )
                        * edge_weights_tensor
                    )

            # The path weights already sum to one over the complete epoch.
            # Apply the same epoch-level normalization to the auxiliary edge
            # classification term so its relative weight does not grow with
            # the number of graph batches.
            batch_fraction = len(sample_positions) / float(epoch_sample_count)
            scaled_edge_type_loss = edge_type_loss * batch_fraction
            total_batch_loss = (
                path_loss + cfg.edge_type_loss_weight * scaled_edge_type_loss
            )
            total_batch_loss.backward()
            optimizer.step()
            path_loss_sum += path_loss.detach()
            edge_type_loss_sum += scaled_edge_type_loss.detach()
            batch_count += 1

        if batch_count == 0:
            raise RuntimeError("CCC epoch produced no positive/negative graph batches")
        # Path sample weights are normalized over the complete epoch, so its
        # reported epoch loss is the sum of the batch contributions.
        path_loss = path_loss_sum
        edge_type_loss = edge_type_loss_sum
        total_loss = path_loss + cfg.edge_type_loss_weight * edge_type_loss
        return total_loss, path_loss, edge_type_loss

    @staticmethod
    def _iter_channel_balanced_graph_batches(
        positive_positions: np.ndarray,
        positive_channels: np.ndarray,
        positive_targets: np.ndarray,
        positive_edge_ids: np.ndarray,
        negative_positions: np.ndarray,
        negative_channels: np.ndarray,
        *,
        batch_size: int,
        random_seed: int,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Yield local-graph targets as matched positive/negative channel pairs."""
        positive_positions = np.asarray(positive_positions, dtype=np.int64).reshape(
            -1, 2
        )
        negative_positions = np.asarray(negative_positions, dtype=np.int64).reshape(
            -1, 2
        )
        positive_channels = np.asarray(positive_channels, dtype=np.int64).reshape(-1)
        negative_channels = np.asarray(negative_channels, dtype=np.int64).reshape(-1)
        positive_targets = np.asarray(positive_targets, dtype=np.float32).reshape(-1)
        positive_edge_ids = np.asarray(positive_edge_ids, dtype=np.int64).reshape(-1)
        if not (
            len(positive_positions)
            == len(positive_channels)
            == len(positive_targets)
            == len(positive_edge_ids)
        ):
            raise ValueError("Positive CCC sample arrays must be aligned")
        if len(negative_positions) != len(negative_channels):
            raise ValueError("Negative CCC sample arrays must be aligned")
        if batch_size < 2:
            raise ValueError("graph batch_size must be >= 2")

        rng = np.random.default_rng(random_seed)
        positive_order = np.argsort(positive_channels, kind="stable")
        negative_order = np.argsort(negative_channels, kind="stable")
        positive_values, positive_starts, positive_counts = np.unique(
            positive_channels[positive_order],
            return_index=True,
            return_counts=True,
        )
        negative_values, negative_starts, negative_counts = np.unique(
            negative_channels[negative_order],
            return_index=True,
            return_counts=True,
        )
        if not np.array_equal(positive_values, negative_values) or not np.array_equal(
            positive_counts, negative_counts
        ):
            raise ValueError(
                "Positive and negative samples must have identical per-channel counts"
            )
        channels = positive_values
        positive_by_channel = {}
        negative_by_channel = {}
        counts = {}
        for channel, pos_start, neg_start, count in zip(
            channels, positive_starts, negative_starts, positive_counts
        ):
            pos_ids = positive_order[pos_start : pos_start + count]
            neg_ids = negative_order[neg_start : neg_start + count]
            if len(pos_ids) == 0:
                raise ValueError(
                    f"Channel {channel} must contain a non-zero matched pair"
                )
            positive_by_channel[int(channel)] = rng.permutation(pos_ids)
            negative_by_channel[int(channel)] = rng.permutation(neg_ids)
            counts[int(channel)] = int(count)

        positive_order_parts = []
        negative_order_parts = []
        for offset in range(max(counts.values(), default=0)):
            active = np.asarray(
                [channel for channel in channels if counts[int(channel)] > offset],
                dtype=np.int64,
            )
            rng.shuffle(active)
            positive_order_parts.append(
                np.asarray(
                    [positive_by_channel[int(channel)][offset] for channel in active],
                    dtype=np.int64,
                )
            )
            negative_order_parts.append(
                np.asarray(
                    [negative_by_channel[int(channel)][offset] for channel in active],
                    dtype=np.int64,
                )
            )
        if not positive_order_parts:
            return
        positive_order = np.concatenate(positive_order_parts)
        negative_order = np.concatenate(negative_order_parts)
        pair_batch_size = max(1, batch_size // 2)
        for start in range(0, len(positive_order), pair_batch_size):
            pos_ids = positive_order[start : start + pair_batch_size]
            neg_ids = negative_order[start : start + pair_batch_size]
            positions = np.vstack(
                (positive_positions[pos_ids], negative_positions[neg_ids])
            )
            batch_channels = np.concatenate(
                (positive_channels[pos_ids], negative_channels[neg_ids])
            )
            labels = np.concatenate(
                (
                    positive_targets[pos_ids],
                    np.zeros(len(neg_ids), dtype=np.float32),
                )
            )
            classes = np.concatenate(
                (
                    np.ones(len(pos_ids), dtype=np.float32),
                    np.zeros(len(neg_ids), dtype=np.float32),
                )
            )
            permutation = rng.permutation(len(positions))
            yield (
                positions[permutation],
                batch_channels[permutation],
                labels[permutation],
                classes[permutation],
                positive_edge_ids[pos_ids],
            )

    @staticmethod
    def _channel_balanced_binary_weights(
        channels: np.ndarray,
        class_labels: np.ndarray,
        channel_totals: np.ndarray,
    ) -> np.ndarray:
        """Give each active channel/class equal total weight over one epoch."""
        channels = np.asarray(channels, dtype=np.int64)
        class_labels = np.asarray(class_labels, dtype=np.float32)
        channel_totals = np.asarray(channel_totals, dtype=np.int64).reshape(-1)
        if channels.shape != class_labels.shape:
            raise ValueError("channels and class_labels must have identical shapes")
        if len(channels) == 0:
            raise ValueError("At least one positive/negative path sample is required")
        if np.any(channel_totals < 0):
            raise ValueError("channel_totals must be non-negative")
        epoch_active = np.flatnonzero(channel_totals > 0)
        if len(epoch_active) == 0:
            raise ValueError("channel_totals contains no active channel")
        batch_active = np.unique(channels)
        if np.any(batch_active < 0) or np.any(batch_active >= len(channel_totals)):
            raise ValueError("channels contains an out-of-range channel")
        weights = np.zeros(len(channels), dtype=np.float32)
        for channel in batch_active:
            total = int(channel_totals[channel])
            if total < 1:
                raise ValueError(f"Batch contains inactive channel {channel}")
            for class_label in (0.0, 1.0):
                selected = (channels == channel) & (class_labels == class_label)
                count = int(selected.sum())
                if count == 0:
                    raise ValueError(
                        f"Channel {channel} lacks class " f"{int(class_label)} samples"
                    )
                weights[selected] = 1.0 / (len(epoch_active) * 2.0 * total)
        return weights

    @staticmethod
    def _balanced_edge_type_samples(
        graph: dgl.DGLGraph,
        *,
        n_channels: int,
        max_per_channel: int,
        max_total: int,
        target_node_ids: np.ndarray,
        random_seed: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return bounded, macro-weighted edge-type classification samples."""
        edge_types = graph.edata["edge_type"]
        if edge_types.ndim == 2:
            edge_types = edge_types.argmax(dim=1)
        types = edge_types.detach().cpu().numpy().astype(np.int64, copy=False)
        non_self = ~graph.edata["is_self"].detach().cpu().numpy().astype(bool)
        _, targets = graph.edges(order="eid")
        target_ids = targets.detach().cpu().numpy().astype(np.int64, copy=False)
        target_node_ids = np.asarray(target_node_ids, dtype=np.int64).reshape(-1)
        eligible_targets = np.isin(target_ids, target_node_ids)
        rng = np.random.default_rng(random_seed)
        eligible_ids = np.flatnonzero(non_self & eligible_targets).astype(
            np.int64, copy=False
        )
        eligible_order = eligible_ids[np.argsort(types[eligible_ids], kind="stable")]
        active_channels, starts, counts = np.unique(
            types[eligible_order], return_index=True, return_counts=True
        )
        if np.any(active_channels < 0) or np.any(active_channels >= n_channels):
            raise ValueError("Local graph contains an out-of-range edge type")
        selected_by_channel = {}
        for channel, start, available in zip(active_channels, starts, counts):
            eligible = eligible_order[start : start + available]
            count = min(len(eligible), max_per_channel)
            selected = (
                rng.choice(eligible, count, replace=False)
                if count < len(eligible)
                else eligible
            )
            selected_by_channel[int(channel)] = rng.permutation(
                np.asarray(selected, dtype=np.int64)
            )
        selected_ids = []
        selected_targets = []
        for offset in range(
            max((len(ids) for ids in selected_by_channel.values()), default=0)
        ):
            channel_order = np.asarray(
                [
                    channel
                    for channel in active_channels
                    if len(selected_by_channel[int(channel)]) > offset
                ],
                dtype=np.int64,
            )
            rng.shuffle(channel_order)
            for channel in channel_order:
                selected_ids.append(selected_by_channel[int(channel)][offset])
                selected_targets.append(int(channel))
                if len(selected_ids) >= max_total:
                    break
            if len(selected_ids) >= max_total:
                break
        if not selected_ids:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32),
            )
        edge_ids = np.asarray(selected_ids, dtype=np.int64)
        targets = np.asarray(selected_targets, dtype=np.int64)
        sampled_channels, sampled_counts = np.unique(targets, return_counts=True)
        weights = np.zeros(len(targets), dtype=np.float32)
        for channel, count in zip(sampled_channels, sampled_counts):
            weights[targets == channel] = 1.0 / (len(sampled_channels) * int(count))
        permutation = rng.permutation(len(edge_ids))
        return edge_ids[permutation], targets[permutation], weights[permutation]

    def _require_trained(self) -> None:
        if not self._trained or self.model is None or self._feature_graphs is None:
            raise RuntimeError("CellSTIC has not completed training")

    def _fused_features(self) -> torch.Tensor:
        self._require_trained()
        self.feat_integrator.eval()
        with torch.no_grad():
            fused, _ = self.feat_integrator(self._feature_graphs)
        return fused

    def _predict_features(self) -> AnnData:
        primary = self.modality_datas[0]
        fused = self._fused_features().detach().cpu().numpy()
        if not self.skip_spatial_domain_prediction:
            n_clusters = (
                self.feat_train_cfg.n_clusters
                if self.auto_n_clusters is None
                else self.auto_n_clusters
            )
            cluster_labels = ClusteringUtils.cluster_louvain(
                fused,
                primary.obs_names,
                n_clusters=int(n_clusters),
                random_state=self.config.seed,
            ).astype(str)
            primary.obs["cluster"] = pd.Categorical(
                cluster_labels, categories=sorted(np.unique(cluster_labels))
            )
        primary.obsm["out"] = fused
        return primary

    def _build_level_graph(
        self,
        level_data: Dict[str, Any],
        modality1_features: torch.Tensor,
        spatial_coords: torch.Tensor,
    ) -> dgl.DGLGraph:
        if self._base_graph is None:
            raise RuntimeError("CCC graph is unavailable")
        (
            mapped_graph,
            mapped_ligand,
            mapped_receptor,
            knn_strength,
        ) = EdgeTypeMapper.map_level_graph_and_strengths(
            self._base_graph,
            self._ligand_strength_adj,
            self._receptor_strength_adj,
            self._knn_per_modality,
            level_data,
            self._edge_type_map,
            normalization_leaf_count=len(self._edge_type_map),
        )
        graph = DGLGraphUtils.build_dgl_graph_from_virtual_types_graph(
            mapped_graph,
            node_features=modality1_features,
            device=self.device,
            spatial_coords=spatial_coords,
            ligand_strength_adj=mapped_ligand,
            receptor_strength_adj=mapped_receptor,
            knn_strength_adj=knn_strength,
        )
        return graph

    def _predict_ccc_level(
        self,
        level_idx: int,
        level_data: Dict[str, Any],
        modality1_features: torch.Tensor,
        spatial_coords: torch.Tensor,
        fused_features: torch.Tensor,
        candidate_pairs: np.ndarray,
        channel_indices: Optional[Sequence[int]] = None,
    ) -> List[sparse.csr_matrix]:
        graph = self._build_level_graph(level_data, modality1_features, spatial_coords)
        n_cells = spatial_coords.shape[0]
        n_model_types = len(level_data)
        selected_channels = (
            tuple(range(n_model_types))
            if channel_indices is None
            else tuple(int(index) for index in channel_indices)
        )
        if any(index < 0 or index >= n_model_types for index in selected_channels):
            raise ValueError("channel_indices contains an invalid model channel")
        if len(candidate_pairs) == 0 or not selected_channels:
            return [
                sparse.csr_matrix((n_cells, n_cells), dtype=np.float32)
                for _ in selected_channels
            ]
        values_by_channel = [
            np.empty(len(candidate_pairs), dtype=np.float32) for _ in selected_channels
        ]
        batch_size = self.precision_model.edge_prob_batch_size
        with torch.no_grad():
            encoded_nodes, _ = self.precision_model.encode_graph(
                graph,
                fused_features,
                collect_edge_features=False,
            )
            for start in range(0, len(candidate_pairs), batch_size):
                end = min(start + batch_size, len(candidate_pairs))
                positions = torch.as_tensor(
                    candidate_pairs[start:end],
                    dtype=torch.long,
                    device=self.device,
                )
                batch_probabilities = (
                    self.precision_model.edge_probabilities(
                        positions,
                        encoded_nodes,
                        level_idx,
                        selected_channels,
                    )
                    .cpu()
                    .numpy()
                )
                for output_index, values in enumerate(values_by_channel):
                    values[start:end] = batch_probabilities[:, output_index]
        return [
            sparse.csr_matrix(
                (
                    values_by_channel[output_index],
                    (candidate_pairs[:, 0], candidate_pairs[:, 1]),
                ),
                shape=(n_cells, n_cells),
            )
            for output_index in range(len(selected_channels))
        ]

    def _predict_ccc_stability(
        self,
        level_idx: int,
        level_data: Dict[str, Any],
        modality1_features: torch.Tensor,
        spatial_coords: torch.Tensor,
        fused_features: torch.Tensor,
        candidate_pairs: np.ndarray,
        null_cutoffs: np.ndarray,
        channel_indices: Optional[Sequence[int]] = None,
    ) -> List[sparse.csr_matrix]:
        """Estimate edge-selection stability using repeated MC-dropout inference."""
        n_cells = spatial_coords.shape[0]
        n_model_types = len(level_data)
        selected_channels = (
            tuple(range(n_model_types))
            if channel_indices is None
            else tuple(int(index) for index in channel_indices)
        )
        if any(index < 0 or index >= n_model_types for index in selected_channels):
            raise ValueError("channel_indices contains an invalid model channel")
        n_types = len(selected_channels)
        if len(candidate_pairs) == 0 or n_types == 0:
            return [
                sparse.csr_matrix((n_cells, n_cells), dtype=np.float32)
                for _ in range(n_types)
            ]
        config = self.config.postprocess
        cutoffs = torch.as_tensor(
            null_cutoffs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        if cutoffs.shape[1] != n_types:
            raise ValueError("null_cutoffs must contain one value per selected channel")
        selected_counts = np.zeros(
            (len(candidate_pairs), n_types),
            dtype=np.uint32,
        )
        base_graph = self._build_level_graph(
            level_data, modality1_features, spatial_coords
        )
        fork_devices = (
            [
                (
                    self.device.index
                    if self.device.index is not None
                    else torch.cuda.current_device()
                )
            ]
            if self.device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=fork_devices, enabled=True):
            progress = tqdm(
                range(config.stability_repeats),
                desc="Leaf CCC stability",
                mininterval=0.5,
            )
            for repeat in progress:
                torch.manual_seed(self.config.seed + 100_003 + repeat)
                self.precision_model.eval()
                for module in self.precision_model.modules():
                    if isinstance(module, torch.nn.Dropout):
                        module.train()
                with torch.no_grad():
                    encoded_nodes, _ = self.precision_model.encode_graph(
                        base_graph,
                        fused_features,
                        collect_edge_features=False,
                    )
                    batch_size = self.precision_model.edge_prob_batch_size
                    for start in range(0, len(candidate_pairs), batch_size):
                        end = min(start + batch_size, len(candidate_pairs))
                        positions = torch.as_tensor(
                            candidate_pairs[start:end],
                            dtype=torch.long,
                            device=self.device,
                        )
                        probabilities = (
                            self.precision_model.edge_probabilities(
                                positions,
                                encoded_nodes,
                                level_idx,
                                selected_channels,
                            )
                            >= cutoffs
                        )
                        selected_counts[start:end] += (
                            probabilities.cpu().numpy().astype(np.uint32, copy=False)
                        )
        self.precision_model.eval()
        stability = selected_counts.astype(np.float32) / float(config.stability_repeats)
        return [
            sparse.csr_matrix(
                (
                    stability[:, channel],
                    (candidate_pairs[:, 0], candidate_pairs[:, 1]),
                ),
                shape=(n_cells, n_cells),
            )
            for channel in range(n_types)
        ]

    @staticmethod
    def _aggregate_leaf_probabilities(
        leaf_probabilities: Sequence[sparse.spmatrix],
        leaf_edge_type_map: Mapping[str, int],
        level_data: Mapping[str, Any],
        n_cells: int,
    ) -> Tuple[List[sparse.csr_matrix], Dict[str, int]]:
        """Sum descendant leaf scores with one tree-wide normalization.

        CCC leaf channels are independent sigmoid scores rather than mutually
        exclusive class probabilities. Summation preserves cumulative evidence
        from multiple active LR pairs. Dividing every group by the total number
        of real leaves keeps every hierarchy level on the same data-independent
        scale and bounds scores to [0, 1]. Per-group min-max normalization is
        avoided because it would make groups and datasets incomparable.
        """
        expected_shape = (n_cells, n_cells)
        if len(leaf_probabilities) != len(leaf_edge_type_map):
            raise ValueError(
                "leaf_probabilities and leaf_edge_type_map must have equal length"
            )
        for name, index in leaf_edge_type_map.items():
            if not 0 <= index < len(leaf_probabilities):
                raise ValueError(f"Leaf channel {name!r} has invalid index {index}")
            if leaf_probabilities[index].shape != expected_shape:
                raise ValueError(
                    f"Leaf channel {name!r} must have shape {expected_shape}, "
                    f"got {leaf_probabilities[index].shape}"
                )
            data = leaf_probabilities[index].data
            if np.any(~np.isfinite(data)) or np.any(data < 0) or np.any(data > 1):
                raise ValueError(
                    f"Leaf channel {name!r} must contain finite scores in [0, 1]"
                )

        aggregated: List[sparse.csr_matrix] = []
        aggregated_edge_type_map: Dict[str, int] = {}
        for group_name in sorted(level_data):
            group_data = level_data[group_name]
            members = group_data if isinstance(group_data, list) else [group_data]
            descendant_names = sorted(
                {
                    str(member["edge_type_name"])
                    for member in members
                    if isinstance(member, Mapping)
                    and member.get("edge_type_name") is not None
                }
            )
            missing = [
                name for name in descendant_names if name not in leaf_edge_type_map
            ]
            if missing:
                raise ValueError(
                    f"Hierarchy group {group_name!r} references unknown leaf "
                    f"channels: {missing}"
                )
            if not descendant_names:
                continue

            group_matrix = sparse.csr_matrix(expected_shape, dtype=np.float32)
            for name in descendant_names:
                group_matrix = group_matrix + sparse.csr_matrix(
                    leaf_probabilities[leaf_edge_type_map[name]]
                )
            group_matrix = group_matrix / float(len(leaf_edge_type_map))
            group_matrix.eliminate_zeros()
            aggregated_edge_type_map[str(group_name)] = len(aggregated)
            aggregated.append(group_matrix.tocsr())
        if not aggregated:
            raise ValueError("Hierarchy level has no group with descendant leaves")
        return aggregated, aggregated_edge_type_map

    def _iter_tree_results(self) -> Iterator[Dict[str, Any]]:
        """Yield leaf predictions and normalized-sum ancestors root-to-leaf."""
        if not self._ccc_trained:
            return
        self._require_trained()
        primary = self.modality_datas[0]
        modality1_features = torch.as_tensor(
            primary.obsm["feat"], dtype=torch.float32, device=self.device
        )
        spatial_coords = torch.as_tensor(
            primary.obsm["spatial"], dtype=torch.float32, device=self.device
        )
        fused_features = self._fused_features()
        self.precision_model.eval()
        threshold = EdgeFilterUtils.compute_spot_distance_threshold(
            spatial_coords, n_spots=self.graph_generator.n_spots
        )
        candidate_pairs = EdgeFilterUtils.spatial_candidate_pairs(
            spatial_coords,
            threshold,
            max_neighbors=self.graph_generator.max_spatial_neighbors,
        )

        hierarchy_levels = _hierarchy_levels(self._hierarchy_dict)
        if not hierarchy_levels:
            raise RuntimeError("CCC hierarchy contains no levels")
        leaf_level_idx = len(hierarchy_levels) - 1
        leaf_level_key = hierarchy_levels[leaf_level_idx]
        leaf_level_data = self._hierarchy_dict[leaf_level_key]
        leaf_model_edge_type_map = {
            name: index for index, name in enumerate(sorted(leaf_level_data))
        }
        missing_leaf_names = sorted(
            set(self._edge_type_map) - set(leaf_model_edge_type_map)
        )
        if missing_leaf_names:
            raise ValueError(
                "Deepest hierarchy level omits model LR channels: "
                f"{missing_leaf_names}"
            )
        leaf_names = sorted(self._edge_type_map)
        leaf_channel_indices = [leaf_model_edge_type_map[name] for name in leaf_names]
        leaf_edge_type_map = {name: index for index, name in enumerate(leaf_names)}
        leaf_probabilities = self._predict_ccc_level(
            leaf_level_idx,
            leaf_level_data,
            modality1_features,
            spatial_coords,
            fused_features,
            candidate_pairs,
            channel_indices=leaf_channel_indices,
        )
        leaf_postprocess = None
        if self.postprocess_before_aggregation and self._base_graph is not None:
            channel_names = leaf_names
            cutoffs = None
            stability = None
            if self.config.postprocess.stability_repeats > 0:
                cutoffs = CCCPostprocessor.null_cutoffs(
                    leaf_probabilities,
                    channel_names,
                    self._base_graph,
                    self._edge_type_map,
                    candidate_pairs,
                    tail_fraction=self.config.postprocess.null_tail_fraction,
                    min_null_edges=self.config.postprocess.min_null_edges,
                )
                stability = self._predict_ccc_stability(
                    leaf_level_idx,
                    leaf_level_data,
                    modality1_features,
                    spatial_coords,
                    fused_features,
                    candidate_pairs,
                    cutoffs,
                    channel_indices=leaf_channel_indices,
                )
            leaf_postprocess = CCCPostprocessor.calibrate(
                leaf_probabilities,
                channel_names,
                self._base_graph,
                self._edge_type_map,
                candidate_pairs,
                tail_fraction=self.config.postprocess.null_tail_fraction,
                min_null_edges=self.config.postprocess.min_null_edges,
                stability_matrices=stability,
                stability_threshold=self.config.postprocess.stability_threshold,
                precomputed_null_cutoffs=cutoffs,
            )

        aggregation_leaf_probabilities = (
            leaf_postprocess.selected_scores
            if leaf_postprocess is not None
            else leaf_probabilities
        )

        for level_idx, level_key in enumerate(hierarchy_levels):
            level_num = int(level_key.split("_")[1])
            level_data = self._hierarchy_dict[level_key]
            is_leaf = level_idx == leaf_level_idx
            if is_leaf:
                probabilities = aggregation_leaf_probabilities
                edge_type_map = leaf_edge_type_map
            else:
                probabilities, edge_type_map = self._aggregate_leaf_probabilities(
                    aggregation_leaf_probabilities,
                    leaf_edge_type_map,
                    level_data,
                    primary.n_obs,
                )
            level_dir = self.output_path / f"level_{level_num}"
            result = {
                "level_key": level_key,
                "level_num": level_num,
                "pos_edge_probs_np": probabilities,
                "edge_type_map": edge_type_map,
                "output_source": (
                    "calibrated_leaf_prediction"
                    if is_leaf and leaf_postprocess is not None
                    else (
                        "model_leaf_prediction"
                        if is_leaf
                        else (
                            "normalized_sum_over_calibrated_leaf_predictions"
                            if leaf_postprocess is not None
                            else "normalized_sum_over_raw_leaf_predictions"
                        )
                    )
                ),
                "output_dir": level_dir,
                "adata": primary,
                "hierarchy_dict": self._hierarchy_dict,
            }
            if is_leaf and leaf_postprocess is not None:
                result["raw_leaf_probabilities"] = leaf_probabilities
                result["postprocess"] = leaf_postprocess
            yield result

    def _pack_output(
        self, adata: AnnData, tree_results: Iterable[Mapping[str, Any]]
    ) -> AnnData:
        level_meta: List[Dict[str, Any]] = []
        postprocess_meta: Optional[Dict[str, Any]] = None
        for result in tree_results:
            level_num = int(result["level_num"])
            edge_type_map = {
                str(name): int(index) for name, index in result["edge_type_map"].items()
            }
            probabilities = result["pos_edge_probs_np"]
            channel_indices = sorted(edge_type_map.values())
            if channel_indices != list(range(len(edge_type_map))):
                raise ValueError(
                    f"Level {level_num} edge_type_map indices must be contiguous "
                    f"from zero, got {channel_indices}"
                )
            output_keys = [_obsp_key(level_num, name) for name in edge_type_map]
            if len(set(output_keys)) != len(output_keys):
                raise ValueError(
                    f"Level {level_num} channel names collide after output-key "
                    "sanitization"
                )
            if isinstance(probabilities, (list, tuple)) and len(probabilities) != len(
                edge_type_map
            ):
                raise ValueError(
                    f"Level {level_num} has {len(probabilities)} probability "
                    f"matrices for {len(edge_type_map)} channels"
                )
            for channel_name, channel_idx in edge_type_map.items():
                if isinstance(probabilities, (list, tuple)):
                    matrix = sparse.csr_matrix(
                        probabilities[channel_idx], dtype=np.float32
                    )
                else:
                    matrix = sparse.csr_matrix(
                        np.asarray(probabilities, dtype=np.float32)[:, :, channel_idx]
                    )
                if matrix.shape != (adata.n_obs, adata.n_obs):
                    raise ValueError(
                        f"Level {level_num} channel {channel_name!r} must have "
                        f"shape ({adata.n_obs}, {adata.n_obs}), got {matrix.shape}"
                    )
                if (
                    np.any(~np.isfinite(matrix.data))
                    or np.any(matrix.data < 0)
                    or np.any(matrix.data > 1)
                ):
                    raise ValueError(
                        f"Level {level_num} channel {channel_name!r} must "
                        "contain finite probabilities in [0, 1]"
                    )
                diagonal = matrix.diagonal()
                if np.any(diagonal):
                    matrix = matrix - sparse.diags(diagonal, format="csr")
                matrix.eliminate_zeros()
                adata.obsp[_obsp_key(level_num, channel_name)] = matrix
            raw_leaf_probabilities = result.get("raw_leaf_probabilities")
            if raw_leaf_probabilities is not None:
                for channel_name, channel_idx in edge_type_map.items():
                    adata.obsp[
                        _postprocess_obsp_key(level_num, channel_name, "raw")
                    ] = sparse.csr_matrix(
                        raw_leaf_probabilities[channel_idx], dtype=np.float32
                    )
            postprocess = result.get("postprocess")
            if postprocess is not None:
                for channel_name, channel_idx in edge_type_map.items():
                    outputs = {
                        "null_tail": postprocess.null_tail_probabilities[channel_idx]
                    }
                    if postprocess.stability is not None:
                        outputs["stability"] = postprocess.stability[channel_idx]
                    for statistic, matrix in outputs.items():
                        adata.obsp[
                            _postprocess_obsp_key(level_num, channel_name, statistic)
                        ] = sparse.csr_matrix(matrix, dtype=np.float32)
                postprocess_meta = {
                    "method": "descriptive_expression_negative_score_calibration",
                    "statistical_interpretation": (
                        "in-sample descriptive reference; null-tail values are "
                        "not p-values and do not provide FDR control"
                    ),
                    "leaf_level_num": level_num,
                    "null_tail_fraction": (
                        self.config.postprocess.null_tail_fraction
                    ),
                    "min_null_edges": self.config.postprocess.min_null_edges,
                    "stability_method": (
                        "MC-dropout selection frequency"
                        if self.config.postprocess.stability_repeats > 0
                        else "disabled"
                    ),
                    "stability_repeats": self.config.postprocess.stability_repeats,
                    "stability_threshold": (
                        self.config.postprocess.stability_threshold
                    ),
                    "output_suffixes": {
                        "selected_leaf_score": "",
                        "raw_leaf_score": "raw",
                        "descriptive_null_tail_probability": "null_tail",
                        "stability_frequency": (
                            "stability"
                            if self.config.postprocess.stability_repeats > 0
                            else None
                        ),
                    },
                    "summary": postprocess.summary,
                }
            level_meta.append(
                {
                    "level_key": str(result["level_key"]),
                    "level_num": level_num,
                    "edge_type_map": edge_type_map,
                    "output_source": str(result["output_source"]),
                }
            )

        metadata = {
            "schema_version": CELLSTIC_SCHEMA_VERSION,
            "hierarchy_level_order": "root_to_leaf",
            "root_level_num": 1,
            "nonleaf_aggregation": (
                "sum_descendant_leaf_scores_divided_by_total_leaf_count"
            ),
            "postprocess_before_aggregation": (self.postprocess_before_aggregation),
            "aggregation_source": (
                "calibrated_leaf_scores"
                if self.postprocess_before_aggregation
                else "raw_leaf_scores"
            ),
            "feature_checkpoint": str(self.checkpoint_path / FEATURE_CHECKPOINT_NAME),
            "ccc_checkpoint": str(self.checkpoint_path / CCC_CHECKPOINT_NAME),
            "ccc_trained": bool(self._ccc_trained),
            "spatial_domain_prediction_skipped": self.skip_spatial_domain_prediction,
            "hierarchy_dict_json": json.dumps(
                self._hierarchy_dict, default=_json_default, ensure_ascii=False
            ),
            "tree_levels_json": json.dumps(
                level_meta, default=_json_default, ensure_ascii=False
            ),
            "model_edge_type_map_json": json.dumps(
                self._edge_type_map, default=_json_default, ensure_ascii=False
            ),
            "config_json": json.dumps(
                _config_dict(self.config), default=_json_default, ensure_ascii=False
            ),
            "ccc_postprocess_json": (
                json.dumps(postprocess_meta, default=_json_default, ensure_ascii=False)
                if postprocess_meta is not None
                else "null"
            ),
        }
        adata.uns["cellstic"] = metadata
        return adata

    def run(self) -> AnnData:
        """Train, infer, write ``out.h5ad``, and return the exported AnnData."""
        if self._trained:
            raise RuntimeError("CellSTICPipeline instances are single-use")
        self._train()
        adata_out = self._predict_features()
        self._pack_output(adata_out, self._iter_tree_results())
        output_file = self.result_path / "out.h5ad"
        with tempfile.NamedTemporaryFile(
            prefix=".cellstic-",
            suffix=".h5ad",
            dir=self.result_path,
            delete=False,
        ) as handle:
            temporary_file = Path(handle.name)
        try:
            adata_out.write_h5ad(temporary_file)
            temporary_file.replace(output_file)
        finally:
            temporary_file.unlink(missing_ok=True)
        checkpoints = [self.checkpoint_path / FEATURE_CHECKPOINT_NAME]
        if self.run_ccc:
            checkpoints.append(self.checkpoint_path / CCC_CHECKPOINT_NAME)
        print(
            f"CellSTIC done -> {output_file}; checkpoints: "
            + ", ".join(str(path) for path in checkpoints)
        )
        return adata_out


def run_cellstic(
    modality_datas: Sequence[AnnData],
    config: CellSTICConfig,
    *,
    output_path: PathLike,
    ligand_receptor_map: Optional[Dict] = None,
    cell_chat_db: Optional[Dict] = None,
    modality_types: Optional[Sequence[str]] = None,
    auto_n_clusters: Optional[int] = None,
    run_ccc: bool = True,
    skip_spatial_domain_prediction: bool = False,
    postprocess_before_aggregation: bool = True,
    device: Optional[Union[str, torch.device]] = None,
) -> AnnData:
    """Run CellSTIC and save the exported ``out.h5ad`` result."""
    pipeline = CellSTICPipeline(
        modality_datas,
        config,
        output_path=output_path,
        ligand_receptor_map=ligand_receptor_map,
        cell_chat_db=cell_chat_db,
        modality_types=modality_types,
        auto_n_clusters=auto_n_clusters,
        run_ccc=run_ccc,
        skip_spatial_domain_prediction=skip_spatial_domain_prediction,
        postprocess_before_aggregation=postprocess_before_aggregation,
        device=device,
    )
    return pipeline.run()


__all__ = ["CellSTICPipeline", "prepare_modalities", "run_cellstic"]
