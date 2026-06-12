"""End-to-end CellSTIC runner: train, evaluate, pack results into a self-contained AnnData."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from anndata import AnnData
from scipy import sparse

from model import CellSTIC
from model.train import CellSTICConfig, CellSTICTrainConfig
from model.train.config_gen import build_config
from model.train.data import (
    CELLSTIC_UNS_KEY,
    obsp_key,
    pack_results_into_adata,
    require_cellstic_meta,
)

from .evaluator import CellSTICEvaluator
from .trainer import CellSTICTrainer


def _modality_tensors_from_adata(
    modality_datas: List[AnnData],
    device: torch.device,
) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
    """Return ``(modality1_features, modality_features, spatial_coords)`` on ``device``."""
    m0 = modality_datas[0]
    modality1_features = torch.tensor(m0.obsm["feat"], dtype=torch.float32).to(device)
    spatial_coords = torch.tensor(m0.obsm["spatial"], dtype=torch.float32).to(device)
    modality_features = [
        torch.tensor(d.obsm["feat"], dtype=torch.float32).to(device) for d in modality_datas
    ]
    return modality1_features, modality_features, spatial_coords


def _spatial_distances_from_adata(m0: AnnData):
    """Return ``obsp['spatial_distances']`` or ``None`` if missing or too small."""
    spatial_distances = m0.obsp.get("spatial_distances")
    if spatial_distances is None:
        return None
    sd_arr = spatial_distances.toarray() if hasattr(spatial_distances, "toarray") else np.asarray(spatial_distances)
    if sd_arr.ndim < 2 or sd_arr.shape[0] < 2:
        return None
    return spatial_distances


def reconstruct_pos_edge_probs(
    adata: AnnData,
    level_num: int,
    edge_type_map: Mapping[str, int],
) -> np.ndarray:
    """Rebuild ``(n_cells, n_cells, n_lr)`` tensor from packed obsp matrices."""
    n_cells = adata.n_obs
    n_types = max(edge_type_map.values()) + 1 if edge_type_map else 0
    probs = np.zeros((n_cells, n_cells, n_types), dtype=np.float32)
    for lr_name, channel_idx in edge_type_map.items():
        key = obsp_key(level_num, lr_name)
        if key not in adata.obsp:
            raise KeyError(f"Missing obsp[{key!r}] required to reconstruct tree level {level_num}.")
        mat = adata.obsp[key]
        dense = mat.toarray() if sparse.issparse(mat) else np.asarray(mat)
        probs[:, :, int(channel_idx)] = dense.astype(np.float32, copy=False)
    return probs


def tree_results_from_adata(
    adata: AnnData,
    output_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Reconstruct tree-level evaluator outputs from a packed CellSTIC AnnData."""
    meta = require_cellstic_meta(adata)
    hierarchy_dict = meta["hierarchy_dict"]
    base_output = Path(output_path if output_path is not None else meta.get("output_path") or ".")

    results: List[Dict[str, Any]] = []
    for level_info in meta["tree_levels"]:
        level_num = int(level_info["level_num"])
        edge_type_map = dict(level_info["edge_type_map"])
        results.append(
            {
                "level_key": level_info["level_key"],
                "level_num": level_num,
                "pos_edge_probs_np": reconstruct_pos_edge_probs(adata, level_num, edge_type_map),
                "edge_type_map": edge_type_map,
                "output_dir": base_output / f"tree_level_{level_num}",
                "adata": adata,
                "hierarchy_dict": hierarchy_dict,
            }
        )
    return results


def single_level_from_adata(
    adata: AnnData,
    level_num: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Return ``(pos_edge_probs_np, edge_type_map)`` for one tree level (default: deepest)."""
    meta = require_cellstic_meta(adata)
    if level_num is None:
        level_num = int(meta["deepest_level"])

    for level_info in meta["tree_levels"]:
        if int(level_info["level_num"]) == int(level_num):
            edge_type_map = dict(level_info["edge_type_map"])
            return reconstruct_pos_edge_probs(adata, int(level_num), edge_type_map), edge_type_map

    raise ValueError(f"Tree level {level_num} not found in adata.uns[{CELLSTIC_UNS_KEY!r}].")


@dataclass
class CellSTICRunResult:
    """Outputs from a full CellSTIC train + evaluate run."""

    model: CellSTIC
    adata: AnnData
    tree_results: List[Dict[str, Any]]
    model_path: Path
    output_path: Path
    adata_path: Path


def run_cellstic(
    modality_datas: List[AnnData],
    ligand_receptor_map: Dict,
    model_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    config: Optional[CellSTICConfig] = None,
    device: Optional[torch.device] = None,
    cell_chat_db: Optional[Dict] = None,
    lr_pair_type_constraints: Optional[Dict[str, List[Tuple[int, int]]]] = None,
    auto_n_clusters: Optional[int] = None,
) -> CellSTICRunResult:
    """
    Train, run feature + tree CCC evaluation, pack everything into the primary AnnData,
    and save it.

    The saved AnnData contains fused embeddings, cluster labels, all tree-level CCC
    matrices, and JSON-encoded hierarchy metadata — sufficient for downstream
    ``SingleLevelAnalysis.from_adata`` / ``TreeLevelAnalysis.from_adata``.

    Optional ``config`` supplies model / train overrides; model dims are inferred from
    ``modality_datas`` inside this function (graph / tree / dropout from ``config`` when given).
    """
    if not modality_datas:
        raise ValueError("modality_datas must contain at least one AnnData.")

    built = build_config(modality_datas, template=config)
    train_cfg = config.train if config is not None else CellSTICTrainConfig()
    model_cfg = built.model
    if not model_cfg.feat.encoder_dims:
        raise ValueError(
            "Model feat.encoder_dims is empty after build_config(). "
            "Each modality AnnData needs obsm['feat'] or a valid X matrix."
        )
    config = CellSTICConfig(model=model_cfg, train=train_cfg)

    resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved_model_path = Path(model_path)
    resolved_output_path = Path(output_path)
    resolved_output_path.mkdir(parents=True, exist_ok=True)
    resolved_model_path.mkdir(parents=True, exist_ok=True)

    resolved_adata_path = resolved_output_path / "cellstic_result.h5ad"
    resolved_adata_path.parent.mkdir(parents=True, exist_ok=True)

    model = CellSTIC(model_cfg, resolved_device)
    m0 = modality_datas[0]
    modality1_features, modality_features, spatial_coords = _modality_tensors_from_adata(
        modality_datas, resolved_device
    )
    spatial_distances = _spatial_distances_from_adata(m0)
    gene_expression = m0.X
    genes = m0.var_names.tolist()

    trainer = CellSTICTrainer(
        model,
        config,
        model_path=resolved_model_path,
        ligand_receptor_map=ligand_receptor_map,
        lr_pair_type_constraints=lr_pair_type_constraints,
        cell_chat_db=cell_chat_db,
        device=resolved_device,
    )
    train_artifacts = trainer.train(
        primary_adata=m0,
        modality1_features=modality1_features,
        modality_features=modality_features,
        spatial_coords=spatial_coords,
        spatial_distances=spatial_distances,
        gene_expression=gene_expression,
        genes=genes,
        is_train_ccc=True,
        is_train_feature=True,
    )

    evaluator = CellSTICEvaluator(
        model,
        config,
        ligand_receptor_map=ligand_receptor_map,
        train_artifacts=train_artifacts,
        lr_pair_type_constraints=lr_pair_type_constraints,
        output_path=resolved_output_path,
        device=resolved_device,
    )

    evaluator.evaluate_mutiple_feature(
        primary_adata=m0,
        auto_n_clusters=auto_n_clusters,
    )

    tree_results = evaluator.evaluate_ccc_precision_tree(
        primary_adata=m0,
        modality1_features=modality1_features,
        spatial_coords=spatial_coords,
    )

    adata = modality_datas[0].copy()
    pack_results_into_adata(
        adata,
        tree_results,
        config=config,
        model_path=resolved_model_path,
        output_path=resolved_output_path,
        auto_n_clusters=auto_n_clusters,
        model_edge_type_map=train_artifacts.edge_type_map,
    )
    adata.write_h5ad(resolved_adata_path)

    return CellSTICRunResult(
        model=model,
        adata=adata,
        tree_results=tree_results,
        model_path=resolved_model_path,
        output_path=resolved_output_path,
        adata_path=resolved_adata_path,
    )
