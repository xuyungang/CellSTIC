"""Training artifacts and CellSTIC result AnnData pack / load utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import dgl
import numpy as np
from anndata import AnnData
from scipy import sparse

from .config import CellSTICConfig

CELLSTIC_UNS_KEY = "cellstic"
CELLSTIC_SCHEMA_VERSION = 1
_OBSP_PREFIX = "cellstic_L"


@dataclass
class CellSTICTrainArtifacts:
    """In-memory CCC / feature graph data produced during training (not written to disk)."""

    modality_g_dgls: List[dgl.DGLGraph]
    base_graph_adj: np.ndarray
    edge_type_map: Dict[str, int]
    hierarchy_dict: Dict[str, Any]
    ligand_strength_adj: Optional[np.ndarray] = None
    receptor_strength_adj: Optional[np.ndarray] = None
    knn_per_modality: Optional[np.ndarray] = None


def _sanitize_lr_key(name: str) -> str:
    return str(name).replace(":", "_").replace("/", "_").replace(" ", "_").strip() or "channel"


def obsp_key(level_num: int, lr_name: str) -> str:
    """Obsp matrix key for one LR channel at a tree level."""
    return f"{_OBSP_PREFIX}{level_num}_lr_{_sanitize_lr_key(lr_name)}"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def _dump_json_blob(value: Any) -> str:
    return json.dumps(value, default=_json_default, ensure_ascii=False)


def _load_json_blob(raw: Union[str, bytes]) -> Any:
    return json.loads(raw)


def materialize_cellstic_meta(meta: Mapping[str, Any]) -> Dict[str, Any]:
    """Expand JSON-encoded tree metadata stored in ``adata.uns['cellstic']``."""
    out = dict(meta)
    if "hierarchy_dict_json" in out:
        out["hierarchy_dict"] = _load_json_blob(out["hierarchy_dict_json"])
    if "tree_levels_json" in out:
        out["tree_levels"] = _load_json_blob(out["tree_levels_json"])
    if "model_edge_type_map_json" in out:
        out["model_edge_type_map"] = _load_json_blob(out["model_edge_type_map_json"])
    if "config_json" in out:
        out["config"] = _load_json_blob(out["config_json"])
    return out


def require_cellstic_meta(adata: AnnData) -> Dict[str, Any]:
    """Return parsed ``adata.uns['cellstic']`` or raise if missing / invalid."""
    if CELLSTIC_UNS_KEY not in adata.uns:
        raise KeyError(
            f"adata.uns[{CELLSTIC_UNS_KEY!r}] not found; "
            "expected an AnnData produced by pipeline.run_cellstic."
        )
    meta = materialize_cellstic_meta(adata.uns[CELLSTIC_UNS_KEY])
    if int(meta.get("schema_version", 0)) != CELLSTIC_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported CellSTIC adata schema version: {meta.get('schema_version')!r}"
        )
    if "hierarchy_dict" not in meta or "tree_levels" not in meta:
        raise KeyError(
            f"adata.uns[{CELLSTIC_UNS_KEY!r}] is missing tree metadata "
            "(hierarchy_dict / tree_levels)."
        )
    return meta


def pack_results_into_adata(
    adata: AnnData,
    tree_results: List[Dict[str, Any]],
    *,
    config: Optional[CellSTICConfig] = None,
    model_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    auto_n_clusters: Optional[int] = None,
    model_edge_type_map: Optional[Dict[str, int]] = None,
) -> AnnData:
    """Pack tree-level CCC matrices and hierarchy metadata into ``adata`` for h5ad export."""
    if not tree_results:
        raise ValueError("tree_results is empty; nothing to pack into adata.")

    hierarchy_dict = tree_results[0].get("hierarchy_dict")
    if hierarchy_dict is None:
        raise ValueError("tree_results[0] missing hierarchy_dict.")

    level_meta: List[Dict[str, Any]] = []
    for result in sorted(tree_results, key=lambda r: int(r["level_num"])):
        level_num = int(result["level_num"])
        edge_type_map = dict(result["edge_type_map"])
        probs = np.asarray(result["pos_edge_probs_np"], dtype=np.float32)

        for lr_name, channel_idx in edge_type_map.items():
            adata.obsp[obsp_key(level_num, lr_name)] = sparse.csr_matrix(
                probs[:, :, int(channel_idx)]
            )

        level_meta.append(
            {
                "level_key": result["level_key"],
                "level_num": level_num,
                "edge_type_map": edge_type_map,
            }
        )

    meta: Dict[str, Any] = {
        "schema_version": CELLSTIC_SCHEMA_VERSION,
        "output_path": str(Path(output_path)) if output_path is not None else None,
        "model_path": str(Path(model_path)) if model_path is not None else None,
        "hierarchy_method": config.model.tree.hierarchy_method if config is not None else None,
        "n_spots": config.model.graph.n_spots if config is not None else None,
        "deepest_level": max(int(r["level_num"]) for r in tree_results),
        "auto_n_clusters": auto_n_clusters,
        "hierarchy_dict_json": _dump_json_blob(hierarchy_dict),
        "tree_levels_json": _dump_json_blob(level_meta),
    }
    if config is not None:
        meta["config_json"] = _dump_json_blob(asdict(config))
    if model_edge_type_map is not None:
        meta["model_edge_type_map_json"] = _dump_json_blob(model_edge_type_map)

    adata.uns[CELLSTIC_UNS_KEY] = meta
    return adata


def ccc_ground_from_adata(adata: AnnData) -> Optional[np.ndarray]:
    """Return optional ground-truth CCC tensor if stored in ``obsp['cellstic_ccc_ground']``."""
    meta = require_cellstic_meta(adata)
    if "cellstic_ccc_ground" not in adata.obsp:
        return None
    mat = adata.obsp["cellstic_ccc_ground"]
    dense = mat.toarray() if sparse.issparse(mat) else np.asarray(mat)
    shape = meta.get("ccc_ground_shape")
    if shape is not None:
        return dense.reshape(tuple(shape))
    return dense


def load_cellstic_adata(path: Union[str, Path]) -> AnnData:
    """Load a saved CellSTIC result AnnData and validate bundled metadata."""
    adata = AnnData.read_h5ad(path)
    require_cellstic_meta(adata)
    return adata
