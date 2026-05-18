from __future__ import annotations

from pathlib import Path
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import yaml
from anndata import AnnData

_SCALAR_TYPES = (int, float, bool, str, type(None))


def _is_flat_scalar_list(obj: Any) -> bool:
    """True if `obj` is a non-empty list of YAML scalars (no nested list/dict)."""
    return (
        isinstance(obj, list)
        and len(obj) > 0
        and all(isinstance(x, _SCALAR_TYPES) for x in obj)
    )


class _ConfigYAMLDumper(yaml.SafeDumper):
    """SafeDumper variant: flat scalar lists use flow style, e.g. `encoder_dims: [100, 120, 80]`."""


def _represent_list(dumper: yaml.SafeDumper, data: list) -> yaml.nodes.Node:
    flow = _is_flat_scalar_list(data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)


_ConfigYAMLDumper.add_representer(list, _represent_list)


def _infer_feature_dim(adata: AnnData) -> int:
    """Infer feature dimension from `obsm['feat']` first, fallback to `X`."""
    feat = adata.obsm.get("feat", None)
    if feat is not None and getattr(feat, "ndim", None) == 2 and feat.shape[1] > 0:
        return int(feat.shape[1])
    if getattr(adata, "X", None) is not None and adata.X.shape[1] > 0:
        return int(adata.X.shape[1])
    raise ValueError("Cannot infer feature dim from adata: missing valid obsm['feat'] and X.")


def _build_feat_encoder_dims(input_dim: int) -> List[int]:
    """
    Build modality encoder dims based on input dim.
    Keep structure similar to existing configs while adapting first-layer dim.
    """
    if input_dim >= 300:
        return [input_dim, 300, 200]
    if input_dim >= 160:
        return [input_dim, 200, 120]
    # Align with hand-tuned NSF configs (`data/nsf/config/nsf_config.yaml`): 100 -> [100, 85, 60]
    if input_dim == 100:
        return [100, 85, 60]
    if input_dim >= 80:
        return [input_dim, 100, 60]
    if input_dim >= 40:
        h1 = max(32, int(round(input_dim * 0.75)))
        h2 = max(20, int(round(input_dim * 0.45)))
        return [input_dim, h1, min(h1 - 1, h2) if h1 > h2 else h2]
    return [input_dim, max(16, int(round(input_dim * 0.67)))]


def _build_ccc_encoder_dims(first_modality_dim: int) -> List[int]:
    """
    Build CCC encoder dims from first modality dimension.
    """
    if first_modality_dim >= 300:
        return [first_modality_dim, 300, 200]
    if first_modality_dim >= 160:
        return [first_modality_dim, 200, 120]
    if first_modality_dim >= 80:
        return [first_modality_dim, 120, 80]
    return [first_modality_dim, max(32, int(round(first_modality_dim * 0.7)))]


def _deep_update(base: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Recursively update nested dict fields in `base` using `updates`.
    """
    for key, value in updates.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, Mapping)
        ):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def generate_config(
    config_path: Union[str, Path],
    adatas: Union[AnnData, Sequence[AnnData]],
    hierarchy_method: str,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate YAML config from user-provided adata list.

    Notes:
    - Keep training and graph hyperparameters unchanged (fixed defaults).
    - Auto-adapt model feature dimensions from adata feature dims.
    - User must provide `hierarchy_method`.
    - User can override any generated/default field via `overrides`.
    """
    if isinstance(adatas, AnnData):
        adata_list = [adatas]
    else:
        adata_list = list(adatas)

    if len(adata_list) == 0:
        raise ValueError("adatas must contain at least one AnnData.")

    feature_dims = [_infer_feature_dim(a) for a in adata_list]
    feat_encoder_dims = [_build_feat_encoder_dims(d) for d in feature_dims]
    feat_decoder_dims = [[dims[-1], dims[0]] for dims in feat_encoder_dims]
    fused_input_dim = int(sum(dims[-1] for dims in feat_encoder_dims))
    feat_output_tail = 100
    feat_output_dims = [fused_input_dim, feat_output_tail]
    ccc_encoder_dims = _build_ccc_encoder_dims(feature_dims[0])
    ccc_decoder_first = ccc_encoder_dims[-1] * 2 + feat_output_dims[-1] * 2
    ccc_decoder_dims = [ccc_decoder_first, 200]

    config: Dict[str, Any] = {
        "model": {
            "dropout": 0.1,
            "ccc": {
                "encoder_dims": ccc_encoder_dims,
                "decoder_dims": ccc_decoder_dims,
            },
            "feat": {
                "encoder_dims": feat_encoder_dims,
                "decoder_dims": feat_decoder_dims,
                "output_dims": feat_output_dims,
            },
            "graph": {
                "cluster_top_k": 20,
                "cluster_size": 60,
                "knn_top_k": 20,
                "expression_percentile": 75,
                "n_spots": 10,
            },
            "tree": {
                "hierarchy_method": hierarchy_method,
            },
        },
        "train": {
            "ccc": {
                "epochs": 300,
                "learning_rate": 0.004,
                "weight_decay": 0.00,
                "edge_type_loss_weight": 1,
                "sampling_rate": 0.3,
            },
            "feat": {
                "epochs": 300,
                "learning_rate": 0.001,
                "n_clusters": 9,
                "weight_decay": 0.00,
                "weight_modality": 0.5,
                "entropy_weight": 0.8,
            },
        },
    }

    if overrides:
        config = _deep_update(deepcopy(config), overrides)

    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(
            config,
            f,
            Dumper=_ConfigYAMLDumper,
            sort_keys=False,
            allow_unicode=False,
            default_flow_style=False,
        )
    return config
