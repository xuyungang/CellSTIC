from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Union

from anndata import AnnData

from .config import CellSTICConfig, apply_config_overrides


def _infer_feature_dim(adata: AnnData) -> int:
    """Infer feature dimension from `obsm['feat']` first, fallback to `X`."""
    feat = adata.obsm.get("feat", None)
    if feat is not None and getattr(feat, "ndim", None) == 2 and feat.shape[1] > 0:
        return int(feat.shape[1])
    if getattr(adata, "X", None) is not None and adata.X.shape[1] > 0:
        return int(adata.X.shape[1])
    raise ValueError("Cannot infer feature dim from adata: missing valid obsm['feat'] and X.")


def _build_feat_encoder_dims(input_dim: int) -> List[int]:
    if input_dim >= 300:
        return [input_dim, 300, 200]
    if input_dim >= 160:
        return [input_dim, 200, 120]
    if input_dim == 100:
        return [100, 85, 60]
    if input_dim >= 80:
        return [input_dim, 100, 80]
    if input_dim >= 40:
        h1 = max(32, int(round(input_dim * 0.75)))
        h2 = max(20, int(round(input_dim * 0.45)))
        return [input_dim, h1, min(h1 - 1, h2) if h1 > h2 else h2]
    return [input_dim, max(16, int(round(input_dim * 0.67)))]


def _build_feat_decoder_dims(input_dim: int, encoder_dims: List[int]) -> List[int]:
    hidden = encoder_dims[-1]
    return [hidden, max(hidden + 1, input_dim - 10)]


def _build_feat_output_dims(fused_input_dim: int, encoder_last: int) -> List[int]:
    mid = max(encoder_last + 40, fused_input_dim - 40)
    return [fused_input_dim, mid, encoder_last]


def _build_ccc_encoder_dims(first_modality_dim: int) -> List[int]:
    if first_modality_dim >= 300:
        return [first_modality_dim, 300, 200]
    if first_modality_dim >= 160:
        return [first_modality_dim, 200, 120]
    if first_modality_dim >= 80:
        return [first_modality_dim, 100, 80]
    return [first_modality_dim, max(32, int(round(first_modality_dim * 0.7)))]


def _build_ccc_decoder_dims(ccc_encoder_dims: List[int], feat_output_last: int) -> List[int]:
    first = ccc_encoder_dims[-1] * 2 + feat_output_last * 2
    second = 150 if first >= 300 else 200
    return [first, second]


def _apply_config_template(config: CellSTICConfig, template: CellSTICConfig) -> None:
    config.model.graph = template.model.graph
    config.model.tree.hierarchy_method = template.model.tree.hierarchy_method
    config.model.dropout = template.model.dropout
    config.model.ccc.temperature = template.model.ccc.temperature


def build_config(
    adatas: Union[AnnData, Sequence[AnnData]],
    hierarchy_method: str = "balanced",
    template: Optional[CellSTICConfig] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> CellSTICConfig:
    """Build an in-memory ``CellSTICConfig`` from modality AnnData list and dim heuristics."""
    from .config import (
        CellSTICCCCConfig,
        CellSTICFeatConfig,
        CellSTICModelConfig,
        CellSTICTreeConfig,
    )

    adata_list = [adatas] if isinstance(adatas, AnnData) else list(adatas)
    if not adata_list:
        raise ValueError("adatas must contain at least one AnnData.")

    if template is not None and template.model.tree.hierarchy_method:
        hierarchy_method = template.model.tree.hierarchy_method

    feature_dims = [_infer_feature_dim(a) for a in adata_list]
    feat_encoder_dims = [_build_feat_encoder_dims(d) for d in feature_dims]
    feat_decoder_dims = [
        _build_feat_decoder_dims(input_dim, enc)
        for input_dim, enc in zip(feature_dims, feat_encoder_dims)
    ]
    fused_input_dim = int(sum(dims[-1] for dims in feat_encoder_dims))
    encoder_last = feat_encoder_dims[0][-1]
    feat_output_dims = _build_feat_output_dims(fused_input_dim, encoder_last)
    ccc_encoder_dims = _build_ccc_encoder_dims(feature_dims[0])
    ccc_decoder_dims = _build_ccc_decoder_dims(ccc_encoder_dims, feat_output_dims[-1])

    config = CellSTICConfig(
        model=CellSTICModelConfig(
            ccc=CellSTICCCCConfig(
                encoder_dims=ccc_encoder_dims,
                decoder_dims=ccc_decoder_dims,
            ),
            feat=CellSTICFeatConfig(
                encoder_dims=feat_encoder_dims,
                decoder_dims=feat_decoder_dims,
                output_dims=feat_output_dims,
            ),
            tree=CellSTICTreeConfig(hierarchy_method=hierarchy_method),
        ),
    )

    if template is not None:
        _apply_config_template(config, template)

    if overrides:
        config = apply_config_overrides(config, overrides)
    return config
