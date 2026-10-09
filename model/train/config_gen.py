"""Dataset-specific overrides for the shared CellSTIC configuration.

Global model, graph, and optimization defaults live in :mod:`.config`.  The
presets below contain only differences required by modality dimensions,
dataset-specific neighbourhood scales, clustering granularity, or an explicit
hierarchy choice.  Every stage or replicate of a dataset resolves to the same
immutable configuration.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Mapping, Optional, Sequence

from .config import CellSTICConfig


def _config(
    *,
    ccc_encoder: Optional[Sequence[int]] = None,
    ccc_decoder: Optional[Sequence[int]] = None,
    feature_encoders: Optional[Sequence[Sequence[int]]] = None,
    feature_decoders: Optional[Sequence[Sequence[int]]] = None,
    feature_output: Optional[Sequence[int]] = None,
    graph: Optional[Mapping[str, Any]] = None,
    n_clusters: Optional[int] = None,
    hierarchy_method: Optional[str] = None,
) -> CellSTICConfig:
    """Apply a dataset's minimal overrides to the canonical defaults."""
    base = CellSTICConfig()

    ccc_updates: Dict[str, Any] = {}
    if ccc_encoder is not None:
        ccc_updates["encoder_dims"] = tuple(ccc_encoder)
    if ccc_decoder is not None:
        ccc_updates["decoder_dims"] = tuple(ccc_decoder)
    ccc = replace(base.model.ccc, **ccc_updates)

    feature_updates: Dict[str, Any] = {}
    if feature_encoders is not None:
        feature_updates["encoder_dims"] = tuple(
            tuple(dims) for dims in feature_encoders
        )
    if feature_decoders is not None:
        feature_updates["decoder_dims"] = tuple(
            tuple(dims) for dims in feature_decoders
        )
    if feature_output is not None:
        feature_updates["output_dims"] = tuple(feature_output)
    feat = replace(base.model.feat, **feature_updates)

    graph_config = replace(base.model.graph, **dict(graph or {}))
    tree = (
        base.model.tree
        if hierarchy_method is None
        else replace(base.model.tree, hierarchy_method=hierarchy_method)
    )
    model = replace(base.model, ccc=ccc, feat=feat, graph=graph_config, tree=tree)

    feature_train = (
        base.train.feat
        if n_clusters is None
        else replace(base.train.feat, n_clusters=n_clusters)
    )
    train = replace(base.train, feat=feature_train)
    return replace(base, model=model, train=train)


_PRESETS: Dict[str, CellSTICConfig] = {
    # Canonical defaults: two 250-dimensional modalities, the default graph,
    # nine clusters, and the shared optimization schedule.
    "human_skin": _config(),
    "human_tonsil": _config(n_clusters=7),
    "human_lymph_node": _config(
        ccc_encoder=(500, 200),
        ccc_decoder=(600, 200),
        feature_encoders=((500, 200), (30, 20)),
        feature_decoders=((200, 500), (20, 30)),
        feature_output=(220, 100),
        graph=dict(cluster_top_k=10, cluster_size=30, n_spots=4),
        n_clusters=8,
        hierarchy_method="llm",
    ),
    "mouse_brain": _config(
        ccc_encoder=(131, 100, 50),
        ccc_decoder=(200, 50),
        feature_encoders=((131, 100, 50),) * 5,
        feature_decoders=((50, 131),) * 5,
        feature_output=(250, 50),
        graph=dict(n_spots=4),
    ),
    "nsf": _config(
        # The runner disables CCC, but compatible CCC dimensions keep the
        # complete configuration valid and reusable.
        ccc_encoder=(100, 70),
        ccc_decoder=(260, 220, 80),
        feature_encoders=((100, 85, 60),) * 3,
        feature_decoders=((60, 100),) * 3,
        feature_output=(180, 60),
        graph=dict(cluster_top_k=10, cluster_size=40, knn_top_k=10),
        n_clusters=5,
    ),
    "domain_sim": _config(
        ccc_encoder=(100, 70),
        ccc_decoder=(260, 220, 80),
        feature_encoders=((100, 85, 60),) * 3,
        feature_decoders=((60, 100),) * 3,
        feature_output=(180, 60),
        graph=dict(cluster_top_k=10, cluster_size=40, knn_top_k=10),
        n_clusters=5,
    ),
    "mouse_embryo": _config(
        ccc_encoder=(500, 300, 200),
        ccc_decoder=(600, 200),
        feature_encoders=((500, 300, 200),),
        feature_decoders=((200, 500),),
        feature_output=(200, 100),
        graph=dict(n_spots=4),
        n_clusters=7,
    ),
    "axolotl_telencephalon": _config(
        ccc_encoder=(500, 200),
        ccc_decoder=(600, 200),
        feature_encoders=((500, 200),),
        feature_decoders=((200, 500),),
        feature_output=(200, 100),
        graph=dict(cluster_top_k=10, cluster_size=30, n_spots=4),
        n_clusters=10,
    ),
    "scmultisim": _config(
        ccc_encoder=(120, 100, 80),
        ccc_decoder=(320, 150),
        feature_encoders=((120, 100, 80), (120, 100, 80)),
        feature_decoders=((80, 120), (80, 120)),
        feature_output=(160, 120, 80),
        graph=dict(cluster_top_k=10, cluster_size=20, n_spots=25),
        n_clusters=4,
    ),
}


def get_cellstic_config(dataset: str, *, tree_height: Optional[int] = None) -> CellSTICConfig:
    """Return the immutable preset shared by all stages/runs of ``dataset``."""
    key = str(dataset).strip().lower()
    try:
        base = _PRESETS[key]
        if tree_height is None:
            return base
        return replace(base, model=replace(
            base.model, tree=replace(base.model.tree, height=tree_height)
        ))
    except KeyError as exc:
        raise KeyError(
            f"Unknown CellSTIC dataset preset {dataset!r}; "
            f"available={sorted(_PRESETS)}"
        ) from exc


__all__ = ["get_cellstic_config"]
