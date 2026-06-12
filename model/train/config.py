"""Configuration dataclasses for CellSTIC."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Mapping, Type, TypeVar

T = TypeVar("T")


@dataclass
class CellSTICCCCConfig:
    """CCC precision head (EGNN encoder + structure decoder)."""

    encoder_dims: list[int] = field(default_factory=list)
    decoder_dims: list[int] = field(default_factory=list)
    temperature: float = 0.4


@dataclass
class CellSTICFeatConfig:
    """Multi-modality feature integration (per-modality GNN + fusion MLP)."""

    encoder_dims: list[list[int]] = field(default_factory=list)
    decoder_dims: list[list[int]] = field(default_factory=list)
    output_dims: list[int] = field(default_factory=list)


@dataclass
class CellSTICGraphConfig:
    """Spatial neighbourhood graph and CCC candidate construction."""

    cluster_top_k: int = 20
    cluster_size: int = 60
    knn_top_k: int = 20
    expression_percentile: float = 75
    n_spots: int = 10


@dataclass
class CellSTICTreeConfig:
    """Ligand–receptor hierarchy construction."""

    hierarchy_method: str = "balanced"


@dataclass
class CellSTICModelConfig:
    """Architecture hyper-parameters passed to `CellSTIC`."""

    dropout: float = 0.1
    ccc: CellSTICCCCConfig = field(default_factory=CellSTICCCCConfig)
    feat: CellSTICFeatConfig = field(default_factory=CellSTICFeatConfig)
    graph: CellSTICGraphConfig = field(default_factory=CellSTICGraphConfig)
    tree: CellSTICTreeConfig = field(default_factory=CellSTICTreeConfig)


@dataclass
class CellSTICCCCTrainConfig:
    """CCC precision training loop."""

    epochs: int = 300
    learning_rate: float = 0.004
    weight_decay: float = 0.0
    edge_type_loss_weight: float = 1.0
    sampling_rate: float = 0.3


@dataclass
class CellSTICFeatTrainConfig:
    """Feature pretraining loop."""

    epochs: int = 300
    learning_rate: float = 0.001
    n_clusters: int = 9
    weight_decay: float = 0.0
    weight_modality: float = 0.5
    entropy_weight: float = 0.8


@dataclass
class CellSTICTrainConfig:
    """Training bundle passed to `CellSTICTrainer`."""

    ccc: CellSTICCCCTrainConfig = field(default_factory=CellSTICCCCTrainConfig)
    feat: CellSTICFeatTrainConfig = field(default_factory=CellSTICFeatTrainConfig)


@dataclass
class CellSTICConfig:
    """Top-level bundle passed to Trainer / Evaluator."""

    model: CellSTICModelConfig = field(default_factory=CellSTICModelConfig)
    train: CellSTICTrainConfig = field(default_factory=CellSTICTrainConfig)


_NESTED: dict[type, dict[str, type]] = {
    CellSTICConfig: {"model": CellSTICModelConfig, "train": CellSTICTrainConfig},
    CellSTICModelConfig: {
        "ccc": CellSTICCCCConfig,
        "feat": CellSTICFeatConfig,
        "graph": CellSTICGraphConfig,
        "tree": CellSTICTreeConfig,
    },
    CellSTICTrainConfig: {"ccc": CellSTICCCCTrainConfig, "feat": CellSTICFeatTrainConfig},
}


def _coerce_dataclass(cls: Type[T], data: Any) -> T:
    if isinstance(data, cls):
        return data
    if data is None:
        return cls()
    if not isinstance(data, Mapping):
        raise TypeError(f"Expected mapping for {cls.__name__}, got {type(data)!r}")

    nested = _NESTED.get(cls, {})
    kwargs: dict[str, Any] = {}
    valid_names = {f.name for f in fields(cls)}
    for key, value in data.items():
        if key not in valid_names:
            continue
        if key in nested and isinstance(value, Mapping):
            kwargs[key] = _coerce_dataclass(nested[key], value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def config_from_mapping(data: Mapping[str, Any]) -> CellSTICConfig:
    """Build a `CellSTICConfig` from a nested mapping."""
    return _coerce_dataclass(CellSTICConfig, data)


def apply_config_overrides(
    config: CellSTICConfig,
    overrides: Mapping[str, Any],
) -> CellSTICConfig:
    """Return a new config with nested dict overrides applied."""
    merged = _deep_update(asdict(config), overrides)
    return config_from_mapping(merged)


def _deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, Mapping):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result
