"""Typed and validated configuration for CellSTIC."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Optional, Tuple


Dims = Tuple[int, ...]
ModalityDims = Tuple[Dims, ...]
_HIERARCHY_METHODS = frozenset({"balanced", "biological", "llm"})


def _positive_int(name: str, value: int, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _finite_float(
    name: str,
    value: float,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if minimum is not None:
        invalid = value < minimum if minimum_inclusive else value <= minimum
        if invalid:
            operator = ">=" if minimum_inclusive else ">"
            raise ValueError(f"{name} must be {operator} {minimum}, got {value!r}")
    if maximum is not None:
        invalid = value > maximum if maximum_inclusive else value >= maximum
        if invalid:
            operator = "<=" if maximum_inclusive else "<"
            raise ValueError(f"{name} must be {operator} {maximum}, got {value!r}")
    return float(value)


def _dims(name: str, values: Dims) -> Dims:
    try:
        normalized = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of dimensions") from exc
    if len(normalized) < 2:
        raise ValueError(f"{name} must describe at least one layer (two dimensions)")
    return tuple(
        _positive_int(f"{name}[{index}]", value)
        for index, value in enumerate(normalized)
    )


def _modality_dims(name: str, values: ModalityDims) -> ModalityDims:
    try:
        normalized = tuple(
            _dims(f"{name}[{index}]", dims) for index, dims in enumerate(values)
        )
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of modality dimensions") from exc
    if not normalized:
        raise ValueError(f"{name} must contain at least one modality")
    return normalized


@dataclass(frozen=True)
class CellSTICCCCModelConfig:
    """CCC graph encoder and pair decoder architecture."""

    encoder_dims: Dims = (250, 150, 100)
    decoder_dims: Dims = (400, 200)
    temperature: float = 0.4
    edge_prob_batch_size: int = 10_000

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "encoder_dims", _dims("ccc.encoder_dims", self.encoder_dims)
        )
        object.__setattr__(
            self, "decoder_dims", _dims("ccc.decoder_dims", self.decoder_dims)
        )
        object.__setattr__(
            self,
            "temperature",
            _finite_float(
                "ccc.temperature",
                self.temperature,
                minimum=0.0,
                minimum_inclusive=False,
            ),
        )
        object.__setattr__(
            self,
            "edge_prob_batch_size",
            _positive_int("ccc.edge_prob_batch_size", self.edge_prob_batch_size),
        )


@dataclass(frozen=True)
class CellSTICFeatureModelConfig:
    """Per-modality encoders/decoders and fused representation architecture."""

    encoder_dims: ModalityDims = (
        (250, 200, 120),
        (250, 200, 120),
    )
    decoder_dims: ModalityDims = (
        (120, 250),
        (120, 250),
    )
    output_dims: Dims = (240, 100)

    def __post_init__(self) -> None:
        encoders = _modality_dims("feat.encoder_dims", self.encoder_dims)
        decoders = _modality_dims("feat.decoder_dims", self.decoder_dims)
        output = _dims("feat.output_dims", self.output_dims)
        object.__setattr__(self, "encoder_dims", encoders)
        object.__setattr__(self, "decoder_dims", decoders)
        object.__setattr__(self, "output_dims", output)

        if len(encoders) != len(decoders):
            raise ValueError(
                "feat.encoder_dims and feat.decoder_dims must contain the same "
                f"number of modalities, got {len(encoders)} and {len(decoders)}"
            )
        for index, (encoder, decoder) in enumerate(zip(encoders, decoders)):
            if decoder[0] != encoder[-1]:
                raise ValueError(
                    f"feat.decoder_dims[{index}][0] must equal the encoded "
                    f"dimension {encoder[-1]}, got {decoder[0]}"
                )
            if decoder[-1] != encoder[0]:
                raise ValueError(
                    f"feat.decoder_dims[{index}][-1] must reconstruct all "
                    f"{encoder[0]} input features, got {decoder[-1]}"
                )

        fused_input = sum(encoder[-1] for encoder in encoders)
        if output[0] != fused_input:
            raise ValueError(
                "feat.output_dims[0] must equal the sum of modality latent "
                f"dimensions ({fused_input}), got {output[0]}"
            )


@dataclass(frozen=True)
class CellSTICGraphConfig:
    """Sparse cluster, feature-neighbour, spatial, and LR graph construction.

    ``max_spatial_neighbors`` is a hard per-cell candidate cap; together with
    the spatial radius derived from ``n_spots``, it bounds graph memory by
    ``O(n_cells * max_spatial_neighbors)``.
    """

    cluster_top_k: int = 20
    cluster_size: int = 60
    knn_top_k: int = 20
    expression_percentile: float = 75.0
    n_spots: int = 10
    max_spatial_neighbors: int = 64

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cluster_top_k",
            _positive_int("graph.cluster_top_k", self.cluster_top_k),
        )
        object.__setattr__(
            self,
            "cluster_size",
            _positive_int("graph.cluster_size", self.cluster_size, minimum=2),
        )
        object.__setattr__(
            self,
            "knn_top_k",
            _positive_int("graph.knn_top_k", self.knn_top_k),
        )
        object.__setattr__(
            self,
            "expression_percentile",
            _finite_float(
                "graph.expression_percentile",
                self.expression_percentile,
                minimum=0.0,
                maximum=100.0,
            ),
        )
        object.__setattr__(
            self, "n_spots", _positive_int("graph.n_spots", self.n_spots)
        )
        object.__setattr__(
            self,
            "max_spatial_neighbors",
            _positive_int("graph.max_spatial_neighbors", self.max_spatial_neighbors),
        )
        required_neighbors = max(self.knn_top_k, self.cluster_top_k)
        if self.max_spatial_neighbors < required_neighbors:
            raise ValueError(
                "graph.max_spatial_neighbors must be >= both graph.knn_top_k "
                "and graph.cluster_top_k, got "
                f"{self.max_spatial_neighbors}, {self.knn_top_k}, and "
                f"{self.cluster_top_k}"
            )


@dataclass(frozen=True)
class CellSTICTreeConfig:
    """Ligand-receptor hierarchy construction."""

    hierarchy_method: str = "balanced"
    height: Optional[int] = None

    def __post_init__(self) -> None:
        if self.hierarchy_method not in _HIERARCHY_METHODS:
            raise ValueError(
                f"tree.hierarchy_method must be one of "
                f"{sorted(_HIERARCHY_METHODS)}, got {self.hierarchy_method!r}"
            )
        if self.height is not None:
            object.__setattr__(
                self, "height", _positive_int("tree.height", self.height, minimum=2)
            )


@dataclass(frozen=True)
class CellSTICModelConfig:
    """Complete CellSTIC model architecture."""

    dropout: float = 0.1
    ccc: CellSTICCCCModelConfig = field(default_factory=CellSTICCCCModelConfig)
    feat: CellSTICFeatureModelConfig = field(default_factory=CellSTICFeatureModelConfig)
    graph: CellSTICGraphConfig = field(default_factory=CellSTICGraphConfig)
    tree: CellSTICTreeConfig = field(default_factory=CellSTICTreeConfig)

    def __post_init__(self) -> None:
        nested = (
            ("model.ccc", self.ccc, CellSTICCCCModelConfig),
            ("model.feat", self.feat, CellSTICFeatureModelConfig),
            ("model.graph", self.graph, CellSTICGraphConfig),
            ("model.tree", self.tree, CellSTICTreeConfig),
        )
        for name, value, expected_type in nested:
            if not isinstance(value, expected_type):
                raise ValueError(
                    f"{name} must be {expected_type.__name__}, got "
                    f"{type(value).__name__}"
                )
        object.__setattr__(
            self,
            "dropout",
            _finite_float(
                "model.dropout",
                self.dropout,
                minimum=0.0,
                maximum=1.0,
                maximum_inclusive=False,
            ),
        )

        primary_input = self.feat.encoder_dims[0][0]
        if self.ccc.encoder_dims[0] != primary_input:
            raise ValueError(
                "ccc.encoder_dims[0] must equal the first modality input "
                f"dimension ({primary_input}), got {self.ccc.encoder_dims[0]}"
            )

        pair_input = 2 * (self.feat.output_dims[-1] + self.ccc.encoder_dims[-1])
        if self.ccc.decoder_dims[0] != pair_input:
            raise ValueError(
                "ccc.decoder_dims[0] must equal twice the concatenated feature "
                f"and CCC latent dimensions ({pair_input}), got "
                f"{self.ccc.decoder_dims[0]}"
            )


@dataclass(frozen=True)
class CellSTICCCCTrainConfig:
    """CCC optimization parameters."""

    epochs: int = 300
    learning_rate: float = 0.004
    weight_decay: float = 0.0
    edge_type_loss_weight: float = 1.0
    sampling_rate: float = 0.3
    min_masked_edges_per_channel: int = 1
    max_masked_edges_per_channel: int = 256
    graph_batch_size: int = 1024

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "epochs", _positive_int("train.ccc.epochs", self.epochs)
        )
        object.__setattr__(
            self,
            "learning_rate",
            _finite_float(
                "train.ccc.learning_rate",
                self.learning_rate,
                minimum=0.0,
                minimum_inclusive=False,
            ),
        )
        object.__setattr__(
            self,
            "weight_decay",
            _finite_float("train.ccc.weight_decay", self.weight_decay, minimum=0.0),
        )
        object.__setattr__(
            self,
            "edge_type_loss_weight",
            _finite_float(
                "train.ccc.edge_type_loss_weight",
                self.edge_type_loss_weight,
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "sampling_rate",
            _finite_float(
                "train.ccc.sampling_rate",
                self.sampling_rate,
                minimum=0.0,
                maximum=1.0,
                minimum_inclusive=False,
            ),
        )
        min_masked = _positive_int(
            "train.ccc.min_masked_edges_per_channel",
            self.min_masked_edges_per_channel,
        )
        max_masked = _positive_int(
            "train.ccc.max_masked_edges_per_channel",
            self.max_masked_edges_per_channel,
        )
        if max_masked < min_masked:
            raise ValueError(
                "train.ccc.max_masked_edges_per_channel must be >= "
                "train.ccc.min_masked_edges_per_channel, got "
                f"{max_masked} and {min_masked}"
            )
        object.__setattr__(self, "min_masked_edges_per_channel", min_masked)
        object.__setattr__(self, "max_masked_edges_per_channel", max_masked)
        graph_batch_size = _positive_int(
            "train.ccc.graph_batch_size", self.graph_batch_size
        )
        if graph_batch_size < 2:
            raise ValueError(
                "train.ccc.graph_batch_size must be >= 2 so every graph batch "
                "can contain a matched positive/negative pair"
            )
        object.__setattr__(self, "graph_batch_size", graph_batch_size)


@dataclass(frozen=True)
class CellSTICFeatureTrainConfig:
    """Multimodal feature-integration optimization parameters."""

    epochs: int = 300
    learning_rate: float = 0.001
    n_clusters: int = 9
    weight_decay: float = 0.0
    reconstruction_loss_weight: float = 0.5
    entropy_weight: float = 0.8
    clustering_temperature: float = 0.1
    clustering_iterations: int = 8

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "epochs", _positive_int("train.feat.epochs", self.epochs)
        )
        object.__setattr__(
            self,
            "learning_rate",
            _finite_float(
                "train.feat.learning_rate",
                self.learning_rate,
                minimum=0.0,
                minimum_inclusive=False,
            ),
        )
        object.__setattr__(
            self,
            "n_clusters",
            _positive_int("train.feat.n_clusters", self.n_clusters, minimum=2),
        )
        object.__setattr__(
            self,
            "weight_decay",
            _finite_float("train.feat.weight_decay", self.weight_decay, minimum=0.0),
        )
        object.__setattr__(
            self,
            "reconstruction_loss_weight",
            _finite_float(
                "train.feat.reconstruction_loss_weight",
                self.reconstruction_loss_weight,
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "entropy_weight",
            _finite_float(
                "train.feat.entropy_weight", self.entropy_weight, minimum=0.0
            ),
        )
        object.__setattr__(
            self,
            "clustering_temperature",
            _finite_float(
                "train.feat.clustering_temperature",
                self.clustering_temperature,
                minimum=0.0,
                minimum_inclusive=False,
            ),
        )
        object.__setattr__(
            self,
            "clustering_iterations",
            _positive_int(
                "train.feat.clustering_iterations", self.clustering_iterations
            ),
        )


@dataclass(frozen=True)
class CellSTICCCCPostprocessConfig:
    """Descriptive null calibration and MC-dropout robustness for leaf edges."""

    null_tail_fraction: float = 0.05
    min_null_edges: int = 20
    stability_repeats: int = 0
    stability_threshold: float = 0.8

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "null_tail_fraction",
            _finite_float(
                "postprocess.null_tail_fraction",
                self.null_tail_fraction,
                minimum=0.0,
                maximum=1.0,
                minimum_inclusive=False,
                maximum_inclusive=False,
            ),
        )
        object.__setattr__(
            self,
            "min_null_edges",
            _positive_int("postprocess.min_null_edges", self.min_null_edges),
        )
        stability_repeats = _positive_int(
            "postprocess.stability_repeats",
            self.stability_repeats,
            minimum=0,
        )
        if stability_repeats == 1:
            raise ValueError(
                "postprocess.stability_repeats must be 0 (disabled) or >= 2"
            )
        object.__setattr__(self, "stability_repeats", stability_repeats)
        object.__setattr__(
            self,
            "stability_threshold",
            _finite_float(
                "postprocess.stability_threshold",
                self.stability_threshold,
                minimum=0.0,
                maximum=1.0,
                minimum_inclusive=False,
            ),
        )


@dataclass(frozen=True)
class CellSTICTrainConfig:
    """Both trainable stages."""

    ccc: CellSTICCCCTrainConfig = field(default_factory=CellSTICCCCTrainConfig)
    feat: CellSTICFeatureTrainConfig = field(default_factory=CellSTICFeatureTrainConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.ccc, CellSTICCCCTrainConfig):
            raise ValueError(
                "train.ccc must be CellSTICCCCTrainConfig, got "
                f"{type(self.ccc).__name__}"
            )
        if not isinstance(self.feat, CellSTICFeatureTrainConfig):
            raise ValueError(
                "train.feat must be CellSTICFeatureTrainConfig, got "
                f"{type(self.feat).__name__}"
            )


@dataclass(frozen=True)
class CellSTICConfig:
    """Top-level configuration bundle passed to the end-to-end pipeline."""

    model: CellSTICModelConfig = field(default_factory=CellSTICModelConfig)
    train: CellSTICTrainConfig = field(default_factory=CellSTICTrainConfig)
    postprocess: CellSTICCCCPostprocessConfig = field(
        default_factory=CellSTICCCCPostprocessConfig
    )
    seed: int = 42
    deterministic: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.model, CellSTICModelConfig):
            raise ValueError(
                f"model must be CellSTICModelConfig, got {type(self.model).__name__}"
            )
        if not isinstance(self.train, CellSTICTrainConfig):
            raise ValueError(
                f"train must be CellSTICTrainConfig, got {type(self.train).__name__}"
            )
        if not isinstance(self.postprocess, CellSTICCCCPostprocessConfig):
            raise ValueError(
                "postprocess must be CellSTICCCCPostprocessConfig, got "
                f"{type(self.postprocess).__name__}"
            )
        object.__setattr__(self, "seed", _positive_int("seed", self.seed, minimum=0))
        if not isinstance(self.deterministic, bool):
            raise ValueError(
                f"deterministic must be a bool, got {self.deterministic!r}"
            )


__all__ = [
    "CellSTICCCCModelConfig",
    "CellSTICCCCPostprocessConfig",
    "CellSTICCCCTrainConfig",
    "CellSTICConfig",
    "CellSTICFeatureModelConfig",
    "CellSTICFeatureTrainConfig",
    "CellSTICGraphConfig",
    "CellSTICModelConfig",
    "CellSTICTrainConfig",
    "CellSTICTreeConfig",
]
