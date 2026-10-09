"""Write the two CellSTIC training checkpoints.

Checkpoints are terminal run artifacts only. CellSTIC intentionally exposes no
checkpoint-loading or validation-only workflow.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any, Mapping, Optional, Union

import torch


PathLike = Union[str, Path]
FEATURE_CHECKPOINT_NAME = "feature.pt"
CCC_CHECKPOINT_NAME = "ccc.pt"
CHECKPOINT_VERSION = 3
CHECKPOINT_FORMAT = "cellstic-state-dict"


def _state_dict_cpu(module: torch.nn.Module) -> Mapping[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in module.state_dict().items()}


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    """Write a checkpoint completely before replacing its final path."""
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}-",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        torch.save(dict(payload), temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_feature_checkpoint(
    module: torch.nn.Module,
    checkpoint_dir: PathLike,
    *,
    config: Mapping[str, Any],
) -> Path:
    """Save the trained multimodal feature integrator."""
    path = Path(checkpoint_dir) / FEATURE_CHECKPOINT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(
        {
            "version": CHECKPOINT_VERSION,
            "format": CHECKPOINT_FORMAT,
            "model": "CellSTIC",
            "component": "feature",
            "weight_origin": "pipeline_run",
            "state_dict": _state_dict_cpu(module),
            "config": dict(config),
        },
        path,
    )
    return path


def save_ccc_checkpoint(
    module: torch.nn.Module,
    checkpoint_dir: PathLike,
    *,
    config: Mapping[str, Any],
    trained: bool,
    edge_type_map: Optional[Mapping[str, int]] = None,
    hierarchy_dict: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Save the CCC predictor and the hierarchy metadata produced by the run."""
    path = Path(checkpoint_dir) / CCC_CHECKPOINT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(
        {
            "version": CHECKPOINT_VERSION,
            "format": CHECKPOINT_FORMAT,
            "model": "CellSTIC",
            "component": "ccc",
            "weight_origin": "pipeline_run",
            "trained": bool(trained),
            "state_dict": _state_dict_cpu(module),
            "config": dict(config),
            "edge_type_map": dict(edge_type_map or {}),
            "hierarchy_dict": dict(hierarchy_dict or {}),
        },
        path,
    )
    return path


__all__ = [
    "CHECKPOINT_FORMAT",
    "CCC_CHECKPOINT_NAME",
    "FEATURE_CHECKPOINT_NAME",
    "save_ccc_checkpoint",
    "save_feature_checkpoint",
]
