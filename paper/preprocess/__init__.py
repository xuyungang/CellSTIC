"""Shared I/O helpers for dataset-specific h5ad assembly scripts.

Only raw-source assembly belongs here. Normalization, feature extraction, and
graph construction are performed in memory by :mod:`pipeline.cellstic`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import anndata as ad
import numpy as np
from anndata import AnnData


PathLike = Union[str, Path]


def read_modalities(
    h5ad_dir: PathLike,
    modality_names: Sequence[str],
) -> List[AnnData]:
    """Load assembled ``<modality>.h5ad`` files in the requested order."""
    h5ad_dir = Path(h5ad_dir)
    paths = [h5ad_dir / f"{name.lower()}.h5ad" for name in modality_names]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing assembled h5ad input(s): {missing_text}. Run the matching "
            "paper/preprocessing/<dataset>.py script first."
        )
    return [ad.read_h5ad(path) for path in paths]


def _metadata_dict(adata: AnnData, key: str) -> Dict[str, Any]:
    value = adata.uns.get(key, {})
    if not isinstance(value, dict):
        return {}
    return {
        str(name): item.tolist() if isinstance(item, np.ndarray) else item
        for name, item in value.items()
    }


def cellstic_metadata(
    adata: AnnData,
) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    """Read the ligand-receptor and edge-type maps."""
    ligand_receptor_map = {
        ligand: [str(receptor) for receptor in receptors]
        for ligand, receptors in _metadata_dict(adata, "ligand_receptor_map").items()
    }
    edge_type_map = {
        name: int(index)
        for name, index in _metadata_dict(adata, "edge_type_map").items()
    }
    return ligand_receptor_map, edge_type_map


def cellstic_pair_type_constraints(
    adata: AnnData,
) -> Dict[str, List[Tuple[int, int]]]:
    """Read optional dataset-specific LR sender/receiver type constraints."""
    return {
        name: [(int(sender), int(receiver)) for sender, receiver in pairs]
        for name, pairs in _metadata_dict(adata, "pair_type_constraints").items()
    }


__all__ = [
    "cellstic_metadata",
    "cellstic_pair_type_constraints",
    "read_modalities",
]
