"""Assemble human-skin RNA and ADT matrices into unprocessed h5ad inputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import anndata as ad
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    ("rna", "raw/GSM6578065_humanskin_RNA.tsv"),
    ("adt", "raw/GSM6578074_humanskin_protein.tsv"),
)


def _spatial_coordinates(obs_names: pd.Index) -> np.ndarray:
    try:
        return np.asarray(
            [
                tuple(float(value) for value in str(name).replace(" ", "").split("x"))
                for name in obs_names
            ],
            dtype=np.float32,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Cannot derive x/y coordinates from human-skin cell IDs"
        ) from exc


def assemble(dataset_dir: Path) -> Tuple[ad.AnnData, ad.AnnData]:
    """Read raw TSV matrices, align cells, and write ``h5ad/{rna,adt}.h5ad``."""
    dataset_dir = Path(dataset_dir)
    modalities = []
    for _, relative_path in SOURCES:
        source = dataset_dir / relative_path
        if not source.is_file():
            raise FileNotFoundError(source)
        frame = pd.read_csv(source, sep="\t", index_col=0).astype(np.float32)
        modalities.append(ad.AnnData(frame))

    common_cells = modalities[0].obs_names.intersection(
        modalities[1].obs_names, sort=False
    )
    if len(common_cells) == 0:
        raise ValueError("Human-skin RNA and ADT do not share cell identifiers")

    coordinates = _spatial_coordinates(common_cells)
    output_dir = dataset_dir / "h5ad"
    output_dir.mkdir(parents=True, exist_ok=True)
    assembled = []
    for (name, _), adata in zip(SOURCES, modalities):
        result = adata[common_cells].copy()
        result.layers["counts"] = result.X.copy()
        result.obsm["spatial"] = coordinates.copy()
        result.uns.update(
            {
                "dataset": "human_skin",
            }
        )
        result.write_h5ad(output_dir / f"{name}.h5ad")
        assembled.append(result)

    print(
        f"human_skin: RNA {assembled[0].shape}, ADT {assembled[1].shape} -> {output_dir}"
    )
    return assembled[0], assembled[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "paper" / "data" / "human_skin",
    )
    args = parser.parse_args()
    assemble(args.dataset_dir)


if __name__ == "__main__":
    main()
