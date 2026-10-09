"""Assemble per-stage axolotl h5ad files without model preprocessing.

The source ``Development.h5ad`` and ``Regeneration.h5ad`` files are split by
``obs['Batch']``.  Each output contains counts, spatial coordinates, annotations,
and LR metadata only; feature extraction is performed later by the pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict

import anndata as ad
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

BATCH_TO_DIR: Dict[str, str] = {
    "Batch1_Adult_telencephalon_rep2_DP8400015234BLA3_1": "dev_Adult",
    "Meta_telencephalon_rep1_DP8400015234BLB2_1": "dev_meta",
    "Stage44_telencephalon_rep2_FP200000239BL_E4": "dev_44",
    "Stage54_telencephalon_rep2_DP8400015649BRD6_2": "dev_54",
    "Stage57_telencephalon_rep2_DP8400015649BRD5_1": "dev_57",
    "Injury_2DPI_rep1_SS200000147BL_D5": "reg_2",
    "Injury_5DPI_rep1_SS200000147BL_D2": "reg_5",
    "Injury_10DPI_rep1_SS200000147BL_B5": "reg_10",
    "Injury_15DPI_rep4_FP200000266TR_E4": "reg_15",
    "Injury_20DPI_rep2_SS200000147BL_B4": "reg_20",
    "Injury_30DPI_rep2_FP200000264BL_A6": "reg_30",
    "Injury_60DPI_rep3_FP200000264BL_A6": "reg_60",
    "Injury_control_FP200000239BL_E3": "reg_control",
}
LIGAND_RECEPTOR_PAIRS = (
    ("WNT7B", "FZD5"),
    ("WNT7A", "FZD5"),
    ("BMP7", "BMPR1A"),
    ("FGF8", "FGFR1"),
)


def _normalize_gene_names(adata: ad.AnnData) -> None:
    names = pd.Index(adata.var_names.astype(str))
    if names.str.contains(r"\|").any():
        adata.var_names = names.str.split("|", n=1).str[0].str.strip()
    adata.var_names_make_unique()


def assemble(root: Path) -> list[Path]:
    root = Path(root)
    sources = sorted((root / "raw").glob("*.h5ad"))
    if not sources:
        raise FileNotFoundError(f"No source h5ad files under {root / 'raw'}")

    lr_map: Dict[str, list[str]] = {}
    for ligand, receptor in LIGAND_RECEPTOR_PAIRS:
        lr_map.setdefault(ligand, []).append(receptor)
    lr_table = pd.DataFrame(
        [
            {"ligand": ligand, "receptor": receptor, "idx": index}
            for index, (ligand, receptor) in enumerate(LIGAND_RECEPTOR_PAIRS, start=1)
        ]
    )

    written: list[Path] = []
    for source in sources:
        adata = ad.read_h5ad(source)
        if "Batch" not in adata.obs:
            raise KeyError(f"{source}: missing obs['Batch']")
        if "spatial" not in adata.obsm:
            if {"x", "y"}.issubset(adata.obs.columns):
                adata.obsm["spatial"] = adata.obs[["x", "y"]].to_numpy()
            else:
                raise KeyError(f"{source}: missing obsm['spatial']")
        _normalize_gene_names(adata)

        for batch in adata.obs["Batch"].astype(str).unique():
            if batch not in BATCH_TO_DIR:
                raise KeyError(f"{source}: no output mapping for batch {batch!r}")
            sample_name = BATCH_TO_DIR[batch]
            sample = adata[adata.obs["Batch"].astype(str) == batch].copy()
            if "counts" not in sample.layers:
                sample.layers["counts"] = sample.X.copy()
            sample.uns.update(
                {
                    "dataset": "axolotl_telencephalon",
                    "lineage": sample_name.split("_", 1)[0],
                    "stage": sample_name,
                    "region": "telencephalon",
                    "ligand_receptor_map": lr_map,
                }
            )
            sample.uns["lr_table"] = lr_table
            output = root / "h5ad" / f"{sample_name}.h5ad"
            output.parent.mkdir(parents=True, exist_ok=True)
            sample.write_h5ad(output)
            written.append(output)
            print(f"{sample_name}: {sample.shape} -> {output}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "paper" / "data" / "axolotl_tele",
    )
    args = parser.parse_args()
    assemble(args.root)


if __name__ == "__main__":
    main()
