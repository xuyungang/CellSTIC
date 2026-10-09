"""Load and split mouse-embryo MOSTA data in memory without h5ad caches."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence

import anndata as ad
import celltypist
import pandas as pd
import scanpy as sc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

ANNOTATION_KEY = "annotation"
LIGAND_RECEPTOR_PAIRS = (
    ("F2", "F2r"),
    ("Thbs4", "Cd36"),
    ("Plg", "Pard3"),
    ("Nts", "Sort1"),
    ("Lpar3", "Adgre5"),
)
CELLTYPIST_MODEL_BY_ORGAN = {
    "Brain": PROJECT_ROOT / "resources" / "celltypist" / "Developing_Mouse_Brain.pkl",
    "Liver": PROJECT_ROOT / "resources" / "celltypist" / "Healthy_Mouse_Liver.pkl",
}


def merge_cell_type_labels(labels: pd.Series) -> pd.Series:
    """Collapse hierarchical CellTypist labels to their lineage prefix.

    CellTypist may emit labels such as ``"Blood: Undefined"`` or
    ``"Neuroblast: ..."``.  The mouse-embryo analyses use the main lineage,
    so every label containing a colon is reduced to the text before it.
    """
    return labels.astype(str).str.split(":", n=1).str[0].str.strip()


def _load_lr_metadata() -> Dict:
    """Build mouse-embryo LR metadata without relying on a raw CSV file."""
    ligand_receptor_map = {
        ligand: [receptor] for ligand, receptor in LIGAND_RECEPTOR_PAIRS
    }
    lr_table = pd.DataFrame(
        [
            {"ligand": ligand, "receptor": receptor, "idx": index}
            for index, (ligand, receptor) in enumerate(LIGAND_RECEPTOR_PAIRS, start=1)
        ]
    )
    return {
        "dataset": "mouse_embryo",
        "ligand_receptor_map": ligand_receptor_map,
        "lr_table": lr_table,
    }


def _ensure_counts(adata: ad.AnnData) -> None:
    if "counts" in adata.layers:
        return
    if "count" in adata.layers:
        adata.layers["counts"] = adata.layers["count"].copy()
    else:
        adata.layers["counts"] = adata.X.copy()


def annotate_cell_types(adata: ad.AnnData, organ: str) -> None:
    """Annotate one organ with its CellTypist model and map subtype labels."""
    try:
        model_path = CELLTYPIST_MODEL_BY_ORGAN[organ]
    except KeyError as error:
        raise ValueError(
            f"No CellTypist model configured for organ {organ!r}"
        ) from error
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    annotation_input = adata.copy()
    annotation_input.X = annotation_input.layers["counts"].copy()
    sc.pp.normalize_total(annotation_input, target_sum=1e4)
    sc.pp.log1p(annotation_input)
    prediction = celltypist.annotate(
        annotation_input,
        model=str(model_path),
        majority_voting=False,
    )
    labels = prediction.predicted_labels
    if isinstance(labels, pd.DataFrame):
        labels = labels.iloc[:, 0]
    labels = labels.astype(str)
    labels = merge_cell_type_labels(labels)
    adata.obs["cell_type"] = pd.Categorical(labels.to_numpy())


def _discover_stages(dataset_dir: Path) -> List[str]:
    return sorted(
        (
            path.name.removeprefix("E").split("_", 1)[0]
            for path in (dataset_dir / "raw").glob("E*_E1S1.MOSTA.h5ad")
        ),
        key=float,
    )


def assemble_stage(
    dataset_dir: Path,
    stage: str,
    *,
    organs: Optional[Sequence[str]] = None,
) -> List[Path]:
    """Split one raw MOSTA stage into flattened stage/organ h5ad files."""
    dataset_dir = Path(dataset_dir)
    sources = sorted((dataset_dir / "raw").glob(f"E{stage}_*.MOSTA.h5ad"))
    if not sources:
        raise FileNotFoundError(f"No MOSTA h5ad for mouse-embryo stage {stage}")

    adata = ad.read_h5ad(sources[0])
    if ANNOTATION_KEY not in adata.obs:
        raise KeyError(f"{sources[0]}: missing obs[{ANNOTATION_KEY!r}]")
    if "spatial" not in adata.obsm:
        raise KeyError(f"{sources[0]}: missing obsm['spatial']")

    adata = adata.copy()
    adata.obs_names = pd.Index([f"{stage}:{name}" for name in adata.obs_names])
    adata.obs["stage"] = stage
    adata.obs["organ"] = adata.obs[ANNOTATION_KEY].astype(str)
    _ensure_counts(adata)

    available_organs = set(adata.obs["organ"].unique())
    selected_organs = list(organs) if organs is not None else sorted(available_organs)
    unknown = sorted(set(selected_organs) - available_organs)
    if unknown:
        raise ValueError(
            f"Stage {stage} has no organs {unknown}; available={sorted(available_organs)}"
        )

    written = []
    for organ in selected_organs:
        subset = adata[adata.obs["organ"] == organ].copy()
        subset.obs["stage"] = subset.obs["stage"].astype("category")
        subset.obs["organ"] = subset.obs["organ"].astype("category")
        annotate_cell_types(subset, str(organ))
        subset.uns.update({**_load_lr_metadata(), "stage": stage, "organ": organ})
        output = dataset_dir / "h5ad" / f"{stage}_{organ}.h5ad"
        output.parent.mkdir(parents=True, exist_ok=True)
        subset.write_h5ad(output)
        written.append(output)
        print(f"mouse_embryo/{stage}_{organ}: {subset.shape} -> {output}")
    return written


def assemble(
    dataset_dir: Path,
    *,
    stages: Optional[Sequence[str]] = None,
    organs: Optional[Sequence[str]] = None,
) -> List[Path]:
    dataset_dir = Path(dataset_dir)
    selected_stages = (
        list(stages) if stages is not None else _discover_stages(dataset_dir)
    )
    if not selected_stages:
        raise FileNotFoundError(
            f"No MOSTA h5ad files found under {dataset_dir / 'raw'}"
        )
    return [
        output
        for stage in selected_stages
        for output in assemble_stage(dataset_dir, str(stage), organs=organs)
    ]


def load_stage_organ(dataset_dir: Path, stage: str, organ: str) -> ad.AnnData:
    """Load one flattened mouse-embryo stage/organ h5ad input."""
    path = Path(dataset_dir) / "h5ad" / f"{stage}_{organ}.h5ad"
    if not path.is_file():
        raise FileNotFoundError(path)
    return ad.read_h5ad(path)


def repair_existing(dataset_dir: Path) -> None:
    """Backfill CellTypist labels into flattened h5ad inputs and CellSTIC outputs."""
    dataset_dir = Path(dataset_dir)
    for h5ad_path in sorted((dataset_dir / "h5ad").glob("*_*.h5ad")):
        stage, organ = h5ad_path.stem.rsplit("_", 1)
        if organ not in CELLTYPIST_MODEL_BY_ORGAN:
            continue
        adata = ad.read_h5ad(h5ad_path)
        _ensure_counts(adata)
        annotate_cell_types(adata, organ)
        adata.write_h5ad(h5ad_path)

        result_path = dataset_dir / "cellstic" / f"{stage}_{organ}" / "out" / "out.h5ad"
        if result_path.is_file():
            result = ad.read_h5ad(result_path)
            labels = adata.obs["cell_type"].reindex(result.obs_names)
            if labels.isna().any():
                raise ValueError(f"Cell IDs do not align: {h5ad_path} -> {result_path}")
            result.obs["cell_type"] = pd.Categorical(labels.astype(str).to_numpy())
            result.write_h5ad(result_path)
        print(f"annotated {h5ad_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path, default=PROJECT_ROOT / "paper" / "data" / "mouse_embryo"
    )
    parser.add_argument("--stages", nargs="+", default=None)
    parser.add_argument("--organs", nargs="+", default=None)
    parser.add_argument("--repair-existing", action="store_true")
    args = parser.parse_args()
    if args.repair_existing:
        repair_existing(args.dataset_dir)
    else:
        assemble(args.dataset_dir, stages=args.stages, organs=args.organs)


if __name__ == "__main__":
    main()
