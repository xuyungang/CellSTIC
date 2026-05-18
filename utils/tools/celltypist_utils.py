"""Cell type annotation with CellTypist (https://github.com/Teichlab/celltypist)."""

from pathlib import Path
from typing import Optional, Literal

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sparse
import anndata as ad
import celltypist

_ROOT = Path(__file__).resolve().parents[2]
_CELLTYPIST_DIR = _ROOT / "component" / "celltypist"

SPECIES_MODEL_MAP = {
    "human_skin": str(_CELLTYPIST_DIR / "Adult_Human_Skin.pkl"),
    "human_tonsil": str(_CELLTYPIST_DIR / "Cells_Human_Tonsil.pkl"),
    "human_lymph_node": str(_CELLTYPIST_DIR / "Immune_All_Low.pkl"),
    "mouse_brain": str(_CELLTYPIST_DIR / "Developing_Mouse_Brain.pkl"),
    "mouse_liver": str(_CELLTYPIST_DIR / "Healthy_Mouse_Liver.pkl"),
}


def _clean_X_inplace(a: ad.AnnData, cap: float = 1e6) -> float:
    """Replace inf/nan and cap in a.X; return max value."""
    if sparse.issparse(a.X):
        d = a.X.data
        d[np.isinf(d) | np.isnan(d)] = 0.0
        d[d > cap] = 0.0
        a.X.eliminate_zeros()
        return float(d.max()) if d.size > 0 else 0.0
    x = np.asarray(a.X, dtype=float)
    x[np.isinf(x) | np.isnan(x)] = 0.0
    x[x > cap] = 0.0
    a.X = x
    return float(np.max(x))


def _prepare_for_celltypist(adata: ad.AnnData) -> ad.AnnData:
    """Copy raw or X, clean, optionally unlog, normalize 1e4 + log1p, set dense a.X."""
    a = adata.raw[adata.obs_names].to_adata() if adata.raw is not None else adata.copy()
    if adata.raw is None:
        print("Warning: adata.raw not set, using adata.X.")
    max_val = _clean_X_inplace(a)
    if max_val < 20 or "log1p" in a.uns:
        if sparse.issparse(a.X):
            a.X.data = np.expm1(np.clip(a.X.data, None, 20))
        else:
            a.X = np.expm1(np.clip(np.asarray(a.X), None, 20))
        _clean_X_inplace(a)
    sc.pp.normalize_total(a, target_sum=1e4, inplace=True)
    sc.pp.log1p(a)
    a.uns["log1p"] = {}
    x = a.X.toarray() if sparse.issparse(a.X) else np.array(a.X, copy=True)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x[x < 0] = 0.0
    a.X = x
    return a


class CellTypistAnnotator:
    """Cell type annotation via CellTypist (model by path or species key)."""

    def __init__(
        self,
        model: Optional[str] = None,
        species: Optional[str] = None,
        mode: Literal["best_match", "prob_match"] = "best_match",
    ):
        if species is not None:
            if species not in SPECIES_MODEL_MAP:
                raise ValueError(f"Unknown species: {species}. Available: {list(SPECIES_MODEL_MAP.keys())}")
            self.model = SPECIES_MODEL_MAP[species]
        else:
            self.model = model or "Immune_All_Low.pkl"
        self.mode = "best match" if mode == "best_match" else "prob match"

    def annotate(
        self,
        adata: ad.AnnData,
        majority_voting: bool = False,
        min_prop: float = 0.0,
        p_thres: float = 0.5,
        output_path: Optional[str] = None,
    ) -> ad.AnnData:
        """Run CellTypist; write predicted_labels to adata.obs, optionally save CSV."""
        print(f"Annotating {adata.n_obs} cells with {adata.n_vars} genes using model: {self.model}")
        a = _prepare_for_celltypist(adata)
        pred = celltypist.annotate(
            a, model=self.model, mode=self.mode,
            majority_voting=majority_voting, min_prop=min_prop, p_thres=p_thres,
        )
        adata.obs["predicted_labels"] = pred.predicted_labels
        n_types = adata.obs["predicted_labels"].nunique()
        print(f"Annotation done. {n_types} unique cell types.")
        if output_path:
            pd.DataFrame({"cell_id": adata.obs_names, "predicted_labels": adata.obs["predicted_labels"]}).to_csv(
                output_path, index=False
            )
            print(f"Saved: {Path(output_path).resolve()}")
        return adata


def annotate_with_celltypist(
    adata: ad.AnnData,
    species: Optional[str] = None,
    model: Optional[str] = None,
    mode: Literal["best_match", "prob_match"] = "best_match",
    majority_voting: bool = False,
    min_prop: float = 0.0,
    p_thres: float = 0.5,
    output_path: Optional[str] = None,
) -> ad.AnnData:
    """Convenience: build CellTypistAnnotator and run annotate."""
    ann = CellTypistAnnotator(model=model, species=species, mode=mode)
    return ann.annotate(adata, majority_voting=majority_voting, min_prop=min_prop, p_thres=p_thres, output_path=output_path)
