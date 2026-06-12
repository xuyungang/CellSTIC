"""Load CCI matrices and spatial metadata from CellSTIC result AnnData."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from anndata import AnnData

from utils.time import filter as time_filter


def lr_name_to_stem(lr_name: str) -> str:
    return str(lr_name).replace(":", "-")


def _stage_sort_key(stage: str):
    s = str(stage)
    return float(s) if s.replace(".", "", 1).isdigit() else s


def parse_run_tag(run_tag: str) -> Optional[Tuple[str, str]]:
    """Parse ``<stage>_<organ>`` directory names (e.g. ``9.5_Brain``).

    Names like ``develop_44`` are treated as a single stage tag, not stage/organ.
    """
    if "_" not in run_tag:
        return None
    stage, organ = run_tag.split("_", 1)
    if not stage or not organ:
        return None
    if stage.replace(".", "", 1).isdigit():
        return stage, organ
    return None


def _resolve_obs_key(adata: AnnData, preferred: str, fallbacks: Tuple[str, ...]) -> str:
    if preferred in adata.obs:
        return preferred
    for key in fallbacks:
        if key in adata.obs:
            return key
    raise ValueError(
        f"Expected obs column {preferred!r} or one of {fallbacks}; got {list(adata.obs.columns)}"
    )


def resolve_lr_channel(edge_type_map: Dict[str, int], lr_pair: str) -> Optional[Tuple[str, int]]:
    candidates = [
        lr_pair,
        lr_pair.replace("-", ":"),
        lr_pair.replace("_", "-"),
        lr_pair.replace("-", "_"),
    ]
    for name in candidates:
        if name in edge_type_map:
            return name, int(edge_type_map[name])
    stem_norm = lr_pair.replace("_", "-")
    for name, ch in edge_type_map.items():
        if lr_name_to_stem(name).replace("_", "-") == stem_norm:
            return name, int(ch)
    return None


class CciSource:
    """Read CCI from a combined AnnData, an ``adata_map``, or per-run files under ``result_root``."""

    def __init__(
        self,
        *,
        adata: Optional[AnnData] = None,
        stage_key: str = "stage",
        organ_key: str = "organ",
        result_root: Optional[Union[str, Path]] = None,
        adata_map: Optional[Dict[Tuple[str, str], AnnData]] = None,
    ):
        if adata is None and result_root is None and not adata_map:
            raise ValueError("Provide adata, result_root, or adata_map.")
        self._combined = adata
        self.stage_key = stage_key
        self.organ_key = organ_key
        if adata is not None:
            self.stage_key = _resolve_obs_key(adata, stage_key, ("stage",))
            self.organ_key = _resolve_obs_key(adata, organ_key, ("organ", "annotation"))
        self.result_root = Path(result_root) if result_root is not None else None
        self._adata_map = dict(adata_map or {})
        self._cache: Dict[Tuple[str, str], AnnData] = {}
        self._edge_type_cache: Dict[Tuple[str, str], Dict[str, int]] = {}

    @classmethod
    def from_adata(
        cls,
        adata: AnnData,
        *,
        stage_key: str = "stage",
        organ_key: str = "organ",
    ) -> "CciSource":
        return cls(adata=adata, stage_key=stage_key, organ_key=organ_key)

    def list_stages(self) -> List[str]:
        stages: set = set()
        if self._combined is not None:
            stages.update(self._combined.obs[self.stage_key].astype(str).unique())
        if self.result_root is not None and self.result_root.is_dir():
            for d in self.result_root.iterdir():
                if not d.is_dir() or not (d / "cellstic_result.h5ad").exists():
                    continue
                parsed = parse_run_tag(d.name)
                if parsed is not None:
                    stages.add(parsed[0])
                    continue
                stages.add(d.name)
            for d in self.result_root.iterdir():
                if not d.is_dir() or (d / "cellstic_result.h5ad").exists():
                    continue
                for sub in d.iterdir():
                    if sub.is_dir() and (sub / "cellstic_result.h5ad").exists():
                        stages.add(d.name)
        stages.update(str(s) for s, _ in self._adata_map)
        return sorted(stages, key=_stage_sort_key)

    def _result_paths(self, stage: str, organ: str) -> List[Path]:
        stage = str(stage)
        organ = str(organ)
        run_tag = f"{stage}_{organ}"
        paths = [self.result_root / run_tag / "cellstic_result.h5ad"]
        if self.result_root is not None:
            paths.extend([
                self.result_root / stage / organ / "cellstic_result.h5ad",
                self.result_root / stage / "result" / "cellstic_result.h5ad",
                self.result_root / stage / "cellstic_result.h5ad",
            ])
        return paths

    def _subset_combined(self, stage: str, organ: str) -> Optional[AnnData]:
        obs = self._combined.obs
        mask = (
            obs[self.stage_key].astype(str).eq(str(stage))
            & obs[self.organ_key].astype(str).eq(str(organ))
        )
        if not mask.any():
            return None
        return self._combined[mask].copy()

    def get_adata(self, stage: str, organ: str) -> Optional[AnnData]:
        key = (str(stage), str(organ))
        if key in self._cache:
            return self._cache[key]
        if key in self._adata_map:
            self._cache[key] = self._adata_map[key]
            return self._cache[key]
        if self._combined is not None:
            sub = self._subset_combined(stage, organ)
            if sub is not None:
                self._cache[key] = sub
            return sub
        if self.result_root is None:
            return None
        import scanpy as sc

        for path in self._result_paths(stage, organ):
            if path.exists():
                adata = sc.read_h5ad(path)
                self._cache[key] = adata
                return adata
        return None

    def _edge_type_map(self, stage: str, organ: str) -> Dict[str, int]:
        key = (str(stage), str(organ))
        if key in self._edge_type_cache:
            return self._edge_type_cache[key]
        from pipeline.runner import single_level_from_adata

        adata = self.get_adata(stage, organ)
        if adata is None:
            return {}
        _, edge_type_map = single_level_from_adata(adata)
        self._edge_type_cache[key] = dict(edge_type_map)
        return self._edge_type_cache[key]

    def list_organs(self, stage: str, annotation_filter: Optional[List[str]] = None) -> List[str]:
        organs: set = set()
        if self._combined is not None:
            obs = self._combined.obs
            stage_mask = obs[self.stage_key].astype(str).eq(str(stage))
            organs.update(obs.loc[stage_mask, self.organ_key].astype(str).unique())
        if self.result_root is not None and self.result_root.is_dir():
            prefix = f"{stage}_"
            for d in self.result_root.iterdir():
                if not d.is_dir() or not (d / "cellstic_result.h5ad").exists():
                    continue
                parsed = parse_run_tag(d.name)
                if parsed is not None and parsed[0] == str(stage):
                    organs.add(parsed[1])
                elif d.name.startswith(prefix):
                    organs.add(d.name[len(prefix):])
            stage_path = self.result_root / str(stage)
            if stage_path.is_dir():
                for d in stage_path.iterdir():
                    if d.is_dir() and (d / "cellstic_result.h5ad").exists():
                        organs.add(d.name)
            if not organs:
                for organ in annotation_filter or []:
                    if self.get_adata(stage, organ) is not None:
                        organs.add(organ)
        for s, o in self._adata_map:
            if str(s) == str(stage):
                organs.add(str(o))
        return sorted(o for o in organs if time_filter.annotation_pass(o, annotation_filter))

    def list_lr_stems(self, stage: str, organ: str, lr_filter: Optional[List[str]] = None) -> List[str]:
        stems = [lr_name_to_stem(name) for name in self._edge_type_map(stage, organ)]
        return sorted(s for s in stems if time_filter.lr_match(s, lr_filter))

    def load_cci_dataframe(self, stage: str, organ: str, lr_stem: str) -> Optional[pd.DataFrame]:
        adata = self.get_adata(stage, organ)
        if adata is None:
            return None
        from pipeline.runner import single_level_from_adata

        probs, edge_type_map = single_level_from_adata(adata)
        resolved = resolve_lr_channel(edge_type_map, lr_stem)
        if resolved is None:
            return None
        _, channel = resolved
        cells = adata.obs_names.astype(str)
        return pd.DataFrame(probs[:, :, channel], index=cells, columns=cells)

    def load_cci_for_lr_pair(self, stage: str, organ: str, lr_pair: str) -> Optional[pd.DataFrame]:
        for stem in (lr_pair, lr_pair.replace("-", "_"), lr_pair.replace("_", "-")):
            df = self.load_cci_dataframe(stage, organ, stem)
            if df is not None:
                return df
        return None
