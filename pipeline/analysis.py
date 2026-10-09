"""Reusable visualizations for CellSTIC predictions and hierarchy tables.

This module exposes spatial CCC maps, spatial-domain maps, CCC hierarchy plots,
and square hierarchy plots rendered from supplied plotting tables. Data
collection and paper-specific statistical analysis belong in :mod:`paper.analysis`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from anndata import AnnData


PathLike = Union[str, Path]
_OBSP_PREFIX = "cellstic_L"


def _obsp_key(level_num: int, channel_name: str) -> str:
    safe_name = (
        str(channel_name).replace(":", "_").replace("/", "_").replace(" ", "_").strip()
        or "channel"
    )
    return f"{_OBSP_PREFIX}{level_num}_lr_{safe_name}"


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, (list, tuple)) and value and hasattr(value[0], "toarray"):
        return np.stack([matrix.toarray() for matrix in value], axis=2)
    return np.asarray(value)


def _retain_sparse_probabilities(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and all(
        hasattr(item, "tocsr") for item in value
    ):
        return [item.tocsr() for item in value]
    return _to_numpy(value)


def _load_domain_from_csv(
    adata: AnnData,
    path: PathLike,
    cell_id_column: str,
    domain_column: str,
    key: str,
) -> None:
    frame = pd.read_csv(path)
    if cell_id_column not in frame or domain_column not in frame:
        raise ValueError(
            f"domain CSV must have columns {cell_id_column!r} and " f"{domain_column!r}"
        )
    labels = (
        frame.set_index(cell_id_column)[domain_column]
        .astype(str)
        .reindex(adata.obs_names)
    )
    if labels.isna().all():
        raise ValueError("No overlap between adata.obs_names and domain CSV cell ids")
    adata.obs[key] = pd.Categorical(labels.fillna("unknown").to_numpy())


def _group_names(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels).astype(str)
    valid = labels != "unknown"
    names = np.unique(labels[valid])
    names = np.asarray(
        sorted(
            names,
            key=lambda value: (
                not str(value).replace(".", "", 1).isdigit(),
                str(value),
            ),
        )
    )
    onehot = np.zeros((len(labels), len(names)), dtype=float)
    lookup = {name: index for index, name in enumerate(names)}
    for row in np.flatnonzero(valid):
        onehot[row, lookup[labels[row]]] = 1.0
    return onehot, names


def _leaf_group_map(
    hierarchy: Mapping[str, Any], leaves: Sequence[str]
) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    level_keys = sorted(
        (key for key in hierarchy if key.startswith("level_")),
        key=lambda key: int(key.split("_")[1]),
    )
    out: Dict[str, Dict[str, str]] = {}
    for key in level_keys:
        mapping: Dict[str, str] = {}
        data = hierarchy[key]
        for group, members in data.items():
            members = [members] if isinstance(members, dict) else members
            for member in members:
                leaf = (
                    member.get("edge_type_name") if isinstance(member, dict) else None
                )
                if leaf in leaves:
                    mapping[leaf] = group
        out[key] = {leaf: mapping.get(leaf, leaf) for leaf in leaves}
    return level_keys, out


def _sanitize(value: str) -> str:
    return "".join(
        char if char.isalnum() or char in "_-" else "_" for char in str(value)
    )


class Analysis:
    """Minimal visualization facade for one CellSTIC prediction.

    Parameters may be supplied directly or populated with
    :meth:`from_result`. A final-level CCC prediction needs ``adata``,
    ``pos_edge_probs``, and ``edge_type_map``. Hierarchy visualization needs
    the list returned by ``CellSTICPipeline.predict_tree()``.
    Square cell-type hierarchy plots accept packed out.h5ad data or a source table.
    """

    def __init__(
        self,
        *,
        adata: Optional[AnnData] = None,
        pos_edge_probs: Optional[Any] = None,
        edge_type_map: Optional[Mapping[str, int]] = None,
        tree_level_results: Optional[Sequence[Mapping[str, Any]]] = None,
        output_path: PathLike = "analysis",
    ) -> None:
        self.adata = adata
        self.pos_edge_probs = (
            None
            if pos_edge_probs is None
            else _retain_sparse_probabilities(pos_edge_probs)
        )
        self.edge_type_map = (
            None
            if edge_type_map is None
            else {str(name): int(index) for name, index in edge_type_map.items()}
        )
        self.tree_level_results = (
            None
            if tree_level_results is None
            else [dict(result) for result in tree_level_results]
        )
        self.output_path = Path(output_path)

        if self.tree_level_results:
            leaf_result = max(
                self.tree_level_results,
                key=lambda result: int(result.get("level_num", 0)),
            )
            if self.adata is None:
                self.adata = leaf_result.get("adata")
            if self.pos_edge_probs is None:
                probabilities = leaf_result.get("pos_edge_probs_np")
                if probabilities is not None:
                    self.pos_edge_probs = _retain_sparse_probabilities(probabilities)
            if self.edge_type_map is None:
                edge_map = leaf_result.get("edge_type_map")
                if edge_map is not None:
                    self.edge_type_map = {
                        str(name): int(index) for name, index in edge_map.items()
                    }

    @classmethod
    def from_result(
        cls,
        result: Any,
        *,
        output_path: Optional[PathLike] = None,
    ) -> "Analysis":
        """Build an analysis facade from ``out.h5ad`` or an in-memory payload.

        Packed CellSTIC hierarchy matrices are reconstructed lazily from the
        sparse ``obsp`` entries in the exported AnnData object.
        """
        prediction = getattr(result, "prediction", result)
        pipeline = getattr(result, "pipeline", None)
        resolved_output = output_path
        if resolved_output is None and pipeline is not None:
            resolved_output = Path(pipeline.output_path)
        if resolved_output is None:
            resolved_output = "analysis"

        if isinstance(prediction, AnnData):
            metadata = prediction.uns.get("cellstic", {})
            levels_raw = metadata.get("tree_levels_json", "[]")
            hierarchy_raw = metadata.get("hierarchy_dict_json", "{}")
            levels = json.loads(str(levels_raw)) if levels_raw else []
            hierarchy_dict = json.loads(str(hierarchy_raw)) if hierarchy_raw else {}
            tree_results: List[Dict[str, Any]] = []
            for level in levels:
                level_num = int(level["level_num"])
                edge_type_map = {
                    str(name): int(index)
                    for name, index in level["edge_type_map"].items()
                }
                probabilities = [None] * len(edge_type_map)
                for channel_name, channel_idx in edge_type_map.items():
                    key = _obsp_key(level_num, channel_name)
                    if key not in prediction.obsp:
                        raise KeyError(
                            f"Packed CellSTIC output is missing obsp[{key!r}]"
                        )
                    matrix = prediction.obsp[key]
                    probabilities[channel_idx] = (
                        matrix.tocsr() if hasattr(matrix, "tocsr") else matrix
                    )
                tree_results.append(
                    {
                        "level_key": str(level["level_key"]),
                        "level_num": level_num,
                        "pos_edge_probs_np": probabilities,
                        "edge_type_map": edge_type_map,
                        "output_source": level.get("output_source", "unknown"),
                        "adata": prediction,
                        "hierarchy_dict": hierarchy_dict,
                    }
                )
            return cls(
                adata=prediction,
                tree_level_results=tree_results or None,
                output_path=resolved_output,
            )

        if isinstance(prediction, tuple) and len(prediction) >= 3:
            probabilities, edge_type_map, adata = prediction[:3]
            if not isinstance(adata, AnnData):
                raise TypeError(
                    "The third predict_ccc() value must be an AnnData object"
                )
            return cls(
                adata=adata,
                pos_edge_probs=probabilities,
                edge_type_map=edge_type_map,
                output_path=resolved_output,
            )

        if isinstance(prediction, list):
            return cls(
                tree_level_results=prediction,
                output_path=resolved_output,
            )

        raise TypeError(
            "Unsupported result: expected out.h5ad AnnData, a CCC tuple, "
            "or tree-level result list"
        )

    def _require_adata(self) -> AnnData:
        if self.adata is None:
            raise ValueError("No AnnData is available for visualization")
        if "spatial" not in self.adata.obsm:
            raise KeyError("adata.obsm['spatial'] is required for spatial plots")
        return self.adata

    def _require_ccc(self) -> Tuple[AnnData, Any, Dict[str, int]]:
        adata = self._require_adata()
        if self.pos_edge_probs is None or self.edge_type_map is None:
            raise ValueError(
                "Spatial CCC visualization requires pos_edge_probs and edge_type_map"
            )
        expected_prefix = (adata.n_obs, adata.n_obs)
        probabilities = self.pos_edge_probs
        if isinstance(probabilities, (list, tuple)):
            if not all(
                getattr(matrix, "shape", None) == expected_prefix
                for matrix in probabilities
            ):
                raise ValueError(
                    "Every sparse CCC channel must have shape " f"{expected_prefix}"
                )
            n_channels = len(probabilities)
        else:
            probabilities = _to_numpy(probabilities)
            if probabilities.ndim != 3 or probabilities.shape[:2] != expected_prefix:
                raise ValueError(
                    "pos_edge_probs must have shape "
                    f"({adata.n_obs}, {adata.n_obs}, n_channels); got "
                    f"{probabilities.shape}"
                )
            n_channels = probabilities.shape[2]
        if self.edge_type_map:
            indices = list(self.edge_type_map.values())
            if min(indices) < 0 or max(indices) >= n_channels:
                raise ValueError(
                    "edge_type_map contains an index outside the probability channels"
                )
        return adata, probabilities, self.edge_type_map

    @staticmethod
    def _spatial_view(adata: AnnData) -> AnnData:
        """Create a lightweight, non-mutating AnnData view for plotting."""
        obs = adata.obs.copy()
        var = pd.DataFrame(index=adata.var_names.copy())
        return AnnData(
            X=None,
            obs=obs,
            var=var,
            obsm={"spatial": np.asarray(adata.obsm["spatial"]).copy()},
        )

    def plot_cell_type_hierarchy_square(
        self,
        *,
        h5ad_path: Optional[PathLike] = None,
        source_data: Optional[Union[PathLike, pd.DataFrame]] = None,
        cell_type_key: str = "cell_type",
        pairs: Optional[Sequence[Tuple[str, str]]] = None,
        sender: Optional[str] = None,
        threshold: float = 0.0,
        min_count_threshold: int = 1,
        min_receiver_cells: int = 200,
        save_path: Optional[PathLike] = None,
        dpi: int = 600,
        size_mm: float = 84.0,
        branch_colors: Optional[Mapping[str, str]] = None,
        pair_labels: Optional[Mapping[str, str]] = None,
    ) -> Path:
        """Export a square L2 → L3 → LR hierarchy for four cell-type pairs.

        Read a pipeline ``out.h5ad`` through ``h5ad_path``, or use the packed
        AnnData already stored on this Analysis instance. LR scores are summed
        by directed cell-type pair after removing self edges and applying the
        score/count filters. No training or prediction is rerun.

        ``pairs`` explicitly assigns top/right/bottom/left sides. Otherwise,
        choose four abundant receivers with distinct LR compositions for
        ``sender``; if omitted, use the sender with greatest total strength to
        eligible receivers. See ``HierarchySquareVisualizer.prepare_source``
        for the deterministic selection rule. Set ``min_receiver_cells`` lower
        for small datasets, or supply explicit pairs.

        An existing panel E CSV/DataFrame can instead be passed as
        ``source_data`` for a pure redraw; aggregation options then do not apply.

        PDF, SVG, PNG and source CSV files are written to
        ``output_path/cell_type_hierarchy_square`` by default. ``save_path`` can
        select another directory or a figure filename. ``branch_colors`` maps
        L2 codes to colors; ``pair_labels`` overrides labels by square side.
        Return the SVG path. The original paper scripts remain independent.
        """
        from utils.visualization import HierarchySquareVisualizer

        visualizer = HierarchySquareVisualizer()
        if source_data is not None:
            if (h5ad_path is not None or pairs is not None or sender is not None
                    or cell_type_key != "cell_type" or threshold != 0.0
                    or min_count_threshold != 1 or min_receiver_cells != 200):
                raise ValueError("source_data redraw cannot be combined with h5ad_path or aggregation options")
        else:
            packed = h5ad_path if h5ad_path is not None else self.adata
            if packed is None:
                raise ValueError("Supply h5ad_path, packed AnnData, or a source_data table")
            source_data = visualizer.prepare_source(
                packed, cell_type_key=cell_type_key, pairs=pairs, sender=sender,
                threshold=threshold, min_count_threshold=min_count_threshold,
                min_receiver_cells=min_receiver_cells,
            )
        target = (
            Path(save_path)
            if save_path is not None
            else self.output_path / "cell_type_hierarchy_square"
        )
        return visualizer.plot(
            source_data=source_data,
            save_path=target,
            dpi=dpi,
            size_mm=size_mm,
            branch_colors=branch_colors,
            pair_labels=pair_labels,
        )

    def plot_spatial_ccc(
        self,
        *,
        lr_filter: Optional[List[str]] = None,
        threshold: float = 0.0,
        save_path: Optional[PathLike] = None,
        figsize: Optional[Tuple[int, int]] = None,
        max_outgoing_edges_per_node: int = 3,
        max_incoming_edges_per_node: int = 3,
        show_intensity_axis: bool = True,
        domain_key: Optional[str] = None,
        domain_path: Optional[PathLike] = None,
        domain_file_cell_id_column: str = "cell_id",
        domain_file_domain_column: str = "cluster",
        domains: Optional[List[str]] = None,
    ) -> Path:
        """Export leaf-LR CCC metrics and both spatial CCC map variants.

        No aggregate ``total`` channel or ROC/PR curve is generated. Each leaf
        LR pair receives a strength-colored map and a region-annotated map.
        """
        from utils.visualization import (
            LigandReceptorSpatialVisualizer,
        )

        adata, probabilities, edge_type_map = self._require_ccc()
        leaf_channels = {
            name: index
            for name, index in edge_type_map.items()
            if str(name).strip().lower() not in {"total", "root"}
        }
        if not leaf_channels:
            raise ValueError(
                "CCC spatial visualization requires a leaf LR level, not a root or total channel"
            )
        ordered_channels = sorted(leaf_channels.items(), key=lambda item: item[1])
        if lr_filter is not None:
            lr_filter = [
                pair
                for pair in lr_filter
                if pair.strip().lower() not in {"total", "root"}
            ]
            if not lr_filter:
                raise ValueError("Only leaf ligand-receptor pairs may be requested")
            requested = set(lr_filter)
            ordered_channels = [
                item for item in ordered_channels if item[0] in requested
            ]
            if not ordered_channels:
                raise ValueError(
                    "None of the requested ligand-receptor pairs is available"
                )

        selected_indices = [index for _, index in ordered_channels]
        if isinstance(probabilities, (list, tuple)):
            probabilities = [probabilities[index] for index in selected_indices]
        else:
            probabilities = probabilities[:, :, selected_indices]
        edge_type_map = {
            name: index for index, (name, _) in enumerate(ordered_channels)
        }

        adata_viz = self._spatial_view(adata)
        target = Path(save_path) if save_path is not None else self.output_path / "ccc"
        target.mkdir(parents=True, exist_ok=True)

        visualizer = LigandReceptorSpatialVisualizer()
        if domain_path is not None:
            domain_key = domain_key or "domain"
            _load_domain_from_csv(
                adata_viz,
                domain_path,
                domain_file_cell_id_column,
                domain_file_domain_column,
                domain_key,
            )
        elif domain_key is None:
            domain_key = next(
                (key for key in ("domain", "cluster") if key in adata_viz.obs),
                None,
            )
        if domain_key is None or domain_key not in adata_viz.obs:
            raise KeyError(
                "A spatial domain label is required for the region-annotated CCC map; "
                "set domain_key or domain_path"
            )

        visualizer.plot_ligand_receptor_spatial_distribution(
            adata=adata_viz,
            edge_type_map=edge_type_map,
            pos_edge_probs=probabilities,
            threshold=threshold,
            lr_filter=lr_filter,
            save_path=str(target),
            figsize=figsize,
            max_outgoing_edges_per_node=max_outgoing_edges_per_node,
            max_incoming_edges_per_node=max_incoming_edges_per_node,
            show_intensity_axis=show_intensity_axis,
        )
        visualizer.plot_ligand_receptor_spatial_distribution_by_region(
            adata=adata_viz,
            edge_type_map=edge_type_map,
            pos_edge_probs=probabilities,
            threshold=threshold,
            lr_filter=lr_filter,
            save_path=str(target),
            figsize=figsize,
            max_outgoing_edges_per_node=max_outgoing_edges_per_node,
            max_incoming_edges_per_node=max_incoming_edges_per_node,
            region_key=domain_key,
            regions_to_plot=domains,
        )
        return target

    def plot_hierarchy_spatial_strength(
        self,
        *,
        level: int = 2,
        threshold: float = 0.0,
        save_path: Optional[PathLike] = None,
        channel_titles: Optional[Mapping[str, str]] = None,
    ) -> Path:
        """Export spatial strength maps for every channel at a saved tree level.

        Sum incoming and outgoing weights after removing self edges, then show
        log1p strength with a shared color scale. Ancestor matrices are used as
        stored. No training or descendant re-aggregation is performed.

        PNG, SVG, PDF and source CSV files share ``ccc/level{level}_spatial/``
        by default; ``save_path`` may select another directory. Return the SVG.
        """
        from utils.visualization import HierarchySpatialVisualizer

        if not isinstance(level, (int, np.integer)) or level < 1:
            raise ValueError("level must be a positive integer")
        selected = next(
            (item for item in self.tree_level_results or []
             if int(item.get("level_num", 0)) == level),
            None,
        )
        if selected is None:
            raise ValueError(f"No saved hierarchy level {level}; use Analysis.from_result with tree results")
        adata = selected.get("adata")
        if adata is None:
            adata = self._require_adata()
        if "spatial" not in adata.obsm:
            raise KeyError("adata.obsm['spatial'] is required for spatial plots")
        probabilities = selected.get("pos_edge_probs_np")
        edge_map = selected.get("edge_type_map")
        if probabilities is None or not edge_map:
            raise ValueError(f"Hierarchy level {level} lacks channel scores or names")
        is_sequence = isinstance(probabilities, (list, tuple))
        if not is_sequence:
            probabilities = _to_numpy(probabilities)
        if not is_sequence and probabilities.ndim != 3:
            raise ValueError("Dense hierarchy scores must have shape (n_obs, n_obs, n_channels)")
        count = len(probabilities) if is_sequence else probabilities.shape[2]
        if any(int(index) < 0 or int(index) >= count for index in edge_map.values()):
            raise ValueError("Hierarchy channel index is outside the score array")
        matrices = {
            str(name): probabilities[int(index)] if is_sequence else probabilities[:, :, int(index)]
            for name, index in edge_map.items()
        }
        target = Path(save_path) if save_path is not None else self.output_path / "ccc" / f"level{level}_spatial"
        return HierarchySpatialVisualizer().plot(
            adata=adata, channel_matrices=matrices, level=level,
            threshold=threshold, save_path=target, channel_titles=channel_titles,
        )

    def plot_spatial_domains(
        self,
        *,
        domain_key: Optional[str] = None,
        domain_path: Optional[PathLike] = None,
        domain_file_cell_id_column: str = "cell_id",
        domain_file_domain_column: str = "cluster",
        save_path: Optional[PathLike] = None,
        palette: Optional[str] = None,
        size_multiplier: float = 1.0,
        figsize: Tuple[int, int] = (8, 6),
        feature_key: str = "feat",
        true_label_key: Optional[str] = None,
        n_neighbors: int = 40,
    ) -> Path:
        """Export spatial domains, UMAP, partitions, and clustering metrics.

        Besides the returned spatial SVG, this writes ``umap.svg``,
        ``cell_partitions.csv``, and ``clustering_metrics.csv`` alongside the
        domain output.  Labels may come from ``domain_key``/``domain_path``;
        optional ground-truth labels are read from ``true_label_key``.
        """
        from utils.metrics import MetricsComputer, SpatialVisualizer

        source = self._require_adata()
        adata_viz = self._spatial_view(source)
        if feature_key not in source.obsm:
            raise KeyError(
                f"adata.obsm[{feature_key!r}] is required for UMAP and metrics export"
            )
        adata_viz.obsm[feature_key] = _to_numpy(source.obsm[feature_key]).copy()
        if domain_path is not None:
            domain_key = domain_key or "domain"
            _load_domain_from_csv(
                adata_viz,
                domain_path,
                domain_file_cell_id_column,
                domain_file_domain_column,
                domain_key,
            )
        elif domain_key is None:
            domain_key = next(
                (key for key in ("domain", "cluster") if key in adata_viz.obs),
                None,
            )

        if domain_key is None or domain_key not in adata_viz.obs:
            raise KeyError(
                "No spatial domain labels found; set domain_key or domain_path"
            )
        adata_viz.obs["cluster"] = pd.Categorical(adata_viz.obs[domain_key].astype(str))

        target = (
            Path(save_path)
            if save_path is not None
            else self.output_path / "domain" / "spatial_domain.svg"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        # The metrics utility owns the standard domain artifact bundle.  Give
        # it the run root when the default layout is used, otherwise use the
        # parent of the explicitly requested spatial figure.
        export_root = (
            target.parent.parent if target.parent.name == "domain" else target.parent
        )
        true_labels = None
        if true_label_key is not None:
            if true_label_key not in adata_viz.obs:
                raise KeyError(
                    f"true_label_key {true_label_key!r} is absent from adata.obs"
                )
            true_labels = adata_viz.obs[true_label_key].to_numpy()
        MetricsComputer.run_region_umap_metrics_export(
            adata_viz,
            save_dir=export_root,
            feature_key=feature_key,
            true_labels=true_labels,
            cluster_key="cluster",
            n_neighbors=n_neighbors,
        )

        # Preserve the caller's explicit spatial-plot styling and path.  The
        # standard export above already created this file when it is the
        # default target, so only render a second copy when options differ.
        standard_target = export_root / "domain" / "spatial_domain.svg"
        if (
            target != standard_target
            or palette is not None
            or size_multiplier != 0.85
            or figsize != (8, 6)
        ):
            SpatialVisualizer.generate_spatial_domain_visualization(
                adata_viz,
                save_path=str(target),
                palette=palette,
                size_multiplier=size_multiplier,
                figsize=figsize,
            )
        return target

    def plot_hierarchy(
        self,
        *,
        group_by: str = "domain",
        group_key: Optional[str] = None,
        domain_path: Optional[PathLike] = None,
        domain_file_cell_id_column: str = "cell_id",
        domain_file_domain_column: str = "cluster",
        threshold: float = 0.7,
        min_count_threshold: int = 500,
        min_width_fraction: float = 0.01,
        save_path: Optional[PathLike] = None,
        figsize: Tuple[float, float] = (8.0, 5.0),
        dpi: int = 300,
        show_title: bool = False,
    ) -> Path:
        """Render hierarchy views plus cell-type icicles and stacked bars.

        ``group_by='domain'`` aggregates hierarchy strength by spatial-domain
        pairs. ``group_by='cell_type'`` aggregates by cell-type pairs.
        """
        from utils.visualization.icicle_viz import IcicleVisualizer
        from utils.visualization.stacked_bar_viz import (
            SenderReceiverStackedBarVisualizer,
        )

        if not self.tree_level_results:
            raise ValueError(
                "Hierarchy visualization requires pipeline.predict_tree() results"
            )
        hierarchy_dict = next(
            (
                result.get("hierarchy_dict")
                for result in self.tree_level_results
                if result.get("hierarchy_dict")
            ),
            None,
        )
        if not hierarchy_dict:
            raise ValueError("tree results contain no hierarchy_dict")

        target = Path(save_path) if save_path is not None else self.output_path
        target.mkdir(parents=True, exist_ok=True)
        normalized_results: List[Dict[str, Any]] = []
        for position, source in enumerate(self.tree_level_results, start=1):
            result = dict(source)
            level_num = int(result.get("level_num", position))
            result.setdefault("output_dir", target / f"level_{level_num}")
            adata = result.get("adata")
            if isinstance(adata, AnnData):
                result["adata"] = AnnData(X=None, obs=adata.obs.copy())
            probabilities = result.get("pos_edge_probs_np")
            if probabilities is not None:
                result["pos_edge_probs_np"] = _retain_sparse_probabilities(
                    probabilities
                )
            normalized_results.append(result)

        if group_by not in {"domain", "cell_type"}:
            raise ValueError("group_by must be 'domain' or 'cell_type'")
        leaf_result = max(
            normalized_results, key=lambda item: int(item.get("level_num", 0))
        )
        leaf_adata = leaf_result.get("adata")
        if not isinstance(leaf_adata, AnnData):
            raise ValueError("Leaf hierarchy result contains no AnnData metadata")
        if domain_path is not None:
            domain_key = group_key or "domain"
            _load_domain_from_csv(
                leaf_adata,
                domain_path,
                domain_file_cell_id_column,
                domain_file_domain_column,
                domain_key,
            )
            if group_by == "domain":
                group_key = domain_key
        if group_key is not None:
            label_key = group_key
        elif group_by == "cell_type":
            label_key = "cell_type"
        else:
            label_key = next(
                (key for key in ("domain", "cluster") if key in leaf_adata.obs),
                "domain",
            )
        if label_key not in leaf_adata.obs:
            raise KeyError(f"group_by={group_by!r} requires adata.obs[{label_key!r}]")
        labels = leaf_adata.obs[label_key].astype(str).to_numpy()
        probabilities = leaf_result["pos_edge_probs_np"]
        edge_map = leaf_result["edge_type_map"]
        leaves = [
            name for name, _ in sorted(edge_map.items(), key=lambda item: item[1])
        ]
        level_keys, leaf_to_group = _leaf_group_map(hierarchy_dict, leaves)
        onehot, groups = _group_names(labels)
        radial = IcicleVisualizer()
        out_dir = target / "alluvial" / f"{group_by}_pairs"
        out_dir.mkdir(parents=True, exist_ok=True)
        mapping_path = out_dir / "hierarchy_node_mapping.txt"
        with mapping_path.open("w", encoding="utf-8") as handle:
            handle.write("level\tnode_id\tnode_name\tchildren\n")
            for level_key in level_keys:
                level_data = hierarchy_dict.get(level_key, {})
                for node_index, (node_name, members) in enumerate(
                    level_data.items(), start=1
                ):
                    children = []
                    for member in members if isinstance(members, list) else [members]:
                        if isinstance(member, dict):
                            child = (
                                member.get("edge_type_name")
                                or member.get("group_name")
                                or str(member)
                            )
                        else:
                            child = str(member)
                        children.append(child)
                    handle.write(
                        f"{level_key}\t{level_key.replace('_', '').upper()}-{node_index}\t{node_name}\t{' | '.join(children)}\n"
                    )
        leaf_matrices = {}
        for leaf, channel in edge_map.items():
            if isinstance(probabilities, (list, tuple)):
                matrix = probabilities[channel].tocsr(copy=True)
                rows = np.repeat(np.arange(matrix.shape[0]), np.diff(matrix.indptr))
                matrix.data[matrix.indices == rows] = 0
                if threshold:
                    matrix.data[matrix.data < threshold] = 0
                matrix.eliminate_zeros()
            else:
                matrix = np.asarray(probabilities)[:, :, channel].copy()
                np.fill_diagonal(matrix, 0)
                if threshold:
                    matrix[matrix < threshold] = 0
            leaf_matrices[leaf] = matrix
        for src_index, src in enumerate(groups):
            for tgt_index, tgt in enumerate(groups):
                strengths: Dict[str, float] = {}
                source_mask = onehot[:, src_index].astype(bool)
                target_mask = onehot[:, tgt_index].astype(bool)
                for leaf, matrix in leaf_matrices.items():
                    selected = matrix[source_mask][:, target_mask]
                    selected_count = (
                        selected.nnz
                        if hasattr(selected, "nnz")
                        else int(np.count_nonzero(selected))
                    )
                    if min_count_threshold and selected_count < min_count_threshold:
                        continue
                    strengths[leaf] = float(selected.sum())
                if sum(strengths.values()) <= 0:
                    continue
                radial._plot_alluvial_core(
                    leaf_strengths=strengths,
                    leaf_names=leaves,
                    level_keys=level_keys,
                    leaf_to_group=leaf_to_group,
                    title=f"{src} → {tgt}" if show_title else "",
                    save_path=(
                        out_dir / f"alluvial_{group_by}_{_sanitize(src)}_to_"
                        f"{_sanitize(tgt)}_icicle.svg"
                    ),
                    min_width_fraction=min_width_fraction,
                    figsize=figsize,
                    dpi=dpi,
                    high_contrast=True,
                )
        bar_dir = target / "alluvial" / f"{group_by}_bar"
        bar_dir.mkdir(parents=True, exist_ok=True)
        for item in normalized_results:
            level_num = int(item.get("level_num", 0))
            if level_num <= 1:
                continue
            SenderReceiverStackedBarVisualizer().plot(
                graph=item["pos_edge_probs_np"],
                edge_type_map=item["edge_type_map"],
                adata=leaf_adata,
                cell_type_key=label_key,
                threshold=threshold,
                order_by="edge_index",
                top_n=20,
                save_path=(bar_dir / f"lr_pair_stacked_bar_level_{level_num}.svg"),
            )
        return target


CellSTICAnalysis = Analysis

__all__ = ["Analysis", "CellSTICAnalysis"]
