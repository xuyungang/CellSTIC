"""
Analysis pipeline: comprehensive (single run) and tree-level (per-level) analysis.
Init with data + config; call per-step methods with optional overrides; or run_all().
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from anndata import AnnData

from utils.analysis import (
    SenderReceiverStackedBarVisualizer,
    DifferentialAnalyzer,
    DomainVisualizer,
    load_domain_from_csv,
    AggregatedHeatmapVisualizer,
    LigandReceptorSpatialVisualizer,
    StrengthDistanceVisualizer,
    compute_spot_level_metrics,
    AlluvialVisualizer,
    plot_alluvial_and_icicle_per_cell_type_pair,
    plot_alluvial_and_icicle_per_domain,
)
from utils.metrics import MetricsComputer
from utils.time.ccdf import compute_save_and_plot_ccdf
from utils.time.efficiency import compute_save_and_plot_efficiency
from utils.time.strength_vs_distance import compute_save_and_plot_strength_vs_distance
from utils.time.celltype_strength_over_stages import (
    compute_save_and_plot_celltype_strength_over_stages,
)
from utils.time.density_over_stages import compute_save_and_plot_density_over_stages
from utils.time.strong_nodes_over_stages import compute_save_and_plot_strong_nodes_over_stages


def _merge(base: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge overrides into base; None keys in overrides mean use base."""
    if not overrides:
        return dict(base)
    out = dict(base)
    for k, v in overrides.items():
        if v is not None:
            out[k] = v
    return out


class DomainAnalysis:
    """Domain-level analysis (e.g. domain×domain communication, domain-specific metrics)."""

    _DEFAULTS = {
        "domain_path": None,
        "domain_file_cell_id_column": "cell_id",
        "domain_file_domain_column": "cluster",
        "cell_type_key": "cell_type",
        "top_n_cell_types": 15,
        "figsize": (10, 5),
        "save_path": None,
        "legend_loc": "outside right",
    }

    def __init__(
        self,
        adata: Optional[AnnData] = None,
        output_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        self.adata = adata
        self.output_path = Path(output_path) if output_path else None
        if self.output_path:
            self.output_path.mkdir(parents=True, exist_ok=True)
        self._params = {**self._DEFAULTS, **{k: v for k, v in kwargs.items() if v is not None}}
        self._domain_viz = DomainVisualizer()

    def _p(self, **overrides) -> Dict[str, Any]:
        return _merge(self._params, overrides if overrides else None)

    def _log(self, message: str) -> None:
        print(f"[DomainAnalysis] {message}")

    def run_domain_cell_type_stacked_bar(self, **overrides) -> None:
        """Stacked bar of cell type composition per region; passes domain_path to DomainVisualizer (loads CSV inside plot)."""
        self._log("run_domain_cell_type_stacked_bar")
        if self.adata is None:
            raise ValueError("DomainAnalysis.run_domain_cell_type_stacked_bar requires adata")
        p = self._p(**overrides)
        domain_path = p.get("domain_path")
        if not domain_path:
            raise ValueError("domain_path is required (path to domain CSV, e.g. cell_partitions.csv)")
        out = Path(p["save_path"]) if p.get("save_path") else (
            (self.output_path / "domain" / "domain_cell_type_stacked_bar.svg") if self.output_path else Path("domain_cell_type_stacked_bar.svg")
        )
        self._domain_viz.plot_domain_cell_type_stacked_bar(
            adata=self.adata,
            domain_path=Path(domain_path),
            domain_file_cell_id_column=p.get("domain_file_cell_id_column"),
            domain_file_domain_column=p.get("domain_file_domain_column"),
            cell_type_key=p["cell_type_key"],
            top_n_cell_types=p["top_n_cell_types"],
            save_path=out,
            figsize=p.get("figsize"),
            dpi=p.get("dpi"),
            legend_loc=p.get("legend_loc", "outside right"),
            )


class SingleLevelAnalysis:
    """Single-run analysis: heatmaps, CCC, differential, community, LR spatial, metrics."""

    _DEFAULTS = {
        "lr_filter": None,
        "ccc_ground": None,
        "threshold": 0.5,
        "hotspot_top_pct": 60.0,
        "min_count_threshold": 1,
        "annotation_key": "annotation",
        "cell_type_key": "cell_type",
        "domain1": "0",
        "domain_key": "domain",
        "domain_path": None,
        "domain_file_cell_id_column": "cell_id",
        "domain_file_domain_column": "cluster",
        "subcluster_lr_pair": ("Penk", "Oprk1"),
        "subcluster_min_cells_per_cluster": 50,
        "stacked_bar_top_n": 20,
        "community_resolution_range": (0.01, 10.0),
        "target_n_communities": None,
        "output_path": None,
        "save_path": None,
        "figsize": (8, 6),
        "dpi": 600,
        "score_method": "sum",
        "linewidth": 2.0,
        "point_size": None,
        "zscore_context": None,
        "font_size": None,
    }

    def __init__(
        self,
        adata: AnnData,
        pos_edge_probs_np: np.ndarray,
        edge_type_map: Dict[str, int],
        output_path: Path,
        **kwargs,
    ):
        self.adata = adata
        self.pos_edge_probs_np = pos_edge_probs_np
        self.edge_type_map = edge_type_map
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self._params = {**self._DEFAULTS, **{k: v for k, v in kwargs.items() if v is not None}}

        self._domain_heat = AggregatedHeatmapVisualizer()
        self._differential = DifferentialAnalyzer()
        self._lr_spatial = LigandReceptorSpatialVisualizer()
        self._strength_dist = StrengthDistanceVisualizer()

    def _p(self, **overrides) -> Dict[str, Any]:
        return _merge(self._params, overrides if overrides else None)

    def _log(self, message: str) -> None:
        print(f"[SingleLevelAnalysis] {message}")

    @classmethod
    def from_adata(
        cls,
        adata: AnnData,
        output_path: Union[str, Path],
        level_num: Optional[int] = None,
        **kwargs,
    ) -> "SingleLevelAnalysis":
        """Build analyzer from a self-contained CellSTIC AnnData bundle."""
        from pipeline.runner import single_level_from_adata
        from model.train.data import ccc_ground_from_adata

        pos_edge_probs_np, edge_type_map = single_level_from_adata(adata, level_num=level_num)
        params = dict(kwargs)
        if "ccc_ground" not in params:
            ground = ccc_ground_from_adata(adata)
            if ground is not None:
                params["ccc_ground"] = ground
        return cls(
            adata=adata,
            pos_edge_probs_np=pos_edge_probs_np,
            edge_type_map=edge_type_map,
            output_path=Path(output_path),
            **params,
        )

    def _resolve_lr_filter(self, p: Dict[str, Any], edge_type_map: Optional[Dict[str, int]]) -> List[str]:
        """When lr_filter is None, default to all keys in edge_type_map; otherwise lr_filter or []."""
        lr_filter = p.get("lr_filter")
        if lr_filter is None and edge_type_map:
            return list(edge_type_map.keys())
        return lr_filter if isinstance(lr_filter, list) else []

    def _apply_region_filter(
        self,
        p: Dict[str, Any],
        adata: AnnData,
        pos_edge_probs_np: np.ndarray,
    ) -> Optional[Tuple[AnnData, np.ndarray]]:
        """
        Shared region filtering: load domain from domain_path if set, then subset to
        regions_to_plot if set. Returns (adata, pos_edge_probs_np) or None when
        regions_to_plot is set but no cells match (caller should return).
        """
        domain_path = p.get("domain_path")
        if domain_path is not None:
            load_domain_from_csv(
                adata,
                domain_path=Path(domain_path),
                cell_id_column=p["domain_file_cell_id_column"],
                domain_column=p["domain_file_domain_column"],
                domain_obs_key="domain",
            )
        regions_to_plot = p.get("regions_to_plot")
        if regions_to_plot is not None and len(regions_to_plot) > 0:
            if "domain" not in adata.obs:
                raise ValueError(
                    "regions_to_plot requires domain_path (or adata.obs['domain']) to filter by region."
                )
            regions_set = set(str(r) for r in regions_to_plot)
            mask = adata.obs["domain"].astype(str).isin(regions_set).to_numpy()
            if not np.any(mask):
                self._log(f"No cells in regions_to_plot {regions_to_plot}; skip.")
                return None
            adata = adata[mask].copy()
            pos_edge_probs_np = pos_edge_probs_np[mask][:, mask, :]
        return (adata, pos_edge_probs_np)

    def _region_path_suffix(self, regions_to_plot: Optional[List[Any]]) -> str:
        """Return path suffix for region filtering, e.g. '' or '_regions_1_2_5'."""
        if regions_to_plot is None or len(regions_to_plot) == 0:
            return ""
        return "_regions_" + "_".join(str(r) for r in regions_to_plot)

    def run_cell_type_heatmaps(self, **overrides) -> None:
        """
        Cell type×cell type communication heatmaps (total + per LR).

        Optional region filtering: set domain_path and regions_to_plot; output path
        is suffixed with region names when filtering.
        """
        self._log("run_cell_type_heatmaps")
        p = self._p(**overrides)
        base = Path(p.get("output_path") or self.output_path)
        out = base / ("cell_type_heatmaps" + self._region_path_suffix(p.get("regions_to_plot")))
        out.mkdir(parents=True, exist_ok=True)

        adata = p.get("adata", self.adata)
        pos_edge_probs_np = p.get("pos_edge_probs_np", self.pos_edge_probs_np)
        edge_type_map = p.get("edge_type_map", self.edge_type_map)
        if adata is None or pos_edge_probs_np is None or edge_type_map is None:
            return
        result = self._apply_region_filter(p, adata, pos_edge_probs_np)
        if result is None:
            return
        adata, pos_edge_probs_np = result

        lr_filter = self._resolve_lr_filter(p, edge_type_map)
        self._domain_heat.plot_cell_type_cell_type_heatmaps(
            adata=adata,
            pos_edge_probs_np=pos_edge_probs_np,
            edge_type_map=edge_type_map,
            cell_type_key=p["cell_type_key"],
            cell_type_filter=p.get("cell_type_filter"),
            lr_filter=lr_filter,
            threshold=p["threshold"],
            save_dir=out,
            font_size=p.get("font_size"),
        )

    def run_strength_vs_distance(self, **overrides) -> None:
        """Communication strength vs spatial distance line chart (one line if no ground truth, two if present)."""
        self._log("run_strength_vs_distance")
        p = self._p(**overrides)
        adata = p.get("adata", self.adata)
        pos_edge_probs_np = p.get("pos_edge_probs_np", self.pos_edge_probs_np)
        if adata is None or pos_edge_probs_np is None:
            return
        result = self._apply_region_filter(p, adata, pos_edge_probs_np)
        if result is None:
            return
        adata, pos_edge_probs_np = result
        out = Path(p.get("output_path") or self.output_path) / ("strength_vs_distance" + self._region_path_suffix(p.get("regions_to_plot")))
        out.mkdir(parents=True, exist_ok=True)
        lr_filter = self._resolve_lr_filter(p, self.edge_type_map)
        self._strength_dist.plot_strength_vs_distance(
            pos_edge_probs_np=pos_edge_probs_np,
            edge_type_map=self.edge_type_map,
            adata=adata,
            save_path=out / "strength_vs_distance.svg",
            lr_filter=lr_filter,
            threshold=p["threshold"],
            ccc_ground=p.get("ccc_ground"),
            figsize=p.get("figsize"),
            linewidth=p.get("linewidth"),
        )

    def run_simple_heatmaps(self, **overrides) -> None:
        """Simple communication intensity heatmap per LR pair (no domain). Saves under ``spatial_heatmaps/`` as ``.svg``."""
        self._log("run_simple_heatmaps")
        p = self._p(**overrides)
        base = Path(p.get("output_path") or self.output_path)
        out = base / ("spatial_heatmaps" + self._region_path_suffix(p.get("regions_to_plot")))
        out.mkdir(parents=True, exist_ok=True)
        adata = p.get("adata", self.adata)
        pos_edge_probs_np = p.get("pos_edge_probs_np", self.pos_edge_probs_np)
        edge_type_map = p.get("edge_type_map", self.edge_type_map)
        if adata is None or pos_edge_probs_np is None or edge_type_map is None:
            return
        result = self._apply_region_filter(p, adata, pos_edge_probs_np)
        if result is None:
            return
        adata, pos_edge_probs_np = result
        lr_filter = self._resolve_lr_filter(p, edge_type_map)
        self._domain_heat.plot_simple_communication_heatmap(
            adata=adata,
            pos_edge_probs_np=pos_edge_probs_np,
            edge_type_map=edge_type_map,
            save_dir=out,
            lr_filter=lr_filter,
            threshold=p["threshold"],
            score_method=p.get("score_method"),
            cmap="coolwarm",
            point_size=p.get("point_size"),
            font_size=p.get("font_size"),
        )

    def run_spot_level_metrics(self, **overrides) -> None:
        """Spearman + hotspot overlap per LR pair; bar chart. Requires ccc_ground."""
        self._log("run_spot_level_metrics")
        p = self._p(**overrides)
        ccc_ground = p.get("ccc_ground")
        if ccc_ground is None:
            return
        lr_filter = self._resolve_lr_filter(p, self.edge_type_map)
        if not lr_filter:
            return
        adata = p.get("adata", self.adata)
        pos_edge_probs_np = p.get("pos_edge_probs_np", self.pos_edge_probs_np)
        if pos_edge_probs_np is None:
            return
        result = self._apply_region_filter(p, adata if adata is not None else self.adata, pos_edge_probs_np)
        if result is None:
            return
        _, pos_edge_probs_np = result
        base = Path(p.get("output_path") or self.output_path)
        out = base / ("spot_level_metrics" + self._region_path_suffix(p.get("regions_to_plot")))
        out.mkdir(parents=True, exist_ok=True)
        extra_kwargs = {}
        if p.get("figsize") is not None:
            extra_kwargs["figsize"] = p.get("figsize")
        if p.get("dpi") is not None:
            extra_kwargs["dpi"] = p.get("dpi")

        compute_spot_level_metrics(
            pos_edge_probs_np=pos_edge_probs_np,
            ccc_ground=ccc_ground,
            edge_type_map=self.edge_type_map,
            lr_filter=lr_filter,
            score_method="mean",
            threshold=p["threshold"],
            hotspot_top_pct=p.get("hotspot_top_pct"),
            save_path=out / "spot_level_metrics.svg",
            **extra_kwargs,
        )

    def run_roc_pr_metrics(self, **overrides) -> None:
        """Compute ROC/AUC and F1 metrics vs ground-truth CCC (requires ``ccc_ground``)."""
        self._log("run_roc_pr_metrics")
        p = self._p(**overrides)
        ccc_ground = p.get("ccc_ground")
        if ccc_ground is None:
            return

        adata = p.get("adata", self.adata)
        pos_edge_probs_np = p.get("pos_edge_probs_np", self.pos_edge_probs_np)
        edge_type_map = p.get("edge_type_map", self.edge_type_map)
        if adata is None or pos_edge_probs_np is None or edge_type_map is None:
            return

        n_cells, _, n_types = pos_edge_probs_np.shape
        eval_mask = np.ones((n_cells, n_cells, n_types), dtype=bool)
        for k in range(n_types):
            np.fill_diagonal(eval_mask[:, :, k], False)

        base = Path(p.get("output_path") or self.output_path)
        lr_pair_names = [name for name, _ in sorted(edge_type_map.items(), key=lambda x: x[1])]
        result = MetricsComputer.save_roc_pr_metrics_csv(
            pos_edge_probs_np,
            ccc_ground,
            save_dir=base,
            lr_pair_names=lr_pair_names,
            eval_mask=eval_mask,
            f1_threshold=p["threshold"],
        )
        if result is None:
            return

        print("=== Summary ===")
        print(result["summary"].to_string(index=False))
        print("\n=== Per LR pair ===")
        print(result["per_class"].to_string(index=False))

    def run_domain_domain_heatmaps(self, **overrides) -> None:
        """Domain×domain communication heatmaps (total + per LR). Requires domain_path. Saves under ``domain_domain_heatmaps/`` as ``.svg``."""
        self._log("run_domain_domain_heatmaps")
        p = self._p(**overrides)
        adata = p.get("adata", self.adata)
        pos_edge_probs_np = p.get("pos_edge_probs_np", self.pos_edge_probs_np)
        edge_type_map = p.get("edge_type_map", self.edge_type_map)
        if adata is None or pos_edge_probs_np is None or edge_type_map is None:
            return
        domain_path = p.get("domain_path")
        if domain_path is None:
            return
        result = self._apply_region_filter(p, adata, pos_edge_probs_np)
        if result is None:
            return
        adata, pos_edge_probs_np = result

        domains_arr = adata.obs["domain"].astype(str).to_numpy()
        valid_mask = domains_arr != "unknown"
        if not np.any(valid_mask):
            return
        if not np.all(valid_mask):
            adata = adata[valid_mask].copy()
            pos_edge_probs_np = pos_edge_probs_np[valid_mask][:, valid_mask, :]

        base = Path(p.get("output_path") or self.output_path)
        out = base / ("domain_domain_heatmaps" + self._region_path_suffix(p.get("regions_to_plot")))
        out.mkdir(parents=True, exist_ok=True)
        lr_filter = self._resolve_lr_filter(p, edge_type_map)
        self._domain_heat.plot_domain_domain_heatmaps(
            adata=adata,
            pos_edge_probs_np=pos_edge_probs_np,
            edge_type_map=edge_type_map,
            domain_key="domain",
            lr_filter=lr_filter,
            threshold=p["threshold"],
            save_dir=out,
            font_size=p.get("font_size"),
        )

    def run_stacked_bar(self, **overrides) -> None:
        """LR pair stacked bar by cell type."""
        self._log("run_stacked_bar")
        p = self._p(**overrides)
        adata = p.get("adata", self.adata)
        pos_edge_probs_np = p.get("pos_edge_probs_np", self.pos_edge_probs_np)
        edge_type_map = p.get("edge_type_map", self.edge_type_map)
        if adata is None or pos_edge_probs_np is None or edge_type_map is None:
            return
        result = self._apply_region_filter(p, adata, pos_edge_probs_np)
        if result is None:
            return
        adata, pos_edge_probs_np = result
        save_path_override = p.get("save_path")
        if save_path_override is not None:
            path = Path(save_path_override)
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            base_out = Path(p.get("output_path") or self.output_path)
            base_out.mkdir(parents=True, exist_ok=True)
            path = base_out / ("lr_pair_stacked_bar" + self._region_path_suffix(p.get("regions_to_plot")) + ".svg")
        SenderReceiverStackedBarVisualizer().plot(
            graph=pos_edge_probs_np,
            edge_type_map=edge_type_map,
            adata=adata,
            cell_type_key=p["cell_type_key"],
            lr_filter=p["lr_filter"],
            top_n=p["stacked_bar_top_n"],
            threshold=p["threshold"],
            save_path=str(path),
            figsize=p.get("figsize"),
            dpi=300,
        )

    def run_lr_spatial(self, **overrides) -> None:
        """Ligand-receptor spatial distribution figure."""
        self._log("run_lr_spatial")
        p = self._p(**overrides)
        adata = p.get("adata", self.adata)
        pos_edge_probs_np = p.get("pos_edge_probs_np", self.pos_edge_probs_np)
        if adata is None or pos_edge_probs_np is None:
            return
        result = self._apply_region_filter(p, adata, pos_edge_probs_np)
        if result is None:
            return
        adata, pos_edge_probs_np = result
        base = Path(p.get("output_path") or self.output_path)
        path = base / ("lr_spatial" + self._region_path_suffix(p.get("regions_to_plot")))
        self._lr_spatial.plot_ligand_receptor_spatial_distribution(
            adata=adata,
            edge_type_map=self.edge_type_map,
            pos_edge_probs=pos_edge_probs_np,
            save_path=str(path),
            threshold=p["threshold"],
            figsize=(20, 16),
            lr_filter=p["lr_filter"],
            show_intensity_axis=False,
        )

    def run_region_lr_spatial(self, **overrides) -> None:
        """
        LR spatial distribution with cells colored by region (domain).

        Requires domain_path. Uses shared region filtering; optionally restrict
        to regions_to_plot. Output path is suffixed with region names when filtering.
        """
        self._log("run_region_lr_spatial")
        p = self._p(**overrides)
        domain_path = p.get("domain_path")
        if domain_path is None:
            raise ValueError(
                "run_region_lr_spatial requires domain_path "
                "(path to region CSV, e.g. cell_partitions.csv)"
            )
        adata = p.get("adata", self.adata)
        pos_edge_probs_np = p.get("pos_edge_probs_np", self.pos_edge_probs_np)
        if adata is None or pos_edge_probs_np is None:
            return
        result = self._apply_region_filter(p, adata, pos_edge_probs_np)
        if result is None:
            return
        adata, pos_edge_probs_np = result
        base = Path(p.get("output_path") or self.output_path)
        path = base / ("lr_spatial_region" + self._region_path_suffix(p.get("regions_to_plot")))
        self._lr_spatial.plot_ligand_receptor_spatial_distribution_by_region(
            adata=adata,
            edge_type_map=self.edge_type_map,
            pos_edge_probs=pos_edge_probs_np,
            save_path=str(path),
            threshold=p["threshold"],
            figsize=p.get("figsize", (20, 16)),
            lr_filter=p["lr_filter"],
            region_key="domain",
            regions_to_plot=None,
            max_outgoing_edges_per_node=p.get("max_outgoing_edges_per_node", 3),
            max_incoming_edges_per_node=p.get("max_incoming_edges_per_node", 3),
        )

    def run_domain_subclustering(self, **overrides) -> None:
        """Domain subclustering (UMAP, spatial, DEG heatmap). If domain_path is set, load domain from CSV; else use annotation_key."""
        self._log("run_domain1_subclustering")
        p = self._p(**overrides)
        adata = p.get("adata", self.adata)
        domain_path = p.get("domain_path")
        if domain_path is not None:
            load_domain_from_csv(
                adata,
                domain_path=Path(domain_path),
                cell_id_column=p["domain_file_cell_id_column"],
                domain_column=p["domain_file_domain_column"],
                domain_obs_key="domain",
            )
            domain_key = "domain"
        else:
            domain_key = p["annotation_key"]
        out = Path(p.get("output_path") or self.output_path) / f"{p['domain1']}_subclustering"
        self._differential.plot_domain1_subclustering_analysis(
            adata=adata,
            pos_edge_probs_np=self.pos_edge_probs_np,
            edge_type_map=self.edge_type_map,
            save_dir=out,
            domain_key=domain_key,
            domain1=p["domain1"],
            lr_pair=p["subcluster_lr_pair"],
            threshold=0,
            resolution_range=p.get("community_resolution_range"),
            target_n_communities=p.get("target_n_communities"),
            min_cells_per_cluster=p.get("subcluster_min_cells_per_cluster"),
        )

    # split

class TreeLevelAnalysis:
    """Per-level analysis: alluvial plots and hierarchy-node bubble plots."""

    _DEFAULTS = {
        "threshold": 0.7,
        "min_count_threshold": 120,
        "cell_type_key": "cell_type",
        "domain_key": "domain",
        "domain_path": None,
        "domain_file_cell_id_column": "cell_id",
        "domain_file_domain_column": "cluster",
        "alluvial_min_width_fraction": 0.01,
    }

    def __init__(
        self,
        tree_level_results: Optional[List[Dict[str, Any]]] = None,
        output_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        self._single = None
        self._tree = tree_level_results or []
        self._params = {**self._DEFAULTS, **{k: v for k, v in kwargs.items() if v is not None}}
        self._alluvial = AlluvialVisualizer()

        self.output_path = Path(output_path) if output_path else None
        if self.output_path:
            self.output_path.mkdir(parents=True, exist_ok=True)

    def _p(self, **overrides) -> Dict[str, Any]:
        return _merge(self._params, overrides if overrides else None)

    def _log(self, message: str) -> None:
        print(f"[TreeLevelAnalysis] {message}")

    @classmethod
    def from_adata(
        cls,
        adata: AnnData,
        output_path: Union[str, Path],
        **kwargs,
    ) -> "TreeLevelAnalysis":
        """Build analyzer from a self-contained CellSTIC AnnData bundle."""
        from pipeline.runner import tree_results_from_adata

        return cls(
            tree_level_results=tree_results_from_adata(adata, output_path=output_path),
            output_path=output_path,
            **kwargs,
        )

    def _get_tree_results(self, tree_level_results: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Return tree-level results list (from arg or init); validate non-empty."""
        res_list = tree_level_results or self._tree
        if not res_list:
            raise ValueError("TreeLevelAnalysis requires tree_level_results (from init or argument).")
        return res_list

    def run_alluvial_per_domain(self, tree_level_results: Optional[List[Dict[str, Any]]] = None, **overrides) -> None:
        """
        Alluvial plot per spatial domain pair (src→tgt) across hierarchy levels.

        Columns correspond to LR hierarchy levels; flow width encodes aggregated
        communication strength for each LR leaf type within that domain pair.
        """
        self._log("run_alluvial_per_domain")
        res_list = self._get_tree_results(tree_level_results)
        p = self._p(**overrides)
        # hierarchy_dict is shared across levels; take from the first result
        hierarchy_dict = res_list[0].get("hierarchy_dict")
        if hierarchy_dict is None:
            raise ValueError("hierarchy_dict not found in tree_level_results; ensure evaluator saved it.")
        plot_alluvial_and_icicle_per_domain(
            tree_level_results=res_list,
            hierarchy_dict=hierarchy_dict,
            domain_key=p.get("domain_key", "domain"),
            domain_path=Path(p["domain_path"]) if p.get("domain_path") else None,
            domain_file_cell_id_column=p.get("domain_file_cell_id_column", "cell_id"),
            domain_file_domain_column=p.get("domain_file_domain_column", "cluster"),
            threshold=p.get("threshold", self._DEFAULTS["threshold"]),
            min_width_fraction=p.get("alluvial_min_width_fraction", 0.01),
            figsize=p.get("figsize", (8.0, 5.0)),
            dpi=p.get("dpi", 600),
            base_output_dir=self.output_path,
            show_title=False,
        )

    def run_alluvial_per_cell_type_pair(
        self,
        tree_level_results: Optional[List[Dict[str, Any]]] = None,
        **overrides,
    ) -> None:
        """
        Alluvial plot per cell-type pair (src→tgt) across hierarchy levels.

        Columns correspond to LR hierarchy levels; flow width encodes aggregated
        (normalized) communication strength for LR leaf types between that cell-type pair.
        """
        self._log("run_alluvial_per_cell_type_pair")
        res_list = self._get_tree_results(tree_level_results)
        p = self._p(**overrides)
        hierarchy_dict = res_list[0].get("hierarchy_dict")
        if hierarchy_dict is None:
            raise ValueError("hierarchy_dict not found in tree_level_results; ensure evaluator saved it.")
        plot_alluvial_and_icicle_per_cell_type_pair(
            tree_level_results=res_list,
            hierarchy_dict=hierarchy_dict,
            cell_type_key=p.get("cell_type_key", "cell_type"),
            threshold=p.get("threshold", self._DEFAULTS["threshold"]),
            min_count_threshold=p.get("min_count_threshold", self._DEFAULTS["min_count_threshold"]),
            min_width_fraction=p.get("alluvial_min_width_fraction", 0.01),
            figsize=p.get("figsize", (8.0, 5.0)),
            dpi=p.get("dpi", 600),
            base_output_dir=self.output_path,
            show_title=p.get("alluvial_show_title", True),
        )


class TimeSequenceAnalysis:
    """Time-sequence analysis: alluvial plots and hierarchy-node bubble plots."""

    _DEFAULTS = {
        "threshold": 0.0,
        "recompute": True,
        "region_lr_map": None,
        "spatial_root": None,
        "cell_type_key": "cell_type",
        "top_n_cell_types": 10,
        "merge_celltypes_by_colon_prefix": True,
        "annotation_filter": None,
        "lr_filter": None,
        "strong_node_ks": [5, 10, 15],
        "font_size": None,
        "fig_format": "png",
    }

    def __init__(
        self,
        stages: List[str],
        raw_path: Path,
        spatial_root: Path,
        output_path: Path,
        **kwargs,
    ):
        self.stages = stages
        self.raw_path = raw_path
        self.spatial_root = spatial_root
        self.output_path = Path(output_path) if output_path else None

        if self.output_path:
            self.output_path.mkdir(parents=True, exist_ok=True)
        self._params = {**self._DEFAULTS, **{k: v for k, v in kwargs.items() if v is not None}}

    def count_cell_number(self, **overrides) -> None:
        """Delegate to time-series utility to count cells and plot over stages."""
        from utils.time.cell_number_over_stages import count_cell_number_over_stages

        if self.output_path is None:
            raise ValueError("TimeSequenceAnalysis requires output_path for saving figures.")
        p = _merge(self._params, overrides if overrides else None)
        recompute = p.get("recompute", self._DEFAULTS["recompute"])

        print(f"[TimeSequenceAnalysis] Counting cell number over stages...")
        count_cell_number_over_stages(
            stages=self.stages,
            raw_root=self.raw_path,
            output_dir=self.output_path / "cell_number_over_stages",
            recompute=recompute,
            annotation_filter=p.get("annotation_filter"),
            lr_filter=p.get("lr_filter"),
            font_size=p.get("font_size"),
            fig_format=p.get("fig_format", self._DEFAULTS["fig_format"]),
        )
        print(f"[TimeSequenceAnalysis] Cell number over stages counted and plotted.")

    def count_edge_number(self, **overrides) -> None:
        """Delegate to time-series utility to count edges and plot over stages."""
        from utils.time.edge_number_over_stages import count_edge_number_over_stages

        if self.output_path is None:
            raise ValueError("TimeSequenceAnalysis requires output_path for saving figures.")
        p = _merge(self._params, overrides if overrides else None)
        threshold = p.get("threshold", self._DEFAULTS["threshold"])
        recompute = p.get("recompute", self._DEFAULTS["recompute"])

        count_edge_number_over_stages(
            stages=self.stages,
            raw_root=self.raw_path,
            output_dir=self.output_path / "edge_number_over_stages",
            threshold=threshold,
            recompute=recompute,
            annotation_filter=p.get("annotation_filter"),
            lr_filter=p.get("lr_filter"),
            font_size=p.get("font_size"),
            fig_format=p.get("fig_format", self._DEFAULTS["fig_format"]),
        )
    
    def plot_ccdf_degree_strength(self, **overrides) -> None:
        """
        Compute and plot CCDF curves of degree and strength for CCI networks,
        with optional caching of the underlying CCDF data.
        """
        import json
        import pandas as pd

        if self.output_path is None:
            raise ValueError("TimeSequenceAnalysis requires output_path for saving figures.")

        p = _merge(self._params, overrides if overrides else None)
        threshold = p.get("threshold", self._DEFAULTS["threshold"])
        recompute = p.get("recompute", self._DEFAULTS["recompute"])
        region_lr_map = p.get("region_lr_map")

        # Compute and plot CCDF; the utility handles both cache and recompute modes.
        compute_save_and_plot_ccdf(
            cci_root=self.raw_path,
            stages=self.stages,
            region_lr_map=region_lr_map,
            threshold=threshold,
            output_dir=self.output_path / "ccdf_degree_strength",
            recompute=recompute,
            verbose=True,
            font_size=p.get("font_size"),
            fig_format=p.get("fig_format", self._DEFAULTS["fig_format"]),
        )
    
    def plot_efficiency_metrics(self, **overrides) -> None:
        """
            Compute and plot global efficiency, average shortest path, and modularity
            per (organ, stage) with caching.
        """
        if self.output_path is None:
            raise ValueError("TimeSequenceAnalysis requires output_path for saving figures.")

        p = _merge(self._params, overrides if overrides else None)
        threshold = p.get("threshold", self._DEFAULTS["threshold"])
        recompute = p.get("recompute", self._DEFAULTS["recompute"])
        region_lr_map = p.get("region_lr_map")

        compute_save_and_plot_efficiency(
            cci_root=self.raw_path,
            spatial_root=self.spatial_root,
            stages=self.stages,
            region_lr_map=region_lr_map,
            threshold=threshold,
            output_dir=self.output_path / "efficiency_metrics",
            recompute=recompute,
            verbose=True,
            font_size=p.get("font_size"),
            fig_format=p.get("fig_format", self._DEFAULTS["fig_format"]),
        )

    def plot_strength_vs_distance_over_stages(self, **overrides) -> None:
        """
        For each organ, plot communication strength versus spatial distance across stages.

        - One figure per organ.
        - Within each figure, one line per stage for the organ-specific LR pair
          provided in `region_lr_map`.
        - Supports caching via CSV + JSON meta, and can be forced to recompute
          by setting `recompute=True`.
        """
        if self.output_path is None:
            raise ValueError("TimeSequenceAnalysis requires output_path for saving figures.")

        p = _merge(self._params, overrides if overrides else None)
        threshold = p.get("threshold", self._DEFAULTS["threshold"])
        recompute = p.get("recompute", self._DEFAULTS["recompute"])
        region_lr_map = p.get("region_lr_map")

        compute_save_and_plot_strength_vs_distance(
            cci_root=self.raw_path,
            spatial_root=self.spatial_root,
            stages=self.stages,
            region_lr_map=region_lr_map,
            threshold=threshold,
            output_dir=self.output_path / "strength_vs_distance_over_stages",
            recompute=recompute,
            verbose=True,
            font_size=p.get("font_size"),
            fig_format=p.get("fig_format", self._DEFAULTS["fig_format"]),
        )

    def plot_celltype_strength_bars(self, **overrides) -> None:
        """
        Plot stacked bar charts of cell-type communication strength across stages.

        - One figure per organ.
        - Within each figure, bars correspond to stages and stacks to cell types.
        - Uses region_lr_map to select the LR pair per organ, and cell types are
          read from the spatial h5ad files via `cell_type_key`.
        - Supports caching via CSV + JSON meta, and can be forced to recompute
          by setting `recompute=True`.
        """
        if self.output_path is None:
            raise ValueError("TimeSequenceAnalysis requires output_path for saving figures.")

        p = _merge(self._params, overrides if overrides else None)
        threshold = p.get("threshold", self._DEFAULTS["threshold"])
        recompute = p.get("recompute", self._DEFAULTS["recompute"])
        region_lr_map = p.get("region_lr_map")
        cell_type_key = p.get("cell_type_key", self._DEFAULTS["cell_type_key"])
        top_n = p.get("top_n_cell_types", self._DEFAULTS["top_n_cell_types"])
        merge_celltypes_by_colon_prefix = p.get(
            "merge_celltypes_by_colon_prefix",
            self._DEFAULTS["merge_celltypes_by_colon_prefix"],
        )

        if not region_lr_map:
            raise ValueError(
                "plot_celltype_strength_bars requires 'region_lr_map' mapping organ -> LR pair."
            )

        compute_save_and_plot_celltype_strength_over_stages(
            cci_root=self.raw_path,
            spatial_root=self.spatial_root,
            stages=self.stages,
            region_lr_map=region_lr_map,
            threshold=threshold,
            output_dir=self.output_path / "celltype_strength_over_stages",
            cell_type_key=cell_type_key,
            top_n=top_n,
            merge_celltypes_by_colon_prefix=merge_celltypes_by_colon_prefix,
            recompute=recompute,
            verbose=True,
            font_size=p.get("font_size"),
            fig_format=p.get("fig_format", self._DEFAULTS["fig_format"]),
        )

    def plot_graph_density_over_stages(self, **overrides) -> None:
        """
        Plot graph density curves over stages for selected LR pairs.

        - One figure per LR pair (treated as "annotation").
        - Within each figure, each organ is a line over stages.
        - Supports caching via CSV + JSON meta and can be forced to recompute
          by setting `recompute=True`.

        Filters:
            - annotation_filter: optional list of organs to include (e.g. ["Brain", "Liver"]).
            - lr_filter: optional list of LR pairs (CSV stems) to include; if None, include all.
        """
        if self.output_path is None:
            raise ValueError("TimeSequenceAnalysis requires output_path for saving figures.")

        p = _merge(self._params, overrides if overrides else None)
        threshold = p.get("threshold", self._DEFAULTS["threshold"])
        recompute = p.get("recompute", self._DEFAULTS["recompute"])
        annotation_filter = p.get("annotation_filter")
        lr_filter = p.get("lr_filter")

        compute_save_and_plot_density_over_stages(
            stages=self.stages,
            raw_root=self.raw_path,
            output_dir=self.output_path / "graph_density_over_stages",
            threshold=threshold,
            annotation_filter=annotation_filter,
            lr_filter=lr_filter,
            recompute=recompute,
            font_size=p.get("font_size"),
            fig_format=p.get("fig_format", self._DEFAULTS["fig_format"]),
        )

    def plot_strong_nodes_over_stages(self, **overrides) -> None:
        """
        Plot strong-node contribution curves over stages for each organ.

        - Uses region_lr_map to select a single LR pair per organ.
        - For each (stage, organ), aggregates node strengths and computes the
          fraction of total strength contributed by the top-K strongest nodes.
        - By default, K in {5, 10, 15}, producing curves such as:
              Brain Top 5 / 10 / 15 and Liver Top 5 / 10 / 15.
        - Supports caching via CSV + JSON meta and can be forced to recompute
          by setting `recompute=True`.
        """
        if self.output_path is None:
            raise ValueError("TimeSequenceAnalysis requires output_path for saving figures.")

        p = _merge(self._params, overrides if overrides else None)
        threshold = p.get("threshold", self._DEFAULTS["threshold"])
        recompute = p.get("recompute", self._DEFAULTS["recompute"])
        region_lr_map = p.get("region_lr_map")
        ks = p.get("strong_node_ks", self._DEFAULTS["strong_node_ks"])

        if not region_lr_map:
            raise ValueError(
                "plot_strong_nodes_over_stages requires 'region_lr_map' mapping organ -> LR pair."
            )

        compute_save_and_plot_strong_nodes_over_stages(
            cci_root=self.raw_path,
            stages=self.stages,
            region_lr_map=region_lr_map,
            threshold=threshold,
            output_dir=self.output_path / "strong_nodes_over_stages",
            ks=ks,
            recompute=recompute,
            verbose=True,
            font_size=p.get("font_size"),
            fig_format=p.get("fig_format", self._DEFAULTS["fig_format"]),
        )



