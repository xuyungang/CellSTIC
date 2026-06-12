from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import csv
import numpy as np
from anndata import AnnData

from .aggregated_heatmap_viz import AggregatedHeatmapVisualizer
from .alluvial_viz import (
    _build_leaf_to_group_map,
    _sanitize_filename as _sanitize_filename_alluvial,
    _sorted_level_keys,
)
from .domain_viz import load_domain_from_csv
from .icicle_viz import IcicleVisualizer as RadialAlluvialVisualizer
from .sender_receiver_stacked_bar_viz import SenderReceiverStackedBarVisualizer


def _compute_leaf_strengths_by_group_pair(
    *,
    pos_edge_probs_np: np.ndarray,
    edge_type_map: Dict[str, int],
    labels: np.ndarray,
    threshold: float,
    min_count_threshold: Optional[int] = None,
    normalize_per_lr: bool = False,
) -> Tuple[Dict[Tuple[str, str], Dict[str, float]], List[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate leaf-channel strengths into (group_src, group_tgt) pairs.

    Returns:
      strengths_per_pair, leaf_names, group_names, agg_matrix, counts
      - agg_matrix shape: (n_group, n_group, n_leaf)
      - counts shape: (n_group, n_group, n_leaf)
    """
    agg = AggregatedHeatmapVisualizer()
    _, group_names_arr = agg.build_group_onehot(labels)
    group_names = np.asarray(group_names_arr, dtype=str)
    leaf_names = sorted(edge_type_map.keys(), key=lambda k: edge_type_map[k])

    n_g = len(group_names)
    n_lr = len(leaf_names)
    agg_matrix = np.zeros((n_g, n_g, n_lr), dtype=np.float64)
    counts = np.zeros((n_g, n_g, n_lr), dtype=np.float64)

    thr = float(threshold)
    for li, leaf_name in enumerate(leaf_names):
        ch_idx = int(edge_type_map[leaf_name])
        if ch_idx < 0 or ch_idx >= pos_edge_probs_np.shape[2]:
            continue

        g = np.asarray(pos_edge_probs_np[:, :, ch_idx], dtype=np.float64)
        g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(g, 0.0)

        if min_count_threshold is None:
            # Domain path: direct strength aggregation.
            if thr > 0:
                g = np.where(g >= thr, g, 0.0)
            s_mat, _ = agg.compute_domain_domain_matrix(g, labels)
            agg_matrix[:, :, li] = s_mat
            counts[:, :, li] = (s_mat > 0).astype(np.float64)
            continue

        # Cell-type path: count thresholded edges, then apply min_count filter.
        if thr > 0.0:
            mask = g > thr
        else:
            mask = g > 0.0
        c_mat, _ = agg.compute_domain_domain_matrix(mask.astype(np.float64), labels)
        counts[:, :, li] = c_mat

        g_thr = np.where(mask, g, 0.0)
        s_mat, _ = agg.compute_domain_domain_matrix(g_thr, labels)
        s_mat = np.where(c_mat >= float(min_count_threshold), s_mat, 0.0)

        if normalize_per_lr:
            max_strength = float(np.max(s_mat)) if s_mat.size else 0.0
            if max_strength > 0:
                s_mat = s_mat / max_strength
        agg_matrix[:, :, li] = s_mat

    strengths_per_pair: Dict[Tuple[str, str], Dict[str, float]] = {}
    for i in range(n_g):
        for j in range(n_g):
            vals = agg_matrix[i, j, :]
            if float(np.sum(vals)) <= 0:
                continue
            pair = (str(group_names[i]), str(group_names[j]))
            strengths_per_pair[pair] = {
                leaf_names[k]: float(vals[k]) for k in range(n_lr)
            }

    return strengths_per_pair, leaf_names, group_names, agg_matrix, counts


def _compute_leaf_strengths_per_domain_pair(
    tree_level_results: List[Dict[str, Any]],
    *,
    domain_key: str,
    domain_path: Optional[Path],
    domain_file_cell_id_column: str,
    domain_file_domain_column: str,
    threshold: float,
) -> Tuple[Dict[Tuple[str, str], Dict[str, float]], List[str], Path, AnnData]:
    if not tree_level_results:
        raise ValueError("tree_level_results is empty; run evaluator.evaluate_ccc_precision_tree first.")

    leaf_result = max(tree_level_results, key=lambda r: int(r.get("level_num", 0)))
    adata: AnnData = leaf_result["adata"]
    pos_edge_probs_np: np.ndarray = leaf_result["pos_edge_probs_np"]
    edge_type_map: Dict[str, int] = leaf_result["edge_type_map"]

    if domain_path is not None:
        load_domain_from_csv(
            adata,
            domain_path=domain_path,
            cell_id_column=domain_file_cell_id_column,
            domain_column=domain_file_domain_column,
            domain_obs_key=domain_key,
        )
    if domain_key not in adata.obs:
        raise ValueError(f"domain_key '{domain_key}' not found in adata.obs; set domain_path or ensure column exists.")

    domains = adata.obs[domain_key].astype(str).to_numpy()
    leaf_strengths_per_pair, leaf_names, _dom_names, _mat, _counts = _compute_leaf_strengths_by_group_pair(
        pos_edge_probs_np=pos_edge_probs_np,
        edge_type_map=edge_type_map,
        labels=domains,
        threshold=threshold,
        min_count_threshold=None,
        normalize_per_lr=False,
    )

    base_dir = Path(leaf_result["output_dir"]).parent
    return leaf_strengths_per_pair, leaf_names, base_dir, adata


def _compute_leaf_strengths_per_cell_type_pair(
    tree_level_results: List[Dict[str, Any]],
    *,
    cell_type_key: str,
    threshold: float,
    min_count_threshold: int,
) -> Tuple[Dict[Tuple[str, str], Dict[str, float]], List[str], Path]:
    if not tree_level_results:
        raise ValueError("tree_level_results is empty; run evaluator.evaluate_ccc_precision_tree first.")

    leaf_result = max(tree_level_results, key=lambda r: int(r.get("level_num", 0)))
    adata: AnnData = leaf_result["adata"]
    pos_edge_probs_np: np.ndarray = leaf_result["pos_edge_probs_np"]
    edge_type_map: Dict[str, int] = leaf_result["edge_type_map"]

    if cell_type_key not in adata.obs:
        raise ValueError(f"cell_type_key '{cell_type_key}' not found in adata.obs.")

    cell_types = adata.obs[cell_type_key].astype(str).to_numpy()
    unique_cell_types_arr = AggregatedHeatmapVisualizer().build_group_onehot(cell_types)[1]
    unique_cell_types: List[str] = [str(x) for x in np.asarray(unique_cell_types_arr, dtype=str).tolist()]

    if len(unique_cell_types) == 0:
        print(
            "[alluvial][cell_type_pair] Skip: no valid cell types after unknown filtering. "
            f"cell_type_key={cell_type_key!r}."
        )
        return {}, [], Path(leaf_result["output_dir"]).parent

    strengths_per_pair, leaf_names, _ct_names, ct_matrix, counts = _compute_leaf_strengths_by_group_pair(
        pos_edge_probs_np=pos_edge_probs_np,
        edge_type_map=edge_type_map,
        labels=cell_types,
        threshold=threshold,
        min_count_threshold=min_count_threshold,
        normalize_per_lr=True,
    )

    base_dir = Path(leaf_result["output_dir"]).parent

    if not strengths_per_pair:
        # Diagnostic log: matrix exists but everything was zeroed out by threshold/min_count rules.
        nonzero = int(np.count_nonzero(ct_matrix > 0))
        print(
            "[alluvial][cell_type_pair] Skip: no (src→tgt) cell-type pairs with total>0 after filtering. "
            f"cell_type_key={cell_type_key!r}, threshold={threshold}, min_count_threshold={min_count_threshold}, "
            f"ct_matrix_shape={tuple(ct_matrix.shape)}, nonzero_entries={nonzero}."
        )
        if counts is not None and isinstance(counts, np.ndarray) and counts.size > 0:
            cmax = float(np.nanmax(counts))
            # How many (src,tgt,lr) positions met the min_count_threshold?
            passed = int(np.sum(counts >= float(min_count_threshold)))
            total_pos = int(counts.size)
            print(
                "[alluvial][cell_type_pair] counts stats: "
                f"shape={tuple(counts.shape)}, max={cmax:.0f}, "
                f"passed(min_count_threshold)={passed}/{total_pos}."
            )
    return strengths_per_pair, leaf_names, base_dir


def _render_icicle_pair_plot(
    *,
    radial: RadialAlluvialVisualizer,
    strength_dict: Dict[str, float],
    leaf_names: List[str],
    level_keys: List[str],
    leaf_to_group: Dict[str, Dict[str, str]],
    title: str,
    save_path: Path,
    min_width_fraction: float,
    figsize: Tuple[float, float],
    dpi: int,
    high_contrast: bool,
    stable_order_level_keys: Optional[Iterable[str]],
) -> None:
    """Render radial icicle-style hierarchy plot for one pair."""
    radial._plot_alluvial_core(  # type: ignore[attr-defined]
        leaf_strengths=strength_dict,
        leaf_names=leaf_names,
        level_keys=level_keys,
        leaf_to_group=leaf_to_group,
        title=title,
        save_path=save_path.with_name(save_path.stem + "_icicle.svg"),
        min_width_fraction=min_width_fraction,
        figsize=figsize,
        dpi=dpi,
        high_contrast=high_contrast,
        stable_order_level_keys=stable_order_level_keys,
    )


def _plot_alluvial_pair_collection(
    *,
    strengths_per_pair: Dict[Tuple[str, str], Dict[str, float]],
    leaf_names: List[str],
    level_keys: List[str],
    leaf_to_group: Dict[str, Dict[str, str]],
    out_dir: Path,
    show_title: bool,
    prefix: str,
    min_width_fraction: float,
    figsize: Tuple[float, float],
    dpi: int,
    high_contrast: bool,
    stable_order_level_keys: Optional[Iterable[str]],
    radial: RadialAlluvialVisualizer,
) -> None:
    """Render icicle pair plots in batch."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for (src, tgt), strength_dict in strengths_per_pair.items():
        total = sum(strength_dict.values())
        if total <= 0:
            continue
        title = f"{src} → {tgt}" if show_title else ""
        fname = f"{prefix}_{_sanitize_filename_alluvial(src)}_to_{_sanitize_filename_alluvial(tgt)}.svg"
        save_path = out_dir / fname
        _render_icicle_pair_plot(
            radial=radial,
            strength_dict=strength_dict,
            leaf_names=leaf_names,
            level_keys=level_keys,
            leaf_to_group=leaf_to_group,
            title=title,
            save_path=save_path,
            min_width_fraction=min_width_fraction,
            figsize=figsize,
            dpi=dpi,
            high_contrast=high_contrast,
            stable_order_level_keys=stable_order_level_keys,
        )


def _export_level_code_mapping(
    *,
    base_dir: Path,
    level_keys: List[str],
    leaf_to_group: Dict[str, Dict[str, str]],
    leaf_names: List[str],
    filename: str,
) -> None:
    """Export simple level code mapping: L{k}-{i} -> group name."""
    codebook_path = base_dir / "alluvial" / filename
    codebook_path.parent.mkdir(parents=True, exist_ok=True)

    # Derive groups per level from leaf_to_group using leaf_names order.
    with open(codebook_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["level", "code", "node_name"])

        n_levels = len(level_keys)
        for lvl_idx, level_key in enumerate(level_keys):
            # Skip root (level_0) and last level (leaf-level) to match code usage.
            if not (0 < lvl_idx < n_levels - 1):
                continue

            mapping = leaf_to_group[level_key]
            groups: List[str] = []
            seen = set()
            for leaf in leaf_names:
                g = str(mapping.get(leaf, leaf))
                if g not in seen:
                    seen.add(g)
                    groups.append(g)

            for j, g in enumerate(groups, start=1):
                code = f"L{lvl_idx + 1}-{j}"
                writer.writerow([f"Level {lvl_idx + 1}", code, g])


def plot_alluvial_and_icicle_per_domain(
    *,
    tree_level_results: List[Dict[str, Any]],
    hierarchy_dict: Mapping[str, Any],
    domain_key: str = "domain",
    domain_path: Optional[Path] = None,
    domain_file_cell_id_column: str = "cell_id",
    domain_file_domain_column: str = "cluster",
    threshold: float = 0.7,
    min_width_fraction: float = 0.01,
    figsize: Tuple[float, float] = (8.0, 5.0),
    dpi: int = 300,
    base_output_dir: Optional[Union[str, Path]] = None,
    show_title: bool = False,
    stable_order_level_keys: Optional[Iterable[str]] = ("level_2", "level_3"),
    high_contrast: bool = True,
) -> None:
    """
    Render radial icicle-style hierarchy plots for domain–domain communication.
    """
    radial = RadialAlluvialVisualizer()

    leaf_strengths_per_pair, leaf_names, default_base_dir, _adata = _compute_leaf_strengths_per_domain_pair(
        tree_level_results,
        domain_key=domain_key,
        domain_path=domain_path,
        domain_file_cell_id_column=domain_file_cell_id_column,
        domain_file_domain_column=domain_file_domain_column,
        threshold=threshold,
    )

    base_dir = Path(base_output_dir) if base_output_dir is not None else default_base_dir
    level_keys = _sorted_level_keys(hierarchy_dict)
    leaf_to_group = _build_leaf_to_group_map(hierarchy_dict, leaf_names, level_keys)

    _export_level_code_mapping(
        base_dir=base_dir,
        level_keys=level_keys,
        leaf_to_group=leaf_to_group,
        leaf_names=leaf_names,
        filename="domain_level_codebook.tsv",
    )

    # Keep domain skip diagnostics
    positive_pairs = {
        (src, tgt): s for (src, tgt), s in leaf_strengths_per_pair.items() if sum(s.values()) > 0
    }
    for (src_dom, tgt_dom), s in leaf_strengths_per_pair.items():
        if sum(s.values()) <= 0:
            print(
                "[alluvial][domain_pair] Skip: no positive strength after filtering. "
                f"pair=({src_dom}->{tgt_dom}), threshold={threshold}."
            )

    _plot_alluvial_pair_collection(
        strengths_per_pair=positive_pairs,
        leaf_names=leaf_names,
        level_keys=level_keys,
        leaf_to_group=leaf_to_group,
        out_dir=base_dir / "alluvial" / "domain_pairs",
        show_title=show_title,
        prefix="alluvial_domain",
        min_width_fraction=min_width_fraction,
        figsize=figsize,
        dpi=dpi,
        high_contrast=high_contrast,
        stable_order_level_keys=stable_order_level_keys,
        radial=radial,
    )


def plot_alluvial_and_icicle_per_cell_type_pair(
    *,
    tree_level_results: List[Dict[str, Any]],
    hierarchy_dict: Mapping[str, Any],
    cell_type_key: str = "cell_type",
    threshold: float = 0.7,
    min_count_threshold: int = 500,
    min_width_fraction: float = 0.01,
    figsize: Tuple[float, float] = (8.0, 5.0),
    dpi: int = 300,
    base_output_dir: Optional[Union[str, Path]] = None,
    show_title: bool = True,
    stable_order_level_keys: Optional[Iterable[str]] = ("level_2", "level_3"),
    high_contrast: bool = True,
) -> None:
    """
    Render radial icicle-style hierarchy plots for each (source, target) cell-type pair.
    """
    radial = RadialAlluvialVisualizer()

    strengths_per_pair, leaf_names, default_base_dir = _compute_leaf_strengths_per_cell_type_pair(
        tree_level_results,
        cell_type_key=cell_type_key,
        threshold=threshold,
        min_count_threshold=min_count_threshold,
    )
    if not strengths_per_pair or not leaf_names:
        return

    base_dir = Path(base_output_dir) if base_output_dir is not None else default_base_dir
    level_keys = _sorted_level_keys(hierarchy_dict)
    leaf_to_group = _build_leaf_to_group_map(hierarchy_dict, leaf_names, level_keys)

    _export_level_code_mapping(
        base_dir=base_dir,
        level_keys=level_keys,
        leaf_to_group=leaf_to_group,
        leaf_names=leaf_names,
        filename="cell_type_level_codebook.tsv",
    )

    _plot_alluvial_pair_collection(
        strengths_per_pair=strengths_per_pair,
        leaf_names=leaf_names,
        level_keys=level_keys,
        leaf_to_group=leaf_to_group,
        out_dir=base_dir / "alluvial" / "cell_type_pairs",
        show_title=show_title,
        prefix="alluvial_ct",
        min_width_fraction=min_width_fraction,
        figsize=figsize,
        dpi=dpi,
        high_contrast=high_contrast,
        stable_order_level_keys=stable_order_level_keys,
        radial=radial,
    )

    # Additionally generate sender/receiver stacked bars aligned with tree levels
    run_stacked_bars_for_tree(
        tree_level_results,
        output_path=base_dir,
        cell_type_key=cell_type_key,
        threshold=threshold,
        stacked_bar_top_n=20,
    )


def run_stacked_bars_for_tree(
    tree_level_results: List[Dict[str, Any]],
    *,
    output_path: Optional[Union[str, Path]],
    cell_type_key: str = "cell_type",
    threshold: float = 0.7,
    stacked_bar_top_n: int = 20,
) -> None:
    """
    Generate sender/receiver stacked bar plots for each non-root hierarchy level
    in the cell-type space, aligned with the LR hierarchy tree.

    - Intermediate levels: use compact virtual node codes (e.g. L2-1) ordered by
      channel index, consistent with the alluvial legend.
    - Last level: use real leaf LR names (unchanged).
    - Outputs are written under:
      <output_root>/alluvial/cell_type_bar/lr_pair_stacked_bar_level_<k>.svg
    """
    if not tree_level_results:
        return

    sorted_levels = sorted(tree_level_results, key=lambda r: int(r.get("level_num", 0)))
    if len(sorted_levels) <= 1:
        return

    hierarchy_dict = sorted_levels[0].get("hierarchy_dict") or {}
    level_keys = [k for k in hierarchy_dict.keys() if k.startswith("level_")]
    level_keys.sort(key=lambda x: int(x.split("_")[1]))

    figsize_map: Dict[int, Tuple[float, float]] = {
        2: (1.4, 6),
        3: (4, 6),
        4: (12, 6),
    }

    for level_result in sorted_levels[1:]:
        level_num = int(level_result.get("level_num", 0))
        adata = level_result.get("adata")
        pos_edge_probs_np = level_result.get("pos_edge_probs_np")
        edge_type_map = level_result.get("edge_type_map")
        level_output_dir = level_result.get("output_dir")
        if adata is None or pos_edge_probs_np is None or edge_type_map is None or level_output_dir is None:
            continue

        figsize = figsize_map.get(level_num, (6, 6))
        if output_path is not None:
            root = Path(output_path)
        else:
            root = Path(level_output_dir).parent
        analysis_output_root = root / "alluvial" / "cell_type_bar"
        analysis_output_root.mkdir(parents=True, exist_ok=True)

        level_key = f"level_{level_num}"
        try:
            level_idx = level_keys.index(level_key)
        except ValueError:
            level_idx = -1

        if 0 < level_idx < len(level_keys) - 1:
            # Intermediate level: build codes L{level_idx+1}-{i+1} ordered by channel index
            ordered_items = sorted(edge_type_map.items(), key=lambda kv: kv[1])
            coded_edge_type_map = {
                f"L{level_idx + 1}-{i + 1}": ch_idx
                for i, (_, ch_idx) in enumerate(ordered_items)
            }
            is_last_level = False
        else:
            # Root or last level: keep original names (root is skipped anyway)
            coded_edge_type_map = edge_type_map
            is_last_level = level_idx == len(level_keys) - 1

        save_path = analysis_output_root / f"lr_pair_stacked_bar_level_{level_num}.svg"
        SenderReceiverStackedBarVisualizer().plot(
            graph=pos_edge_probs_np,
            edge_type_map=coded_edge_type_map,
            adata=adata,
            cell_type_key=cell_type_key,
            lr_filter=None,
            top_n=stacked_bar_top_n,
            threshold=threshold,
            order_by="edge_index",
            save_path=str(save_path),
            figsize=figsize,
            dpi=300,
            break_lr_labels=is_last_level,
        )
