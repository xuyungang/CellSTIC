"""Spatially restricted ligand-receptor graph construction."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse

from .sparse_graph import SparseTypedGraph


class BaseGraphUtils:
    """Build sparse LR graphs from a distance-limited cell-pair candidate set."""

    @staticmethod
    def build_spatial_graph(
        spatial_coords: np.ndarray,
        gene_expression,
        genes: List[str],
        distance_threshold: float,
        expression_percentile: float,
        ligand_receptor_map: Dict,
        candidate_pairs: Optional[np.ndarray] = None,
    ) -> Tuple[SparseTypedGraph, Dict, SparseTypedGraph, SparseTypedGraph]:
        """Return sparse adjacency and expression-strength channels."""
        coords = np.asarray(spatial_coords)
        if coords.ndim != 2 or coords.shape[1] not in (2, 3):
            raise ValueError(
                "spatial_coords must have shape (n_cells, 2) or (n_cells, 3)"
            )
        if candidate_pairs is None:
            from model.train.edge_filter_utils import EdgeFilterUtils

            candidate_pairs = EdgeFilterUtils.spatial_candidate_pairs(
                coords, distance_threshold
            )
        candidate_pairs = np.asarray(candidate_pairs, dtype=np.int64).reshape(-1, 2)
        channel_values = BaseGraphUtils.filter_edges_with_strength(
            gene_expression,
            genes,
            ligand_receptor_map,
            expression_percentile,
            candidate_pairs=candidate_pairs,
        )

        pair_names = sorted(channel_values)
        edge_type_map = {name: index for index, name in enumerate(pair_names)}
        n_nodes = coords.shape[0]
        shape = (n_nodes, n_nodes)
        base_channels = []
        ligand_channels = []
        receptor_channels = []
        for pair_name in pair_names:
            rows, cols, ligand, receptor = channel_values[pair_name]
            coordinates = (rows, cols)
            base_channels.append(
                sparse.csr_matrix(
                    (np.ones(len(rows), dtype=np.float32), coordinates),
                    shape=shape,
                )
            )
            ligand_channels.append(
                sparse.csr_matrix((ligand, coordinates), shape=shape)
            )
            receptor_channels.append(
                sparse.csr_matrix((receptor, coordinates), shape=shape)
            )
        return (
            SparseTypedGraph.from_channels(base_channels, n_nodes),
            edge_type_map,
            SparseTypedGraph.from_channels(ligand_channels, n_nodes),
            SparseTypedGraph.from_channels(receptor_channels, n_nodes),
        )

    @staticmethod
    def filter_edges_with_strength(
        expression_matrix,
        genes: List[str],
        ligand_receptor_map: Dict,
        expression_percentile: float = 75,
        *,
        candidate_pairs: Optional[np.ndarray] = None,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Evaluate LR expression only for a supplied local candidate set.

        Omitting ``candidate_pairs`` is rejected deliberately: silently falling
        back to all ``n(n-1)`` pairs would reintroduce quadratic behavior.
        Ligands and every receptor-complex subunit must also have strictly
        positive raw expression; a zero percentile cutoff never admits zeros.
        """
        if not ligand_receptor_map:
            return {}
        if candidate_pairs is None:
            raise ValueError(
                "candidate_pairs is required; construct it with a spatial radius or kNN index"
            )
        pairs = np.asarray(candidate_pairs, dtype=np.int64).reshape(-1, 2)
        if len(pairs) == 0:
            return {}
        n_cells = expression_matrix.shape[0]
        if expression_matrix.ndim != 2:
            raise ValueError(
                f"expression_matrix must be 2D, got {expression_matrix.shape}"
            )
        if expression_matrix.shape[1] != len(genes):
            raise ValueError(
                "genes must contain one name per expression column, got "
                f"{len(genes)} names for {expression_matrix.shape[1]} columns"
            )
        if not 0 <= expression_percentile <= 100:
            raise ValueError("expression_percentile must be in [0, 100]")
        if pairs.min() < 0 or pairs.max() >= n_cells:
            raise ValueError("candidate_pairs contains an out-of-range cell index")

        gene_to_idx = {str(gene): idx for idx, gene in enumerate(genes)}
        expression_source = (
            expression_matrix.tocsc()
            if sparse.issparse(expression_matrix)
            else np.asarray(expression_matrix)
        )
        cache: OrderedDict[int, Tuple[np.ndarray, np.ndarray]] = OrderedDict()

        def expression_column(index: int) -> Tuple[np.ndarray, np.ndarray]:
            if index in cache:
                cache.move_to_end(index)
                return cache[index]
            column = expression_source[:, index]
            values = (
                column.toarray().reshape(-1)
                if sparse.issparse(column)
                else np.asarray(column).reshape(-1)
            )
            raw_values = values.astype(np.float32, copy=False)
            if not np.all(np.isfinite(raw_values)):
                raise ValueError(
                    f"Expression column {genes[index]!r} contains non-finite values"
                )
            low, high = float(raw_values.min()), float(raw_values.max())
            normalized = (
                (raw_values - low) / (high - low)
                if high > low
                else np.ones_like(raw_values)
            )
            cache[index] = (raw_values, normalized)
            if len(cache) > 32:
                cache.popitem(last=False)
            return raw_values, normalized

        src, dst = pairs[:, 0], pairs[:, 1]
        channels: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for ligand_name, receptor_specs in ligand_receptor_map.items():
            ligand_idx = gene_to_idx.get(str(ligand_name))
            if ligand_idx is None:
                continue
            if isinstance(receptor_specs, (str, bytes)):
                raise TypeError(
                    f"Receptors for ligand {ligand_name!r} must be a sequence, "
                    "not a string"
                )
            ligand_raw, ligand_values = expression_column(ligand_idx)
            ligand_expressed = ligand_raw > 0
            if not np.any(ligand_expressed):
                continue
            ligand_cutoff = np.percentile(ligand_values, expression_percentile)
            ligand_strength = ligand_values[src]
            ligand_positions = np.flatnonzero(
                ligand_expressed[src]
                & (ligand_strength >= ligand_cutoff)
                & (src != dst)
            )
            if len(ligand_positions) == 0:
                continue
            for receptor_spec in receptor_specs:
                receptor_names = [
                    name.strip()
                    for name in str(receptor_spec).split(":")
                    if name.strip()
                ]
                receptor_indices = [gene_to_idx.get(name) for name in receptor_names]
                if not receptor_indices or any(
                    index is None for index in receptor_indices
                ):
                    continue
                receptor_columns = [
                    expression_column(int(index)) for index in receptor_indices
                ]
                receptor_expressed = np.logical_and.reduce(
                    [raw > 0 for raw, _ in receptor_columns]
                )
                if not np.any(receptor_expressed):
                    continue
                receptor_values = np.mean(
                    np.stack([normalized for _, normalized in receptor_columns]),
                    axis=0,
                )
                receptor_cutoff = np.percentile(
                    receptor_values, expression_percentile
                )
                receptor_strength = receptor_values[dst[ligand_positions]]
                valid_positions = ligand_positions[
                    receptor_expressed[dst[ligand_positions]]
                    & (receptor_strength >= receptor_cutoff)
                ]
                if len(valid_positions) == 0:
                    continue
                pair_name = f"{ligand_name}:{receptor_spec}"
                if pair_name in channels:
                    raise ValueError(f"Duplicate ligand-receptor event {pair_name!r}")
                channels[pair_name] = (
                    src[valid_positions].astype(np.int64, copy=False),
                    dst[valid_positions].astype(np.int64, copy=False),
                    ligand_strength[valid_positions].astype(np.float32, copy=False),
                    receptor_values[dst[valid_positions]].astype(
                        np.float32, copy=False
                    ),
                )
        return channels
