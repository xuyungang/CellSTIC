"""Spatial graph construction with ligand-receptor edge types."""

import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import multiprocessing as mp

try:
    from scipy.sparse import issparse
except ImportError:
    def issparse(x):
        return False


class BaseGraphUtils:
    """Build spatial graph (n_cells, n_cells, n_edge_types) from distances, expression, and ligand-receptor map."""

    @staticmethod
    def build_spatial_graph(
        spatial_distances: np.ndarray,
        gene_expression: np.ndarray,
        genes: List[str],
        distance_threshold: float,
        expression_percentile: float,
        ligand_receptor_map: Dict,
    ) -> Tuple[np.ndarray, Dict, np.ndarray, np.ndarray]:
        """Return (base_graph_adj, edge_type_map, ligand_strength_adj, receptor_strength_adj).
        base_graph_adj is 0/1 (binary); ligand/receptor strength arrays are for edge feature construction.
        High expression is >= expression_percentile (e.g. 75 = 75th percentile)."""
        edges = BaseGraphUtils.filter_edges_with_strength(
            gene_expression, genes, ligand_receptor_map, expression_percentile
        )
        if issparse(spatial_distances):
            spatial_distances = spatial_distances.toarray()
        else:
            spatial_distances = np.asarray(spatial_distances)
        distance_mask = spatial_distances < distance_threshold
        edge_keys = [(i, j) for (i, j) in edges.keys() if distance_mask[i, j]]
        n_threads = max(1, min(mp.cpu_count(), len(edge_keys), 8))
        batch_size = max(1, len(edge_keys) // n_threads)
        edge_batches = []
        for i in range(0, len(edge_keys), batch_size):
            batch = [(idx, edge_keys[idx]) for idx in range(i, min(i + batch_size, len(edge_keys)))]
            edge_batches.append(batch)
        all_edges = []
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            future_to_batch = {
                executor.submit(BaseGraphUtils._process_edge_batch, batch, edges): batch
                for batch in edge_batches
            }
            for future in as_completed(future_to_batch):
                all_edges.extend(future.result())

        edges_by_source_type = defaultdict(list)
        for edge in all_edges:
            source = edge['source']
            pair_name = edge['pair_name']
            edges_by_source_type[(source, pair_name)].append(edge)

        filtered_edges = []
        for (source, pair_name), grp in edges_by_source_type.items():
            filtered_edges.extend(grp)
        edge_type_map = {}
        edge_type_counter = 0
        for edge in all_edges:
            pair_name = edge['pair_name']
            if pair_name not in edge_type_map:
                edge_type_map[pair_name] = edge_type_counter
                edge_type_counter += 1
        n_cells = spatial_distances.shape[0]
        n_types = len(edge_type_map)
        base_graph_adj = np.zeros((n_cells, n_cells, n_types))
        ligand_strength_adj = np.zeros((n_cells, n_cells, n_types))
        receptor_strength_adj = np.zeros((n_cells, n_cells, n_types))
        for edge in filtered_edges:
            src, dst = int(edge['source']), int(edge['target'])
            if src != dst and not distance_mask[src, dst]:
                continue
            type_idx = edge_type_map[edge['pair_name']]
            base_graph_adj[src, dst, type_idx] = 1
            ligand_strength_adj[src, dst, type_idx] = edge['ligand_strength']
            receptor_strength_adj[src, dst, type_idx] = edge['receptor_strength']
        return base_graph_adj, edge_type_map, ligand_strength_adj, receptor_strength_adj

    @staticmethod
    def filter_edges_with_strength(
        expression_matrix: np.ndarray,
        genes: List[str],
        ligand_receptor_map: Dict,
        expression_percentile: float = 75,
    ) -> Dict:
        """Return dict (cell_i, cell_j) -> list of ligand-receptor pair info.
        High expression is >= expression_percentile (default 75th, upper quartile) of that gene's values.
        Supports multiple receptors per pair: use 'RecA:RecB' in ligand_receptor_map; receptor strength is the mean of RecA and RecB. Does not support multiple ligands per pair (one ligand per entry)."""
        edges = {}

        if ligand_receptor_map is None:
            return edges
        if issparse(expression_matrix):
            expression_matrix = expression_matrix.copy()
            expression_matrix = expression_matrix.toarray()
        else:
            expression_matrix = expression_matrix.copy()
        col_min = expression_matrix.min(axis=0, keepdims=True)
        col_max = expression_matrix.max(axis=0, keepdims=True)
        col_range = col_max - col_min
        col_range[col_range == 0] = 1
        expression_matrix = (expression_matrix - col_min) / col_range
        gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
        valid_pairs = []
        pair_names = []
        ligand_indices = []
        receptor_indices_list = []
        all_ligands = []
        all_receptors = []
        for ligand, receptors in ligand_receptor_map.items():
            if ligand in gene_to_idx:
                ligand_idx = gene_to_idx[ligand]
                for receptor_str in receptors:
                    receptor_names = [r.strip() for r in receptor_str.split(':') if r.strip()]
                    receptor_idxs = []
                    all_receptors_exist = True
                    for receptor_name in receptor_names:
                        if receptor_name not in gene_to_idx:
                            all_receptors_exist = False
                            break
                        receptor_idxs.append(gene_to_idx[receptor_name])
                    
                    if all_receptors_exist and len(receptor_idxs) > 0:
                        valid_pairs.append((ligand_idx, receptor_idxs))
                        pair_names.append(f"{ligand}:{receptor_str}")
                        ligand_indices.append(ligand_idx)
                        receptor_indices_list.append(receptor_idxs)
                        all_ligands.append(ligand)
                        all_receptors.append(receptor_str)
        
        if not valid_pairs:
            return edges
        ligand_contributions = expression_matrix[:, ligand_indices]
        if issparse(ligand_contributions):
            ligand_contributions = ligand_contributions.toarray()
        else:
            ligand_contributions = np.asarray(ligand_contributions)
        n_pairs = len(valid_pairs)
        n_cells = expression_matrix.shape[0]
        receptor_contributions = np.zeros((n_cells, n_pairs))
        
        for pair_idx in range(n_pairs):
            receptor_idxs = receptor_indices_list[pair_idx]
            receptor_expr = expression_matrix[:, receptor_idxs]
            if issparse(receptor_expr):
                receptor_expr = receptor_expr.toarray()
            else:
                receptor_expr = np.asarray(receptor_expr)
            receptor_contributions[:, pair_idx] = np.mean(receptor_expr, axis=1)
        ligand_thresholds = np.percentile(ligand_contributions, expression_percentile, axis=0)
        receptor_thresholds = np.percentile(receptor_contributions, expression_percentile, axis=0)
        i_indices, j_indices = np.where(~np.eye(n_cells, dtype=bool))
        if len(i_indices) == 0:
            return edges
        ligand_expr_all = ligand_contributions[i_indices]
        receptor_expr_all = receptor_contributions[j_indices]
        valid_mask = (ligand_expr_all >= ligand_thresholds) & (receptor_expr_all >= receptor_thresholds)
        for idx in range(len(i_indices)):
            i, j = i_indices[idx], j_indices[idx]
            valid_pair_indices = np.where(valid_mask[idx])[0]
            if len(valid_pair_indices) > 0:
                edge_pairs = [
                    {
                        'ligand': all_ligands[pair_idx],
                        'receptor': all_receptors[pair_idx],
                        'pair_name': pair_names[pair_idx],
                        'ligand_strength': float(ligand_expr_all[idx, pair_idx]),
                        'receptor_strength': float(receptor_expr_all[idx, pair_idx]),
                    }
                    for pair_idx in valid_pair_indices
                ]
                edge_pairs.sort(key=lambda x: (x['ligand_strength'] + x['receptor_strength']) / 2, reverse=True)
                edges[(i, j)] = edge_pairs

        return edges

    @staticmethod
    def _process_edge_batch(edge_batch, edges):
        """Pass through edges with ligand_strength and receptor_strength (no distance/strength merge)."""
        batch_edges = []
        for idx, (i, j) in edge_batch:
            pairs = edges[(i, j)]
            batch_edges.extend([
                {
                    'source': i,
                    'target': j,
                    'pair_name': pair['pair_name'],
                    'ligand_strength': pair['ligand_strength'],
                    'receptor_strength': pair['receptor_strength'],
                }
                for pair in pairs
            ])
        return batch_edges
