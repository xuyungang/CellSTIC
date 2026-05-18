"""
Edge filtering utilities for CellSTIC.

This module provides utilities for filtering edge probabilities based on various constraints
such as spatial distance, gene expression, and cell annotations.
"""

import numpy as np
import torch
from typing import Union, List, Tuple, Dict, Optional
from scipy.spatial.distance import cdist
from anndata import AnnData


class EdgeFilterUtils:
    """
    Utilities for filtering edge probabilities based on various constraints.
    """

    @staticmethod
    def compute_spot_distance_threshold(
        spatial_coords_or_distances: Union[torch.Tensor, np.ndarray],
        n_spots: int = 4,
    ) -> float:
        """
        Compute distance threshold as n_spots * (median nearest-neighbor distance).
        Spot distance is defined as the distance between two nearest cells (per-cell
        nearest-neighbor distance); the threshold is n_spots times the minimum of these.

        Args:
            spatial_coords_or_distances: Either (n_cells, 2) spatial coordinates or
                (n_cells, n_cells) pairwise distance matrix.
            n_spots: Multiplier for median spot distance (default 4).

        Returns:
            Threshold distance (n_spots * median_nearest_neighbor_distance).
        """
        if hasattr(spatial_coords_or_distances, "toarray"):
            data = spatial_coords_or_distances.toarray()
        elif isinstance(spatial_coords_or_distances, torch.Tensor):
            data = spatial_coords_or_distances.detach().cpu().numpy()
        else:
            data = np.asarray(spatial_coords_or_distances)
        if data.ndim < 2 or (data.ndim == 2 and data.shape[0] < 2):
            return np.inf
        if data.ndim == 2 and data.shape[1] == 2:
            distances = cdist(data, data)
        elif data.ndim == 2 and data.shape[0] == data.shape[1]:
            distances = np.asarray(data, dtype=np.float64).copy()
        else:
            raise ValueError(
                "spatial_coords_or_distances must be (n, 2) coords or (n, n) distance matrix, got shape {}".format(
                    data.shape
                )
            )
        n = distances.shape[0]
        # Per-cell nearest-neighbor distance (excluding self)
        np.fill_diagonal(distances, np.inf)
        nearest_dist = np.nanmin(distances, axis=1)
        if np.any(~np.isfinite(nearest_dist)) or len(nearest_dist) == 0:
            return np.inf
        spot_distance = float(np.median(nearest_dist))
        return n_spots * spot_distance

    @staticmethod
    def apply_distance_constraint(
        pos_edge_probs: Union[torch.Tensor, np.ndarray],
        spatial_coords: torch.Tensor,
        n_spots: int = 10,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Apply distance constraint to edge probabilities: set edges beyond threshold to 0.
        Threshold is n_spots * median(nearest-neighbor distance) from spatial_coords (spot distance
        = distance between two nearest cells).

        Args:
            pos_edge_probs: Edge probability matrix (n_cells, n_cells, n_edge_types) or tensor
            spatial_coords: Spatial coordinates tensor (n_cells, 2)
            n_spots: Multiplier for median spot distance (default 10); use config.model.graph.n_spots to align with training.

        Returns:
            Constrained edge probabilities with same type as input
        """
        distance_threshold = EdgeFilterUtils.compute_spot_distance_threshold(spatial_coords, n_spots=n_spots)
        
        # Convert to numpy
        is_tensor = isinstance(pos_edge_probs, torch.Tensor)
        if is_tensor:
            pos_edge_probs_np = pos_edge_probs.detach().cpu().numpy()
        else:
            pos_edge_probs_np = np.asarray(pos_edge_probs)
        
        # Calculate distance matrix and create mask
        coords_np = spatial_coords.detach().cpu().numpy()
        distances = cdist(coords_np, coords_np)
        distance_mask = distances <= distance_threshold
        
        # Apply mask to all edge types
        pos_edge_probs_np = pos_edge_probs_np * distance_mask[:, :, np.newaxis]
        
        # Convert back to tensor if needed
        if is_tensor:
            pos_edge_probs = torch.tensor(pos_edge_probs_np, dtype=pos_edge_probs.dtype, device=pos_edge_probs.device)
        else:
            pos_edge_probs = pos_edge_probs_np
        
        print(f"Applied distance constraint ({distance_threshold}): {np.sum(distance_mask)}/{distance_mask.size} edges within threshold")
        
        return pos_edge_probs

    @staticmethod
    def apply_cell_type_constraints(
        base_graph_adj: np.ndarray,
        ligand_strength_adj: np.ndarray,
        receptor_strength_adj: np.ndarray,
        knn_per_modality: np.ndarray,
        edge_type_map: Dict[str, int],
        cell_types: Optional[np.ndarray],
        lr_pair_type_constraints: Optional[Dict[str, List[Tuple[int, int]]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply cell-type based constraints to CCC graphs (scmultisim-style):
        For each ligand–receptor pair (edge channel), keep only (sender_type, receiver_type)
        pairs that appear in lr_pair_type_constraints[pair_name].

        Args:
            base_graph_adj: (n_cells, n_cells, n_edge_types) binary/weighted adjacency
            ligand_strength_adj: same shape as base_graph_adj
            receptor_strength_adj: same shape as base_graph_adj
            knn_per_modality: (n_cells, n_cells, n_modalities) KNN_strength per modality
            edge_type_map: mapping "Ligand:Receptor" -> edge_type_idx
            cell_types: (n_cells,) integer cell type labels
            lr_pair_type_constraints: "Ligand:Receptor" -> list of (ct_sender, ct_receiver)
        """
        if cell_types is None:
            return base_graph_adj, ligand_strength_adj, receptor_strength_adj, knn_per_modality

        # Per-pair (sender_type, receiver_type) constraints from LR.csv (shared core)
        if lr_pair_type_constraints:
            # Log only once (on base_graph_adj); reuse core for strength tensors without extra logs
            base_graph_adj = EdgeFilterUtils._apply_pair_type_constraints_core(
                base_graph_adj, edge_type_map, cell_types, lr_pair_type_constraints, log=True
            )
            ligand_strength_adj = EdgeFilterUtils._apply_pair_type_constraints_core(
                ligand_strength_adj, edge_type_map, cell_types, lr_pair_type_constraints, log=False
            )
            receptor_strength_adj = EdgeFilterUtils._apply_pair_type_constraints_core(
                receptor_strength_adj, edge_type_map, cell_types, lr_pair_type_constraints, log=False
            )

        return base_graph_adj, ligand_strength_adj, receptor_strength_adj, knn_per_modality

    @staticmethod
    def _apply_pair_type_constraints_core(
        arr: np.ndarray,
        edge_type_map: Dict[str, int],
        cell_types: np.ndarray,
        lr_pair_type_constraints: Dict[str, List[Tuple[int, int]]],
        log: bool = True,
    ) -> np.ndarray:
        """
        Shared core: given a (n_cells, n_cells, n_types) tensor and LR pair constraints,
        zero out entries whose (sender_type, receiver_type) is not allowed for that channel.
        """
        if arr.ndim != 3 or edge_type_map is None or not lr_pair_type_constraints:
            return arr

        n_cells, _, n_types = arr.shape
        ct_src = cell_types.reshape(n_cells, 1)
        ct_dst = cell_types.reshape(1, n_cells)
        n_pairs_applied = 0
        for pair_name, type_idx in edge_type_map.items():
            if type_idx >= n_types:
                continue
            type_pairs = lr_pair_type_constraints.get(pair_name)
            if not type_pairs:
                continue
            pair_mask = np.zeros((n_cells, n_cells), dtype=bool)
            for ct1, ct2 in type_pairs:
                pair_mask |= (ct_src == ct1) & (ct_dst == ct2)
            if not pair_mask.any():
                arr[:, :, type_idx] = 0.0
            else:
                arr[:, :, type_idx] *= pair_mask
            n_pairs_applied += 1
        if log:
            print(
                f"[EdgeFilterUtils] _apply_pair_type_constraints_core: "
                f"applied sender/receiver type constraints for {n_pairs_applied} LR channels"
            )
        return arr

    @staticmethod
    def filter_cells_by_annotation(
        modality_datas: List[AnnData],
        pos_edge_probs: np.ndarray,
        annotation_key: str = "annotation",
        include_list: Optional[List[str]] = None,
    ) -> Tuple[List[AnnData], np.ndarray]:
        """
        Keep only cells whose adata.obs[annotation_key] is in include_list; filter modalities and probs.
        Returns filtered list of AnnData and filtered (n', n', n_types) edge probability array.
        """
        if not include_list:
            return modality_datas, pos_edge_probs
        m0 = modality_datas[0]
        if annotation_key not in m0.obs:
            raise ValueError(f"Annotation key '{annotation_key}' not in adata.obs: {list(m0.obs.keys())}")
        mask = m0.obs[annotation_key].astype(str).isin(include_list).values
        if mask.sum() == 0:
            raise ValueError(
                f"No cells in include_list {include_list}. Available: {sorted(m0.obs[annotation_key].astype(str).unique())}"
            )
        filtered_datas = [d[mask, :].copy() for d in modality_datas]
        filtered_probs = pos_edge_probs[np.ix_(mask, mask, np.arange(pos_edge_probs.shape[2]))]
        return filtered_datas, filtered_probs

    @staticmethod
    def filter_connections_by_annotation(
        pos_edge_probs: np.ndarray,
        adata: AnnData,
        annotation_key: str,
        include_list: List[str]
    ) -> np.ndarray:
        """
        Filter connections by annotation. This removes connections but keeps all cells (sets connections to 0).
        
        Args:
            pos_edge_probs: Edge probability matrix as numpy array (n_cells, n_cells, n_edge_types)
            adata: AnnData object containing annotation information
            annotation_key: Name of the annotation column (e.g., 'cell_type')
            include_list: List of annotation values to keep connections for
        
        Returns:
            Filtered edge probabilities as numpy array
        """
        if annotation_key not in adata.obs.columns:
            raise ValueError(f"Annotation key '{annotation_key}' not found in adata.obs.columns")
        
        # Convert include_list to strings for consistent comparison
        include_list_str = [str(d) for d in include_list]
        # Create mask for cells to keep (True for cells in include_list)
        # Handle both string and numeric annotations by converting to string
        annotations = adata.obs[annotation_key].astype(str)
        keep_mask = annotations.isin(include_list_str).values
        n_filtered = (~keep_mask).sum()
        n_kept = keep_mask.sum()
        print(f"Filtering connections by annotation '{annotation_key}': keeping {n_kept} cells from domains {include_list}, removing connections for {n_filtered} cells")
        
        # Set connections to 0 for cells not in include_list
        # Remove connections where source cell is not in include_list
        # Remove connections where target cell is not in include_list
        pos_edge_probs = pos_edge_probs * keep_mask[:, None, None]  # Filter source cells
        pos_edge_probs = pos_edge_probs * keep_mask[None, :, None]  # Filter target cells
        
        return pos_edge_probs
