import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
import pandas as pd
import torch
from anndata import AnnData


class CellTypeCommunicationComputer:
    """Compute cell-type × cell-type × LR communication matrix (used by alluvial, hierarchy bubble)."""

    @staticmethod
    def _cell_types_per_obs(
        cell_map: Union[pd.Series, Mapping[Any, Any]],
        obs_names: List[Any],
        n_cells: int,
    ) -> np.ndarray:
        """
        One string cell-type label per graph row, aligned with pos_edge_probs axis 0.

        When ``cell_map`` is a pandas Series (typical: ``adata.obs[cell_type_key]``), use
        **positional** alignment with the graph: row ``i`` uses ``cell_map.iloc[i]``.
        Using ``Series.get(str(obs_name))`` breaks when obs index labels are integers
        (``str(0)`` != ``0``), yielding wrong or all-"Unknown" labels and empty aggregation.
        """
        if isinstance(cell_map, pd.Series):
            if len(cell_map) != n_cells:
                raise ValueError(
                    f"cell_map length ({len(cell_map)}) != graph n_cells ({n_cells})."
                )
            return cell_map.astype(str).to_numpy()

        out = np.empty(n_cells, dtype=object)
        for i, n in enumerate(obs_names):
            ct = None
            if hasattr(cell_map, "get"):
                ct = cell_map.get(n)
                if ct is None:
                    ct = cell_map.get(str(n))
            out[i] = str(ct) if ct is not None else "Unknown"
        return np.asarray(out, dtype=str)

    def compute_cell_type_communication_matrix(
        self,
        pos_edge_probs: Union[np.ndarray, torch.Tensor],
        cell_map: Union[pd.Series, Mapping[Any, Any]],
        obs_names: List[Any],
        edge_type_map: Optional[Dict[str, int]] = None,
        save_path: Optional[str] = None,
        cell_types_filter: Optional[List[str]] = None,
        threshold: float = 0.7,
        min_count_threshold: int = 500,
    ) -> Dict[str, np.ndarray]:
        """Aggregate edge probs to cell-type x cell-type x LR; optional CSV save."""
        # Convert to numpy array if tensor
        pos_edge_probs = pos_edge_probs.detach().cpu().numpy() if isinstance(pos_edge_probs, torch.Tensor) else np.asarray(pos_edge_probs)
        pos_edge_probs = np.nan_to_num(pos_edge_probs, nan=0.0, posinf=0.0, neginf=0.0)
        n_cells, _, n_edge_types = pos_edge_probs.shape

        ct_per_cell = self._cell_types_per_obs(cell_map, obs_names, n_cells)
        
        # Build ligand-receptor mapping first (needed for early return)
        if edge_type_map:
            # edge_type_map: {pair_name: idx}
            idx_to_lr = {idx: name for name, idx in edge_type_map.items() if idx < n_edge_types}
        else:
            idx_to_lr = {idx: f"LR_{idx}" for idx in range(n_edge_types)}
        
        # Unique cell types (same order as used for idx_map)
        all_cell_types = sorted(np.unique(ct_per_cell).tolist())
        if cell_types_filter is not None:
            cell_types_filter_set = set(cell_types_filter)
            unique_cell_types = sorted([ct for ct in all_cell_types if ct in cell_types_filter_set])
            print(f"Filtering cell types: {len(unique_cell_types)}/{len(all_cell_types)} cell types included")
            if len(unique_cell_types) == 0:
                print(f"  Available cell types in data: {all_cell_types}")
                print(f"  Filter list: {sorted(cell_types_filter)}")
                print(f"  Missing from filter (in data but not in filter): {sorted(set(all_cell_types) - cell_types_filter_set)}")
                print(f"  Missing from data (in filter but not in data): {sorted(cell_types_filter_set - set(all_cell_types))}")
        else:
            unique_cell_types = all_cell_types
        
        n_cell_types = len(unique_cell_types)
        
        # Early return if no cell types after filtering
        if n_cell_types == 0:
            print(f"Warning: No cell types available after filtering. Returning empty result.")
            return {
                'cell_type_matrix': np.zeros((0, 0, n_edge_types)),
                'unique_cell_types': [],
                'idx_to_lr': idx_to_lr,
                'cell_type_to_idx': {},
                'counts': np.zeros((0, 0, n_edge_types)),
            }
        
        cell_type_to_idx = {ct: idx for idx, ct in enumerate(unique_cell_types)}
        
        # Create index map and valid cells mask (row i <-> ct_per_cell[i])
        idx_map = np.array([cell_type_to_idx.get(str(ct_per_cell[i]), -1) for i in range(n_cells)], dtype=np.int32)
        valid_cells = idx_map >= 0 if cell_types_filter is not None else np.ones(n_cells, dtype=bool)
        
        # Initialize matrices
        cell_type_matrix = np.zeros((n_cell_types, n_cell_types, n_edge_types))
        counts = np.zeros((n_cell_types, n_cell_types, n_edge_types))
        mask_base = (np.arange(n_cells)[:, None] != np.arange(n_cells)) & valid_cells[:, None] & valid_cells[None, :]
        
        # Step 1: Filter edges by threshold.
        # threshold > 0: keep strictly above threshold (strict avoids boundary quirks).
        # threshold <= 0: no cutoff — keep any finite off-diagonal edge with prob > 0
        # (still excludes exact zeros; use a tiny floor so near-zero float mass is not lost).
        for edge_type_idx in range(n_edge_types):
            edge_probs = pos_edge_probs[:, :, edge_type_idx]
            thr = float(threshold)
            if thr <= 0.0:
                eps = np.finfo(edge_probs.dtype).eps * 4
                filtered_mask = np.isfinite(edge_probs) & mask_base & (edge_probs > eps)
            else:
                filtered_mask = np.isfinite(edge_probs) & mask_base & (edge_probs > thr)
            
            # Step 2: Calculate total strength (sum of probabilities) between cell type pairs
            cell_type_strength = np.zeros((n_cell_types, n_cell_types))
            cell_type_counts = np.zeros((n_cell_types, n_cell_types))
            for i, j in zip(*np.where(filtered_mask)):
                src_idx, tgt_idx = idx_map[i], idx_map[j]
                if src_idx >= 0 and tgt_idx >= 0:
                    # Accumulate probability strength (total strength)
                    cell_type_strength[src_idx, tgt_idx] += edge_probs[i, j]
                    # Also count edges for reference
                    cell_type_counts[src_idx, tgt_idx] += 1
            
            # Filter low counts (set counts < min_count_threshold to 0)
            low_count_mask = cell_type_counts < min_count_threshold
            cell_type_strength[low_count_mask] = 0
            cell_type_counts[low_count_mask] = 0
            
            counts[:, :, edge_type_idx] = cell_type_counts
            
            # Normalize strength by maximum value for each edge type
            # This ensures each LR pair's strength is normalized to [0, 1] range
            if cell_type_strength.size == 0:
                cell_type_matrix[:, :, edge_type_idx] = cell_type_strength
                continue
            max_strength = np.max(cell_type_strength)
            if max_strength > 0:
                cell_type_strength = cell_type_strength / max_strength
            cell_type_matrix[:, :, edge_type_idx] = cell_type_strength

        
        
        result = {
            'cell_type_matrix': cell_type_matrix,
            'unique_cell_types': unique_cell_types,
            'idx_to_lr': idx_to_lr,
            'cell_type_to_idx': cell_type_to_idx,
            'counts': counts,
        }
        
        if save_path:
            # Save to CSV format
            if save_path.endswith('.csv'):
                save_file = save_path
            else:
                # If save_path is a directory, create CSV file in it
                save_file = os.path.join(save_path, 'cell_type_communication_matrix.csv')
            
            # Create directory if it doesn't exist
            save_dir = os.path.dirname(save_file) or '.'
            os.makedirs(save_dir, exist_ok=True)
            
            # Create a DataFrame with all cell type pairs and LR pairs
            rows = []
            for edge_type_idx in range(n_edge_types):
                lr_name = idx_to_lr.get(edge_type_idx, f"LR_{edge_type_idx}")
                for src_idx, src_ct in enumerate(unique_cell_types):
                    for tgt_idx, tgt_ct in enumerate(unique_cell_types):
                        strength = cell_type_matrix[src_idx, tgt_idx, edge_type_idx]
                        count = counts[src_idx, tgt_idx, edge_type_idx]
                        if strength > 0 or count > 0:  # Only save non-zero entries
                            rows.append({
                                'source_cell_type': src_ct,
                                'target_cell_type': tgt_ct,
                                'lr_pair': lr_name,
                                'communication_strength': strength,
                                'edge_count': int(count)
                            })
            
            # Create DataFrame and save to CSV
            df = pd.DataFrame(rows)
            if len(df) > 0:
                # Sort by LR pair, then by strength (descending)
                df = df.sort_values(['lr_pair', 'communication_strength'], ascending=[True, False])
            df.to_csv(save_file, index=False)
            print(f"Cell type communication matrix saved to {save_file}")
            print(f"Total records: {len(df)}")
            if len(df) > 0:
                print(f"Unique cell type pairs: {df[['source_cell_type', 'target_cell_type']].drop_duplicates().shape[0]}")
                print(f"Unique LR pairs: {df['lr_pair'].nunique()}")
        
        return result
