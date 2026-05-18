"""
Utilities for the scmultisim dataset:
- load_scmultisim_true_labels: load ground-truth labels from label.h5
- get_ligand_receptor_map_with_edge_types: load LR pairs from CSV and build edge_type_map
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union

import h5py
import numpy as np
import pandas as pd


def load_scmultisim_true_labels(label_file: Union[str, Path]) -> np.ndarray:
    """
    Load scmultisim ground-truth labels from a label.h5 file.

    Expected file structure:
        label_file containing dataset "cci_gt".

    Returns:
        label: np.ndarray, same shape semantics as the original cci_gt
        (we keep the transpose here to stay consistent with the original implementation).
    """
    with h5py.File(label_file, "r") as f:
        label = np.array(f["cci_gt"][:])
    # Keep the same behavior as the original implementation: transpose to (H, W, C) layout
    label = np.transpose(label, (1, 2, 0))
    return label


def get_ligand_receptor_map_with_edge_types(file_path: str) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    """
    Load a ligand–receptor table from CSV and construct:
    - ligand_receptor_map: ligand -> [receptor, ...]
    - edge_type_map: "ligand:receptor" -> idx (0-based)

    The CSV must contain at least three columns (case-insensitive):
        - a column whose name contains 'ligand'
        - a column whose name contains 'receptor'
        - a column whose name contains 'idx' (1-based index, converted to 0-based here)
    """
    lr_df = pd.read_csv(file_path, index_col=0)

    ligand_receptor_map: Dict[str, List[str]] = {}

    ligand_col = None
    receptor_col = None
    idx_col = None

    # Automatically detect column names
    for col in lr_df.columns:
        col_lower = col.lower()
        if "ligand" in col_lower and ligand_col is None:
            ligand_col = col
        elif "receptor" in col_lower and receptor_col is None:
            receptor_col = col
        elif "idx" in col_lower and idx_col is None:
            idx_col = col

    if ligand_col is None or receptor_col is None or idx_col is None:
        raise ValueError(
            f"CSV {file_path} does not contain ligand/receptor/idx columns; "
            f"current columns: {list(lr_df.columns)}"
        )

    # Keep as string to avoid truncating numeric identifiers
    lr_df[ligand_col] = lr_df[ligand_col].astype(str)
    lr_df[receptor_col] = lr_df[receptor_col].astype(str)
    lr_df[idx_col] = lr_df[idx_col].astype(int)

    edge_type_map: Dict[str, int] = {}
    for _, row in lr_df.iterrows():
        ligand = str(row[ligand_col]).strip()
        receptor = str(row[receptor_col]).strip()
        idx = int(row[idx_col])

        if ligand and receptor and ligand != "nan" and receptor != "nan":
            if ligand not in ligand_receptor_map:
                ligand_receptor_map[ligand] = []
            if receptor not in ligand_receptor_map[ligand]:
                ligand_receptor_map[ligand].append(receptor)

            key = f"{ligand}:{receptor}"
            # idx is 1-based in the CSV; convert to 0-based to use as channel indices
            edge_type_map[key] = idx - 1

    return ligand_receptor_map, edge_type_map
