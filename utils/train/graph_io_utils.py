"""Save/load feature DGL graphs and graph data (enhanced graph, edge_type_map, hierarchy_dict)."""

from pathlib import Path
from typing import Dict, List

import dgl
import numpy as np
import pickle


class GraphIOUtils:
    """Save and load feature graphs (DGL) and graph data (numpy + pickle)."""

    @staticmethod
    def save_feature_graphs(modality_g_dgls: List[dgl.DGLGraph], model_path: str) -> None:
        """Save list of DGL graphs to model_path/modality_feature_graphs.bin."""
        save_dir = Path(model_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        dgl.save_graphs(str(save_dir / "modality_feature_graphs.bin"), modality_g_dgls)

    @staticmethod
    def load_feature_graphs(model_path: str, device) -> List[dgl.DGLGraph]:
        """Load DGL graphs from model_path and move to device."""
        path = Path(model_path) / "modality_feature_graphs.bin"
        if not path.exists():
            raise FileNotFoundError(f"Feature graphs not found: {path}")
        graphs, _ = dgl.load_graphs(str(path))
        return [g.to(device) for g in graphs]

    @staticmethod
    def save_graph_data(
        enhanced_base_graph: np.ndarray,
        edge_type_map: Dict,
        hierarchy_dict: Dict,
        model_path: str,
        ligand_strength_adj: np.ndarray = None,
        receptor_strength_adj: np.ndarray = None,
        knn_per_modality: np.ndarray = None,
    ) -> None:
        """Save enhanced graph (.npy), edge_type_map, hierarchy_dict (.pkl), optional strength / KNN arrays (.npy).
        knn_per_modality: (n_cells, n_cells, n_modalities), each modality's KNN graph for edge features."""
        save_dir = Path(model_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / "enhanced_base_graph.npy", enhanced_base_graph)
        with open(save_dir / "edge_type_map.pkl", "wb") as f:
            pickle.dump(edge_type_map, f)
        with open(save_dir / "hierarchy_dict.pkl", "wb") as f:
            pickle.dump(hierarchy_dict, f)
        if ligand_strength_adj is not None:
            np.save(save_dir / "ligand_strength_adj.npy", ligand_strength_adj)
        if receptor_strength_adj is not None:
            np.save(save_dir / "receptor_strength_adj.npy", receptor_strength_adj)
        if knn_per_modality is not None:
            np.save(save_dir / "knn_per_modality.npy", knn_per_modality)
