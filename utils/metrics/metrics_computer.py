"""
Metrics: CCC CSV export and Region (spatial + UMAP + metrics + CSV).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import anndata as ad
from typing import Dict, List, Optional, Union

from utils.tools.clustering_utils import ClusteringUtils
from utils.metrics.clust_metrics import ClusteringMetrics
from utils.metrics.f1_metrics import F1MetricsComputer
from utils.metrics.umap_viz import UMAPVisualizer
from utils.metrics.spatial_viz import SpatialVisualizer
from utils.metrics.palette_utils import get_custom_palette


class MetricsComputer:
    """
    Single entry for metrics computation: CCC CSV metrics and Region (spatial + UMAP + metrics + CSV).
    """

    @staticmethod
    def save_roc_pr_metrics_csv(
        pos_edge_probs: Union[torch.Tensor, np.ndarray],
        label: Optional[Union[torch.Tensor, np.ndarray]] = None,
        save_dir: Union[str, Path] = ".",
        title: Optional[str] = None,
        lr_pair_names: Optional[List[str]] = None,
        eval_mask: Optional[np.ndarray] = None,
        f1_threshold: float = 0.5,
        f1_search_optimal_threshold: bool = True,
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Compute CCC metrics vs ground truth and save CSVs under ``metrics/``.

        Saves: ``f1_summary.csv``, ``f1_per_class.csv`` (includes per-LR auroc/auprc).
        Returns DataFrames for summary and per-class metrics.
        """
        save_dir = Path(save_dir)
        metrics_dir = save_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        probs_arr = np.asarray(pos_edge_probs)
        if label is None:
            return None
        labels_arr = np.asarray(label)
        if labels_arr.size == 0:
            return None

        if probs_arr.shape != labels_arr.shape or probs_arr.ndim != 3:
            raise ValueError(
                f"Expected probs and labels with same 3D shape (n, n, m), "
                f"got probs {probs_arr.shape}, labels {labels_arr.shape}"
            )

        f1_result = F1MetricsComputer.compute_and_save(
            pos_edge_probs,
            labels_arr,
            save_dir=save_dir,
            threshold=f1_threshold,
            lr_pair_names=lr_pair_names,
            title=title,
            eval_mask=eval_mask,
            search_optimal_threshold=f1_search_optimal_threshold,
        )
        summary_row = {
            "macro_f1": f1_result["summary"]["macro_f1"],
            "macro_accuracy": f1_result["summary"]["macro_accuracy"],
            "threshold": f1_threshold,
            "macro_auroc": f1_result["summary"].get("macro_auroc"),
            "macro_auprc": f1_result["summary"].get("macro_auprc"),
        }
        if title is not None:
            summary_row["name"] = title
        if "best_threshold" in f1_result["summary"]:
            summary_row["best_threshold"] = f1_result["summary"]["best_threshold"]
            summary_row["macro_f1_best"] = f1_result["summary"]["macro_f1_best"]
            summary_row["macro_accuracy_best"] = f1_result["summary"]["macro_accuracy_best"]

        return {
            "summary": pd.DataFrame([summary_row]),
            "per_class": pd.DataFrame(f1_result["per_class_rows"]),
        }

    plot_roc_pr_and_save_csv = save_roc_pr_metrics_csv

    @staticmethod
    def run_region_umap_metrics_export(
        adata: ad.AnnData,
        save_dir: Union[str, Path] = ".",
        feature_key: str = "feat",
        true_labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
        cluster_key: Optional[str] = None,
        n_clusters: Optional[int] = None,
        n_neighbors: int = 40,
    ) -> None:
        """
        Spatial + UMAP + clustering metrics + CSV export (Region).
        Saves: spatial_domain.svg, umap.svg, cell_partitions.csv, clustering_metrics.csv.
        """
        save_dir = Path(save_dir)

        domain_dir = save_dir / "domain"
        domain_dir.mkdir(parents=True, exist_ok=True)
        if feature_key not in adata.obsm:
            raise ValueError(f"Features not found in adata.obsm['{feature_key}'].")
        if "spatial" not in adata.obsm:
            raise ValueError("Spatial coordinates not found in adata.obsm['spatial'].")

        feats = adata.obsm[feature_key]
        features_np = feats.detach().cpu().numpy() if isinstance(feats, torch.Tensor) else np.asarray(feats, dtype=np.float32)

        if cluster_key is not None:
            if cluster_key not in adata.obs:
                raise ValueError(f"Cluster key '{cluster_key}' not found in adata.obs.")
            cluster_labels = np.asarray(adata.obs[cluster_key].values)
        else:
            n_clusters = n_clusters or ClusteringUtils.auto_n_clusters(features_np.shape[0])
            cluster_labels = ClusteringUtils.cluster_louvain(
                features=features_np,
                obs_names=adata.obs_names,
                n_clusters=n_clusters,
                start_res=0.1,
                end_res=2,
            )
        if true_labels is not None:
            cluster_labels = ClusteringUtils.align_labels(cluster_labels, np.asarray(true_labels))

        unique_clusters = sorted(np.unique(cluster_labels.astype(str)))
        adata.obs["cluster"] = pd.Categorical(cluster_labels.astype(str), categories=unique_clusters)
        palette = get_custom_palette(len(unique_clusters))

        SpatialVisualizer.generate_spatial_domain_visualization(
            adata_source=adata,
            save_path=str(domain_dir / "spatial_domain.svg"),
            palette=palette,
            size_multiplier=0.85,
        )
        UMAPVisualizer.generate_visualization(
            node_features=torch.tensor(features_np, dtype=torch.float32),
            adata_source=adata,
            save_path=str(domain_dir / "umap.svg"),
            palette=palette,
            add_kde_contour=False,
            add_cluster_labels=True,
            n_neighbors=n_neighbors,
        )

        df_data = {
            "cell_id": adata.obs_names.values,
            "cluster": adata.obs["cluster"].values.astype(str),
        }
        if "spatial" in adata.obsm and adata.obsm["spatial"].shape[1] >= 2:
            c = adata.obsm["spatial"]
            df_data["x"], df_data["y"] = c[:, 0], c[:, 1]
            if c.shape[1] >= 3:
                df_data["z"] = c[:, 2]
        partitions_path = domain_dir / "cell_partitions.csv"
        pd.DataFrame(df_data).sort_values(["cluster", "cell_id"]).to_csv(
            partitions_path, index=False, encoding="utf-8"
        )
        print(f"Cell partitions saved to {partitions_path}")

        # Compute and save clustering metrics
        metrics_dict = ClusteringMetrics.calculate_metrics(
            features_np, true_labels=true_labels, cluster_labels=cluster_labels
        )
        metrics_path = domain_dir / "clustering_metrics.csv"
        pd.DataFrame([metrics_dict]).to_csv(metrics_path, index=False)
        print(f"Clustering metrics saved to {metrics_path}")
