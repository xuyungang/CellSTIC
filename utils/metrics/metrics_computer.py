"""
Metrics visualization: one entity aggregating CCC (ROC-PR + CSV) and Region (spatial + UMAP + metrics + CSV).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import anndata as ad
from typing import List, Optional, Union

from sklearn.metrics import roc_auc_score, average_precision_score

from utils.train.clustering_utils import ClusteringUtils
from utils.metrics.clust_metrics import ClusteringMetrics
from utils.metrics.f1_metrics import F1MetricsComputer
from utils.metrics.roc_pr_viz import MetricsCurveVisualizer
from utils.metrics.umap_viz import UMAPVisualizer
from utils.metrics.spatial_viz import SpatialVisualizer
from utils.metrics.palette_utils import get_custom_palette


class MetricsComputer:
    """
    Single entry for metrics computation: CCC (ROC-PR + matrix CSV) and Region (spatial + UMAP + metrics + CSV).
    """

    @staticmethod
    def plot_roc_pr_and_save_csv(
        pos_edge_probs: Union[torch.Tensor, np.ndarray],
        label: Optional[Union[torch.Tensor, np.ndarray]] = None,
        save_dir: Union[str, Path] = ".",
        title: Optional[str] = None,
        csv_name: str = "roc_pr_matrix.csv",
        lr_pair_names: Optional[List[str]] = None,
        cell_names: Optional[List[str]] = None,
        eval_mask: Optional[np.ndarray] = None,
        f1_threshold: float = 0.5,
        f1_search_optimal_threshold: bool = True,
        export_matrix_csv: bool = True,
    ) -> None:
        """
        Draw ROC-PR curve, save matrix CSV, and compute F1/ACC/AUC/AUPRC (CCC).
        Saves: figure, .npz curve data, optionally 2D prob matrix CSV per channel, f1_summary.csv, f1_per_class.csv.
        eval_mask: optional (n, n, m) bool; if provided, only mask==True entries are used for ROC/PR (e.g. exclude self-edges).
        export_matrix_csv: if True, export per-channel probability matrices to CSV in ccc/; set False when caller already saves them (e.g. evaluator uses pos_probs_dense/).

        Behavior:
        - When export_matrix_csv is True, export per-channel probability matrices to CSV for diagnostics.
        - Only when labels are provided, compute ROC/PR curves and supervised metrics (F1 / AUC / AUPRC).
        """
        save_dir = Path(save_dir)
        # CCC outputs (ROC/PR curves + per-channel CSV/NPZ) should live under `<output_root>/ccc/`.
        # The caller may still store dense probability matrices under `pos_probs_dense/` for diagnostics,
        # but ROC/PR assets themselves are requested to be grouped into `ccc/`.
        ccc_dir = save_dir / "ccc"
        ccc_dir.mkdir(parents=True, exist_ok=True)
        fig_path = ccc_dir / "roc_pr.png"
        npz_path = fig_path.with_suffix("").__str__() + "_data.npz"

        probs_arr = np.asarray(pos_edge_probs)

        # 1) Optionally export per-channel probability matrices (skip when caller already saves e.g. pos_probs_dense/)
        if export_matrix_csv and probs_arr.ndim == 3:
            n, _, m = probs_arr.shape
            base_path = save_dir / csv_name
            stem = base_path.stem
            suffix = base_path.suffix or ".csv"
            index = cell_names if (cell_names is not None and len(cell_names) == n) else None

            def _safe_name(s: str) -> str:
                return str(s).replace(":", "_").replace("/", "_").replace(" ", "_").strip() or "channel"

            for k in range(m):
                # Save 2D matrix (n x n) per channel, with row/col = cell names if provided
                mat_k = np.asarray(probs_arr[:, :, k], dtype=float)
                mat_k[~np.isfinite(mat_k)] = np.nan

                channel_label = _safe_name(lr_pair_names[k]) if lr_pair_names and k < len(lr_pair_names) else str(k)
                out_path = ccc_dir / f"{stem}_{channel_label}{suffix}"
                df = pd.DataFrame(mat_k, index=index, columns=index)
                df.to_csv(out_path, index=True)
                print(f"ROC-PR matrix (channel {k}) saved to {out_path}")

        # 2) If no label, skip ROC/PR curve and supervised metrics computation
        if label is None:
            return
        label_arr = np.asarray(label)
        if label_arr.size == 0:
            return
        labels_arr = np.asarray(label_arr)
        if eval_mask is not None and eval_mask.shape == probs_arr.shape:
            probs_1d = probs_arr[eval_mask]
            label_1d = labels_arr[eval_mask]
        else:
            probs_1d = probs_arr.reshape(-1)
            label_1d = labels_arr.reshape(-1)
        MetricsCurveVisualizer.plot_roc_pr_combined(
            probs_1d,
            label_1d,
            save_path=str(fig_path),
            title=title,
            save_data_path=npz_path,
        )

        # Per-channel AUC/AP for diagnosis (when multi-channel)
        if probs_arr.ndim == 3 and eval_mask is not None and eval_mask.shape == probs_arr.shape:
            n, _, m = probs_arr.shape
            per_channel_rows = []
            for k in range(m):
                pm = eval_mask[:, :, k]
                pk = probs_arr[:, :, k][pm]
                lk = labels_arr[:, :, k][pm]
                valid = np.isfinite(pk) & np.isfinite(lk)
                if valid.sum() > 0 and lk[valid].sum() > 0 and lk[valid].sum() < valid.sum():
                    auc_k = roc_auc_score(lk[valid], pk[valid])
                    ap_k = average_precision_score(lk[valid], pk[valid])
                else:
                    auc_k, ap_k = np.nan, np.nan
                ch_name = lr_pair_names[k] if lr_pair_names and k < len(lr_pair_names) else str(k)
                per_channel_rows.append({"channel": ch_name, "roc_auc": auc_k, "average_precision": ap_k})
            if per_channel_rows:
                pd.DataFrame(per_channel_rows).to_csv(ccc_dir / "roc_pr_per_channel.csv", index=False)
                print("Per-channel ROC/AP saved to ccc/roc_pr_per_channel.csv")

        probs_arr = np.asarray(pos_edge_probs)

        # Strict: require same 3D shape (n, n, m)
        if probs_arr.shape != labels_arr.shape or probs_arr.ndim != 3:
            raise ValueError(
                f"Expected probs and labels with same 3D shape (n, n, m), "
                f"got probs {probs_arr.shape}, labels {labels_arr.shape}"
            )

        # F1/ACC (best threshold) + per-class AUC/AUPRC
        F1MetricsComputer.compute_and_save(
            pos_edge_probs,
            labels_arr,
            save_dir=save_dir,
            threshold=f1_threshold,
            lr_pair_names=lr_pair_names,
            title=title,
            eval_mask=eval_mask,
            search_optimal_threshold=f1_search_optimal_threshold,
        )

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
