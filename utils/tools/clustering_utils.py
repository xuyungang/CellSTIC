"""Clustering utilities: PCA, soft K-means, label alignment, Louvain."""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from typing import Optional, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix, issparse
from sklearn.decomposition import PCA

from utils.tools.seed_utils import active_base_seed


def pca(adata: ad.AnnData, use_reps: Optional[str] = None, n_comps: int = 10) -> np.ndarray:
    """PCA on adata.X or adata.obsm[use_reps]; returns (n_samples, n_comps_safe)."""
    data = adata.obsm[use_reps] if use_reps else adata.X
    if issparse(data):
        data = data.toarray()
    data = np.asarray(data, dtype=np.float32)
    n_samples, n_features = data.shape
    n_comps_safe = max(1, min(n_comps, n_features, max(1, n_samples - 1)))
    return PCA(n_components=n_comps_safe, random_state=active_base_seed()).fit_transform(data)


class ClusteringUtils:
    """Auto n_clusters, soft K-means (PyTorch), label alignment (Hungarian), Louvain."""

    @staticmethod
    def auto_n_clusters(n_samples: int, min_clusters: int = 2, max_clusters: int = 15) -> int:
        """Return max(min_clusters, min(sqrt(n_samples), max_clusters))."""
        return max(min_clusters, min(int(np.sqrt(n_samples)), max_clusters))

    @staticmethod
    def cluster_pytorch(
        features: torch.Tensor,
        n_clusters: Optional[int] = None,
        temperature: float = 0.1,
        n_iterations: int = 5,
        normalize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Soft K-means in PyTorch. Returns (cluster_labels, cluster_centers)."""
        eps = 1e-8
        device = features.device
        n_samples = features.size(0)
        n_clusters = n_clusters or ClusteringUtils.auto_n_clusters(n_samples)
        torch.manual_seed(active_base_seed())
        if normalize:
            mu = features.mean(dim=0, keepdim=True)
            sigma = features.std(dim=0, keepdim=True) + eps
            features_norm = (features - mu) / sigma
        else:
            features_norm = features
        center_idx = torch.randint(0, n_samples, (1,), device=device).item()
        centers = features_norm[center_idx : center_idx + 1]
        for _ in range(n_clusters - 1):
            min_dists = torch.min(torch.cdist(features_norm, centers, p=2), dim=1)[0] ** 2
            min_dists = min_dists + torch.rand_like(min_dists) * 0.1 * torch.mean(min_dists)
            probs = min_dists / (torch.sum(min_dists) + eps)
            next_idx = torch.multinomial(probs, 1).item()
            centers = torch.cat([centers, features_norm[next_idx : next_idx + 1]], dim=0)
        for _ in range(n_iterations):
            dists_sq = torch.cdist(features_norm, centers, p=2) ** 2
            soft_assign = F.softmax(-dists_sq / (temperature + eps), dim=1)
            assignment_sums = torch.clamp(soft_assign.sum(dim=0, keepdim=True), min=eps)
            centers_new = (soft_assign.t() @ features_norm) / assignment_sums.t()
            if torch.norm(centers_new - centers, p=2, dim=1).mean() < 1e-4:
                break
            centers = centers_new
        dists = torch.cdist(features_norm, centers, p=2)
        cluster_labels = torch.argmin(dists, dim=1)
        return cluster_labels, centers

    @staticmethod
    def align_labels(cluster_labels: np.ndarray, true_labels: np.ndarray) -> np.ndarray:
        """Align cluster labels to true labels (Hungarian or majority per cluster)."""
        cluster_labels = np.asarray(cluster_labels)
        true_labels = np.asarray(true_labels)
        if cluster_labels.shape[0] != true_labels.shape[0]:
            raise ValueError("cluster_labels and true_labels must have same length")
        unique_clusters = np.unique(cluster_labels)
        unique_true = np.unique(true_labels)
        n_true, n_clusters = len(unique_true), len(unique_clusters)
        true_to_idx = {l: i for i, l in enumerate(unique_true)}
        cluster_to_idx = {l: i for i, l in enumerate(unique_clusters)}
        confusion = np.zeros((n_true, n_clusters), dtype=int)
        for tl, cl in zip(true_labels, cluster_labels):
            confusion[true_to_idx[tl], cluster_to_idx[cl]] += 1
        row_ind, col_ind = linear_sum_assignment(-confusion)
        cluster_to_aligned = {unique_clusters[col_ind[i]]: unique_true[row_ind[i]] for i in range(len(row_ind))}
        for cl in unique_clusters:
            if cl not in cluster_to_aligned:
                mask = cluster_labels == cl
                if mask.any():
                    vals, counts = np.unique(true_labels[mask], return_counts=True)
                    cluster_to_aligned[cl] = vals[np.argmax(counts)]
                else:
                    cluster_to_aligned[cl] = unique_true[0]
        return np.array([cluster_to_aligned[cl] for cl in cluster_labels])

    @staticmethod
    def cluster_louvain(
        features: Union[torch.Tensor, np.ndarray],
        obs_names,
        n_clusters: int,
        n_neighbors: int = 50,
        n_comps: int = 20,
        start_res: float = 0.1,
        end_res: float = 2,
        increment: float = 0.01,
    ) -> np.ndarray:
        """Louvain clustering with resolution search to approximate n_clusters."""
        features = np.asarray(features, dtype=np.float32)
        adata = ad.AnnData(features)
        adata.obs_names = obs_names
        n_samples, n_features = features.shape
        n_comps = max(2, min(n_comps, n_features, max(1, n_samples - 1)))
        adata.obsm["X_pca"] = pca(adata, use_reps=None, n_comps=n_comps)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X_pca", random_state=active_base_seed())
        best_res, best_diff, best_count = None, float("inf"), None
        for res in sorted(np.arange(start_res, end_res, increment), reverse=True):
            sc.tl.louvain(adata, random_state=active_base_seed(), resolution=res)
            count = len(pd.DataFrame(adata.obs["louvain"]).louvain.unique())
            if count == n_clusters:
                return adata.obs["louvain"].values.astype(int)
            if abs(count - n_clusters) < best_diff:
                best_diff = abs(count - n_clusters)
                best_res, best_count = res, count
        if best_res is not None:
            sc.tl.louvain(adata, random_state=active_base_seed(), resolution=best_res)
        return adata.obs["louvain"].values.astype(int)
