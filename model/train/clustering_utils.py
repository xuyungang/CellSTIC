"""Clustering utilities: PCA, soft K-means, label alignment, Louvain."""

from typing import Optional, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits


def pca(
    adata: ad.AnnData, use_reps: Optional[str] = None, n_comps: int = 10
) -> np.ndarray:
    """PCA on adata.X or adata.obsm[use_reps]; returns (n_samples, n_comps_safe)."""
    if n_comps < 1:
        raise ValueError(f"n_comps must be >= 1, got {n_comps}")
    data = adata.obsm[use_reps] if use_reps else adata.X
    if issparse(data):
        data = data.toarray()
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError(f"PCA input must be a non-empty 2D matrix, got {data.shape}")
    n_samples, n_features = data.shape
    n_comps_safe = max(1, min(n_comps, n_features, max(1, n_samples - 1)))
    # Match preprocessing PCA: avoid multithreaded BLAS stalls in QR/SVD.
    with threadpool_limits(limits=1, user_api="blas"):
        return PCA(n_components=n_comps_safe).fit_transform(data)


class ClusteringUtils:
    """Auto n_clusters, soft K-means (PyTorch), label alignment (Hungarian), Louvain."""

    @staticmethod
    def auto_n_clusters(
        n_samples: int, min_clusters: int = 2, max_clusters: int = 15
    ) -> int:
        """Return max(min_clusters, min(sqrt(n_samples), max_clusters))."""
        if n_samples < min_clusters:
            raise ValueError(
                f"n_samples must be >= min_clusters ({min_clusters}), got {n_samples}"
            )
        if min_clusters < 1 or max_clusters < min_clusters:
            raise ValueError("Require 1 <= min_clusters <= max_clusters")
        return max(min_clusters, min(int(np.sqrt(n_samples)), max_clusters))

    @staticmethod
    def cluster_pytorch(
        features: torch.Tensor,
        n_clusters: Optional[int] = None,
        temperature: float = 0.1,
        n_iterations: int = 5,
        random_seed: Optional[int] = None,
        normalize: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run soft K-means without reseeding PyTorch's global RNG.

        ``random_seed`` creates a private generator for a standalone call. Pass a
        persistent ``generator`` when successive calls should consume one
        reproducible random stream, as in feature pretraining.
        """
        eps = 1e-8
        if features.ndim != 2 or features.shape[0] < 2 or features.shape[1] < 1:
            raise ValueError(
                "features must be a 2D tensor with at least two samples, got "
                f"{tuple(features.shape)}"
            )
        if not torch.isfinite(features).all():
            raise ValueError("features contains non-finite values")
        device = features.device
        n_samples = features.size(0)
        n_clusters = (
            ClusteringUtils.auto_n_clusters(n_samples)
            if n_clusters is None
            else int(n_clusters)
        )
        if not 2 <= n_clusters <= n_samples:
            raise ValueError(
                f"n_clusters must be between 2 and {n_samples}, got {n_clusters}"
            )
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if n_iterations < 1:
            raise ValueError(f"n_iterations must be >= 1, got {n_iterations}")
        if random_seed is not None and generator is not None:
            raise ValueError("Pass either random_seed or generator, not both")
        if random_seed is not None:
            generator = torch.Generator(device=str(device))
            generator.manual_seed(random_seed)
        if generator is not None:
            generator_device = torch.device(generator.device)
            same_device = generator_device == device
            same_default_cuda = (
                generator_device.type == device.type == "cuda"
                and (
                    generator_device.index is None
                    or generator_device.index == device.index
                )
            )
        else:
            same_device = True
            same_default_cuda = False
        if not (same_device or same_default_cuda):
            raise ValueError(
                f"generator device {generator.device} does not match feature "
                f"device {device}"
            )
        if normalize:
            mu = features.mean(dim=0, keepdim=True)
            sigma = features.std(dim=0, keepdim=True) + eps
            features_norm = (features - mu) / sigma
        else:
            features_norm = features
        center_idx = torch.randint(
            0, n_samples, (1,), device=device, generator=generator
        ).item()
        centers = features_norm[center_idx : center_idx + 1]
        for _ in range(n_clusters - 1):
            min_dists = (
                torch.min(torch.cdist(features_norm, centers, p=2), dim=1)[0] ** 2
            )
            noise = torch.rand(
                min_dists.shape,
                dtype=min_dists.dtype,
                device=device,
                generator=generator,
            )
            min_dists = min_dists + noise * 0.1 * torch.mean(min_dists)
            total_distance = torch.sum(min_dists)
            if torch.isfinite(total_distance) and total_distance > eps:
                probs = min_dists / total_distance
                next_idx = torch.multinomial(probs, 1, generator=generator).item()
            else:
                next_idx = torch.randint(
                    0,
                    n_samples,
                    (1,),
                    device=device,
                    generator=generator,
                ).item()
            centers = torch.cat(
                [centers, features_norm[next_idx : next_idx + 1]], dim=0
            )
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
        from utils.metrics.clust_metrics import ClusteringMetrics

        return ClusteringMetrics.align_labels(cluster_labels, true_labels)

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
        random_state: int = 0,
    ) -> np.ndarray:
        """Louvain clustering with resolution search to approximate n_clusters."""
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[0] < 2:
            raise ValueError("features must be a 2D matrix with at least two samples")
        adata = ad.AnnData(features)
        adata.obs_names = obs_names
        n_samples, n_features = features.shape
        if len(obs_names) != n_samples:
            raise ValueError("obs_names must contain one identifier per sample")
        if not 2 <= n_clusters <= n_samples:
            raise ValueError(
                f"n_clusters must be between 2 and {n_samples}, got {n_clusters}"
            )
        n_neighbors = min(int(n_neighbors), n_samples - 1)
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1")
        n_comps = max(2, min(n_comps, n_features, max(1, n_samples - 1)))
        adata.obsm["X_pca"] = pca(adata, use_reps=None, n_comps=n_comps)
        sc.pp.neighbors(
            adata, n_neighbors=n_neighbors, use_rep="X_pca", random_state=random_state
        )
        best_res, best_diff = None, float("inf")
        for res in sorted(np.arange(start_res, end_res, increment), reverse=True):
            sc.tl.louvain(adata, random_state=random_state, resolution=res)
            count = len(pd.DataFrame(adata.obs["louvain"]).louvain.unique())
            if count == n_clusters:
                return adata.obs["louvain"].values.astype(int)
            if abs(count - n_clusters) < best_diff:
                best_diff = abs(count - n_clusters)
                best_res = res
        if best_res is not None:
            sc.tl.louvain(adata, random_state=random_state, resolution=best_res)
        return adata.obs["louvain"].values.astype(int)
