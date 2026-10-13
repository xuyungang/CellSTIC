"""Balanced-clustering graph construction on sparse spatial candidates."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse


class ClusterGraphUtils:
    """Build the balanced-cluster branch of the CellSTIC feature graph."""

    @staticmethod
    def balanced_kmean(
        features: torch.Tensor,
        n_clusters: int,
        device: torch.device = torch.device("cpu"),
        tol: float = 1e-4,
        max_iter: int = 100,
        similarity_batch_size: int = 16_384,
        random_seed: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run capacity-constrained cosine k-means.

        Similarities are evaluated in cell batches, so this implements the
        original balanced-clustering rule without materializing the complete
        ``n_clusters x n_cells`` similarity matrix.
        """
        x = torch.as_tensor(features, dtype=torch.float32, device=device)
        if x.ndim != 2 or x.shape[0] < 2:
            raise ValueError(
                "features must be a 2D tensor containing at least two cells, "
                f"got {tuple(x.shape)}"
            )
        if not 2 <= n_clusters <= x.shape[0]:
            raise ValueError(
                f"n_clusters must be between 2 and {x.shape[0]}, got {n_clusters}"
            )
        if similarity_batch_size < 1:
            raise ValueError("similarity_batch_size must be >= 1")

        centroids = ClusterGraphUtils._kmeans_plusplus(
            x, n_clusters=n_clusters, random_seed=random_seed
        )
        n_cells = x.shape[0]
        base_capacity, remainder = divmod(n_cells, n_clusters)
        capacities = torch.full(
            (n_clusters,), base_capacity, dtype=torch.long, device=device
        )
        capacities[:remainder] += 1
        normalized_x = F.normalize(x, p=2, dim=1)
        best_shift = math.inf
        no_improvement_count = 0

        for _ in range(max_iter):
            best_clusters, best_similarities = ClusterGraphUtils._best_clusters(
                normalized_x,
                F.normalize(centroids, p=2, dim=1),
                similarity_batch_size,
            )
            assignment = ClusterGraphUtils._balanced_assignment(
                best_clusters, best_similarities, capacities
            )
            updated = ClusterGraphUtils._update_centroids(x, assignment, n_clusters)
            center_shift = (
                torch.linalg.vector_norm(updated - centroids, dim=1).sum().item()
            )
            centroids = updated
            if center_shift < best_shift:
                best_shift = center_shift
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            if center_shift < tol or no_improvement_count >= 5:
                break

        return assignment.cpu(), centroids.cpu()

    @staticmethod
    def _best_clusters(
        normalized_x: torch.Tensor,
        normalized_centroids: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return each cell's best centroid using bounded temporary memory."""
        n_cells = normalized_x.shape[0]
        best_clusters = torch.empty(
            n_cells, dtype=torch.long, device=normalized_x.device
        )
        best_similarities = torch.empty(
            n_cells, dtype=normalized_x.dtype, device=normalized_x.device
        )
        for start in range(0, n_cells, batch_size):
            end = min(start + batch_size, n_cells)
            similarity = normalized_x[start:end] @ normalized_centroids.T
            values, indices = similarity.max(dim=1)
            best_clusters[start:end] = indices
            best_similarities[start:end] = values
        return best_clusters, best_similarities

    @staticmethod
    def _balanced_assignment(
        best_clusters: torch.Tensor,
        best_similarities: torch.Tensor,
        capacities: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the original confidence-ordered balanced assignment rule."""
        target_device = best_clusters.device
        preferred = best_clusters.detach().cpu().numpy()
        confidence = best_similarities.detach().cpu().numpy()
        capacity = capacities.detach().cpu().numpy()
        assignment = np.full(preferred.shape, -1, dtype=np.int64)
        cluster_sizes = np.zeros_like(capacity)

        # Selecting the most confident cells separately for each preferred
        # cluster is equivalent to the original global confidence ordering,
        # because capacities are independent at this stage.
        for cluster, limit in enumerate(capacity):
            indices = np.flatnonzero(preferred == cluster)
            if len(indices) == 0:
                continue
            order = np.argsort(-confidence[indices], kind="stable")
            selected = indices[order[:limit]]
            assignment[selected] = cluster
            cluster_sizes[cluster] = len(selected)

        for sample in np.flatnonzero(assignment < 0):
            available = np.flatnonzero(cluster_sizes < capacity)
            cluster = available[np.argmin(cluster_sizes[available])]
            assignment[sample] = cluster
            cluster_sizes[cluster] += 1
        return torch.as_tensor(assignment, dtype=torch.long, device=target_device)

    @staticmethod
    def _update_centroids(
        features: torch.Tensor,
        assignment: torch.Tensor,
        n_clusters: int,
    ) -> torch.Tensor:
        centroids = torch.zeros(
            (n_clusters, features.shape[1]),
            dtype=features.dtype,
            device=features.device,
        )
        counts = torch.zeros(n_clusters, dtype=features.dtype, device=features.device)
        centroids.index_add_(0, assignment, features)
        counts.index_add_(
            0, assignment, torch.ones_like(assignment, dtype=features.dtype)
        )
        return centroids / counts.clamp_min(1).unsqueeze(1)

    @staticmethod
    def _kmeans_plusplus(
        features: torch.Tensor,
        n_clusters: int,
        random_seed: int,
    ) -> torch.Tensor:
        """Initialize cosine k-means centers with a private RNG."""
        generator = torch.Generator(device=str(features.device))
        generator.manual_seed(int(random_seed))
        normalized_x = F.normalize(features, p=2, dim=1)
        centers = torch.empty(
            (n_clusters, features.shape[1]),
            dtype=features.dtype,
            device=features.device,
        )
        first = int(
            torch.randint(
                features.shape[0], (1,), generator=generator, device=features.device
            )
        )
        centers[0] = features[first]
        closest_distance = (1.0 - normalized_x @ normalized_x[first]).clamp_min(0)
        for center_index in range(1, n_clusters):
            total = closest_distance.sum()
            if not torch.isfinite(total) or total <= 0:
                selected = int(
                    torch.randint(
                        features.shape[0],
                        (1,),
                        generator=generator,
                        device=features.device,
                    )
                )
            else:
                selected = int(
                    torch.multinomial(
                        closest_distance,
                        1,
                        generator=generator,
                    )
                )
            centers[center_index] = features[selected]
            distance = (1.0 - normalized_x @ normalized_x[selected]).clamp_min(0)
            closest_distance = torch.minimum(closest_distance, distance)
        return centers

    @staticmethod
    def cluster_and_build_graph_balanced(
        features: np.ndarray,
        min_cluster_size: int,
        top_k: int,
        candidate_pairs: np.ndarray,
        device: torch.device = torch.device("cpu"),
        random_seed: int = 0,
    ) -> sparse.csr_matrix:
        """Return the sparse balanced-cluster Top-k graph with self-loops."""
        values = np.asarray(features)
        if values.ndim != 2 or values.shape[0] < 2:
            raise ValueError(
                "features must be a 2D matrix containing at least two cells, "
                f"got {values.shape}"
            )
        if min_cluster_size < 1 or top_k < 1:
            raise ValueError("min_cluster_size and top_k must both be >= 1")
        n_cells = values.shape[0]
        n_clusters = max(2, min(n_cells // min_cluster_size, n_cells // 2))
        labels, _ = ClusterGraphUtils.balanced_kmean(
            torch.as_tensor(values, dtype=torch.float32),
            n_clusters=n_clusters,
            device=device,
            random_seed=random_seed,
        )
        labels_np = labels.numpy()
        normalized = values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-8)
        rows: list[int] = []
        cols: list[int] = []
        weights: list[float] = []
        for cluster in range(n_clusters):
            indices = np.flatnonzero(labels_np == cluster)
            if len(indices) <= 1:
                continue
            similarities = (normalized[indices] @ normalized[indices].T + 1.0) / 2.0
            np.fill_diagonal(similarities, -np.inf)
            neighbor_count = min(top_k, len(indices) - 1)
            selected = np.argpartition(similarities, -neighbor_count, axis=1)[
                :, -neighbor_count:
            ]
            source = np.repeat(indices, neighbor_count)
            target = indices[selected.reshape(-1)]
            edge_weights = similarities[
                np.arange(len(indices))[:, None], selected
            ].reshape(-1)
            rows.extend(source.tolist())
            cols.extend(target.tolist())
            weights.extend(edge_weights.tolist())

        rows.extend(range(n_cells))
        cols.extend(range(n_cells))
        weights.extend(np.ones(n_cells, dtype=np.float32).tolist())
        graph = sparse.csr_matrix((weights, (rows, cols)), shape=(n_cells, n_cells))

        # This is algebraically the former post-fusion spatial mask, expressed
        # sparsely so no n_cells x n_cells distance matrix is required.
        candidates = np.asarray(candidate_pairs, dtype=np.int64).reshape(-1, 2)
        mask_rows = np.r_[candidates[:, 0], np.arange(n_cells)]
        mask_cols = np.r_[candidates[:, 1], np.arange(n_cells)]
        spatial_mask = sparse.csr_matrix(
            (np.ones(len(mask_rows)), (mask_rows, mask_cols)),
            shape=(n_cells, n_cells),
        )
        graph = graph.multiply(spatial_mask).tocsr()
        return graph


__all__ = ["ClusterGraphUtils"]
