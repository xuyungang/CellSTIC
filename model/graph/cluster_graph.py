"""Cluster-based graph construction (balanced k-means + cosine similarity graph)."""

import warnings
from functools import partial

import numpy as np
import torch

from utils.tools.seed_utils import active_base_seed


class ClusterGraphUtils:
    """Balanced k-means clustering and cluster-based adjacency."""

    @staticmethod
    def balanced_kmean(X, n_clusters, device=torch.device('cpu'), tol=1e-4, max_iter=100):
        """Balanced k-means; returns (cluster_assignment, centroids)."""
        # Convert to float
        X = X.float()

        # Transfer to device
        X = X.to(device)

        pairwise_similarity_function = partial(ClusterGraphUtils.pairwise_cosine, device=device)
        
        # Initialize
        centroids, _ = ClusterGraphUtils._kmeans_plusplus(X,
                                            n_clusters,
                                            random_state=active_base_seed(),
                                            pairwise_similarity=pairwise_similarity_function,
                                            n_local_trials=None)
        
        N = len(X)
        n_per_cluster = N // n_clusters
        n_left = N % n_clusters
        
        cluster_assignment = torch.zeros(N, dtype=torch.long, device=device) - 1
        cluster_size = torch.zeros(n_clusters, dtype=torch.long, device=device)
        similarity_matrix = torch.empty((n_clusters, N), device=device)
        last_centroids = centroids.clone()
        X_normalized = torch.nn.functional.normalize(X, p=2, dim=1)
        no_improvement_count = 0
        best_shift = float('inf')
        patience = 5
        
        for i in range(max_iter):
            centroids_normalized = torch.nn.functional.normalize(centroids, p=2, dim=1)
            similarity_matrix = torch.mm(centroids_normalized, X_normalized.t())
            cluster_assignment.fill_(-1)
            cluster_size.fill_(0)
            cluster_assignment, cluster_size = ClusterGraphUtils._optimized_assign_samples(
                similarity_matrix, cluster_assignment, cluster_size, 
                n_per_cluster, n_left, N, n_clusters
            )

            assert torch.all(cluster_assignment != -1)
            last_centroids.copy_(centroids)
            centroids = ClusterGraphUtils._optimized_update_centroids(X, cluster_assignment, n_clusters)
            center_shift = torch.norm(centroids - last_centroids, dim=1).sum()
            if center_shift < best_shift:
                best_shift = center_shift
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
            if center_shift < tol or no_improvement_count >= patience:
                break

        return cluster_assignment.cpu(), centroids.cpu()

    @staticmethod
    def _optimized_assign_samples(similarity_matrix, cluster_assignment, cluster_size, n_per_cluster, n_left, N, n_clusters):
        """Assign samples to clusters under capacity (balanced)."""
        best_clusters = torch.argmax(similarity_matrix, dim=0)
        best_similarities = torch.max(similarity_matrix, dim=0)[0]
        _, sorted_indices = torch.topk(best_similarities, N, largest=True)
        cluster_assignment, cluster_size = ClusterGraphUtils._vectorized_batch_assign(
            sorted_indices, best_clusters, cluster_assignment, cluster_size,
            n_per_cluster, n_left, N, n_clusters
        )
        
        return cluster_assignment, cluster_size

    @staticmethod
    def _vectorized_batch_assign(sorted_indices, best_clusters, cluster_assignment, cluster_size, n_per_cluster, n_left, N, n_clusters):
        """Batch assign by best similarity under per-cluster capacity."""
        cluster_capacity = torch.full((n_clusters,), n_per_cluster, device=cluster_size.device)
        if n_left > 0:
            cluster_capacity[:n_left] += 1
        for idx in sorted_indices:
            sample_idx = idx.item()
            cluster_idx = best_clusters[sample_idx].item()
            if cluster_size[cluster_idx] < cluster_capacity[cluster_idx]:
                cluster_assignment[sample_idx] = cluster_idx
                cluster_size[cluster_idx] += 1
        unassigned_mask = cluster_assignment == -1
        if unassigned_mask.any():
            unassigned_indices = torch.where(unassigned_mask)[0]
            for sample_idx in unassigned_indices:
                min_cluster = torch.argmin(cluster_size).item()
                cluster_assignment[sample_idx] = min_cluster
                cluster_size[min_cluster] += 1
        
        return cluster_assignment, cluster_size

    @staticmethod
    def _optimized_update_centroids(X, cluster_assignment, n_clusters):
        """Update centroids by scatter-add and mean."""
        centroids = torch.zeros((n_clusters, X.shape[1]), device=X.device, dtype=X.dtype)
        cluster_counts = torch.zeros(n_clusters, device=X.device, dtype=torch.long)
        centroids.scatter_add_(0, cluster_assignment.unsqueeze(1).expand(-1, X.shape[1]), X)
        cluster_counts.scatter_add_(0, cluster_assignment, torch.ones_like(cluster_assignment))
        cluster_counts = torch.clamp(cluster_counts, min=1)
        centroids = centroids / cluster_counts.unsqueeze(1).float()
        
        return centroids

    @staticmethod
    def pairwise_cosine(data1, data2, device=torch.device('cpu')):
        """Cosine similarity matrix (normalize then matmul)."""
        data1, data2 = data1.to(device), data2.to(device)
        A_normalized = torch.nn.functional.normalize(data1, p=2, dim=1)
        B_normalized = torch.nn.functional.normalize(data2, p=2, dim=1)
        return torch.mm(A_normalized, B_normalized.t())

    @staticmethod
    def stable_cumsum(arr, dim=None, rtol=1e-05, atol=1e-08):
        """Cumsum in float64; warn if last element != sum."""
        if dim is None:
            arr = arr.flatten()
            dim = 0
        out = torch.cumsum(arr, dim=dim, dtype=torch.float64)
        expected = torch.sum(arr, dim=dim, dtype=torch.float64)
        if not torch.all(torch.isclose(out.take(torch.Tensor([-1]).long().to(arr.device)),
                                       expected, rtol=rtol,
                                       atol=atol, equal_nan=True)):
            warnings.warn('cumsum was found to be unstable: '
                          'its last element does not correspond to sum',
                          RuntimeWarning)
        return out

    @staticmethod
    def _kmeans_plusplus(X, n_clusters, random_state, pairwise_similarity, n_local_trials=None):
        """K-means++ initialization; returns (centers, indices)."""
        n_samples, n_features = X.shape

        generator = torch.Generator(device=str(X.device))
        generator.manual_seed(random_state)
        centers = torch.empty((n_clusters, n_features), dtype=X.dtype, device=X.device)
        if n_local_trials is None:
            n_local_trials = 2 + int(np.log(n_clusters))
        center_id = torch.randint(n_samples, (1,), generator=generator, device=X.device)

        indices = torch.full((n_clusters,), -1, dtype=torch.int, device=X.device)
        centers[0] = X[center_id]
        indices[0] = center_id
        closest_dist_sq = 1 / pairwise_similarity(centers[0, None], X)
        current_pot = closest_dist_sq.sum()
        for c in range(1, n_clusters):
            rand_vals = torch.rand(n_local_trials, generator=generator, device=X.device) * current_pot

            candidate_ids = torch.searchsorted(ClusterGraphUtils.stable_cumsum(closest_dist_sq), rand_vals)
            torch.clip(candidate_ids, None, closest_dist_sq.numel() - 1, out=candidate_ids)
            distance_to_candidates = 1 / pairwise_similarity(X[candidate_ids], X)
            torch.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
            candidates_pot = distance_to_candidates.sum(dim=1)
            best_candidate = torch.argmin(candidates_pot)
            current_pot = candidates_pot[best_candidate]
            closest_dist_sq = distance_to_candidates[best_candidate]
            best_candidate = candidate_ids[best_candidate]
            centers[c] = X[best_candidate]
            indices[c] = best_candidate

        return centers, indices

    @staticmethod
    def initialize(X, num_clusters):
        """Initialize cluster centers (random perm + mean)."""
        num_samples = X.shape[1]
        bs = X.shape[0]

        indices = torch.empty(X.shape[:-1], device=X.device, dtype=torch.long)
        for i in range(bs):
            indices[i] = torch.randperm(num_samples, device=X.device)
        initial_state = torch.gather(X, 1, indices.unsqueeze(-1).repeat(1, 1, X.shape[-1])).reshape(bs, num_clusters, -1, X.shape[-1]).mean(dim=-2)
        return initial_state

    @staticmethod
    def cluster_and_build_graph_balanced(
        features: np.ndarray,
        min_cluster_size: int,
        top_k: int,
        device: torch.device = torch.device('cpu'),
        isPrint: bool = False,
    ) -> np.ndarray:
        """Balanced k-means + per-cluster cosine-sim graph; returns adjacency (self-loops 1)."""
        n_samples = len(features)
        optimal_k = max(2, min(int(n_samples / min_cluster_size), n_samples // 2))
        features_tensor = torch.tensor(features, dtype=torch.float32)
        cluster_labels, _ = ClusterGraphUtils.balanced_kmean(
            X=features_tensor, n_clusters=optimal_k, device=device, tol=1e-4, max_iter=100
        )
        cluster_labels = cluster_labels.numpy()
        n_cells = len(cluster_labels)
        adjacency_matrix = np.zeros((n_cells, n_cells))
        features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
        cosine_sim = (np.dot(features_norm, features_norm.T) + 1) / 2
        for cluster_id in np.unique(cluster_labels):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) <= 1:
                continue
            for i in cluster_indices:
                similarities = cosine_sim[i, cluster_indices].copy()
                similarities[np.where(cluster_indices == i)[0][0]] = 0
                top_k_indices = np.argsort(similarities)[-top_k:]
                top_k_nodes = cluster_indices[top_k_indices]
                
                for j in top_k_nodes:
                    if i != j:
                        adjacency_matrix[i, j] = cosine_sim[i, j]
        
        if isPrint:
            ClusterGraphUtils._print_cluster_statistics(cluster_labels, min_cluster_size)
        np.fill_diagonal(adjacency_matrix, 1)        
        return adjacency_matrix


    @staticmethod
    def _print_cluster_statistics(cluster_labels: np.ndarray, min_cluster_size: int) -> None:
        """Print cluster size stats (min/max/avg and OK/BAD per cluster)."""
        unique = np.unique(cluster_labels)
        sizes = [np.sum(cluster_labels == l) for l in unique]
        print(f"Clusters: {len(unique)}, samples: {len(cluster_labels)}, threshold: [{min_cluster_size}, {min_cluster_size * 3}]")
        for label, size in sorted(zip(unique, sizes), key=lambda x: x[1], reverse=True):
            status = "OK" if min_cluster_size <= size <= min_cluster_size * 3 else "BAD"
            print(f"  Cluster {label}: {size} [{status}]")
        print(f"Summary: min={min(sizes)}, max={max(sizes)}, avg={np.mean(sizes):.1f}")
