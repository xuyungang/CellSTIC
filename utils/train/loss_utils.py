"""Loss utilities: GAEs BCE, edge type CE, reconstruction MSE, unsupervised clustering."""

import torch
import torch.nn.functional as F

from .clustering_utils import ClusteringUtils


class LossUtils:
    """GAEs loss, edge type reconstruction, reconstruction MSE, unsupervised clustering loss."""

    @staticmethod
    def gaes_loss(
        pos_mask_indicator: torch.Tensor,
        pos_edge_probs: torch.Tensor,
        neg_edge_probs: torch.Tensor,
        neg_mask_indicator: torch.Tensor,
        pos_positions: torch.Tensor,
        neg_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Binary cross-entropy over positive/negative edge probs (masked by indicators)."""
        if pos_edge_probs.size(0) == 0 and neg_edge_probs.size(0) == 0:
            return (pos_edge_probs.sum() + neg_edge_probs.sum()) * 0.0
        eps = 1e-8

        def _extract(indicator, edge_probs, positions):
            if edge_probs.size(0) == 0:
                return None
            # Ensure shape match: indicator (n,n,T1), edge_probs (N,T2) -> need T1==T2 for mask
            pos_indicators = indicator[positions[:, 0], positions[:, 1], :]
            n_pos, n_types_ind = pos_indicators.shape
            n_ep, n_types_ep = edge_probs.shape
            if n_types_ind != n_types_ep or n_pos != n_ep:
                return None
            mask = pos_indicators > 0
            valid = edge_probs[mask]
            return valid if valid.size(0) > 0 else None

        def _grad_zero(pos_ep, neg_ep):
            """Return grad-connected zero for backward compatibility."""
            return (pos_ep.sum() + neg_ep.sum()) * 0.0

        _zero = _grad_zero(pos_edge_probs, neg_edge_probs)
        pos_loss = _zero
        if pos_edge_probs.size(0) > 0:
            probs = _extract(pos_mask_indicator, pos_edge_probs, pos_positions)
            if probs is not None:
                probs = torch.clamp(probs, min=eps, max=1.0 - eps)
                pos_loss = -torch.mean(torch.log(probs + eps))
            elif pos_edge_probs.requires_grad:
                # _extract returned None (indicator/probs shape mismatch); use grad-connected zero
                pos_loss = pos_edge_probs.mean() * 0.0
            if not torch.isfinite(pos_loss):
                pos_loss = _zero
        neg_loss = _zero
        if neg_edge_probs.size(0) > 0:
            probs = _extract(neg_mask_indicator, neg_edge_probs, neg_positions)
            if probs is not None:
                probs = torch.clamp(probs, min=eps, max=1.0 - eps)
                neg_loss = -torch.mean((1.0 + probs) * torch.log(1.0 - probs + eps))
            elif neg_edge_probs.requires_grad:
                neg_loss = neg_edge_probs.mean() * 0.0
            if not torch.isfinite(neg_loss):
                neg_loss = _zero
        return pos_loss + neg_loss

    @staticmethod
    def edge_type_reconstruction_loss(
        predicted_edge_types: torch.Tensor, target_edge_types: torch.Tensor
    ) -> torch.Tensor:
        """Cross-entropy for edge type prediction (logits vs one-hot or indices)."""
        if predicted_edge_types is None or target_edge_types is None:
            dev = target_edge_types.device if target_edge_types is not None else predicted_edge_types.device
            return torch.tensor(0.0, device=dev)
        if target_edge_types.dim() > 1 and target_edge_types.size(1) > 1:
            target_indices = target_edge_types.argmax(dim=1)
        else:
            target_indices = target_edge_types.squeeze().long()
        return F.cross_entropy(predicted_edge_types, target_indices)

    @staticmethod
    def reconstruction_loss(
        predicted_features: torch.Tensor, target_features: torch.Tensor
    ) -> torch.Tensor:
        """MSE between predicted and target features (aligned on leading dims)."""
        n = min(predicted_features.size(0), target_features.size(0))
        d = min(predicted_features.size(1), target_features.size(1))
        pred = predicted_features[:n, :d]
        tgt = target_features[:n, :d]
        return torch.mean((pred - tgt) ** 2)

    @staticmethod
    def unsupervised_clustering_loss(
        features: torch.Tensor,
        n_clusters: int = None,
        entropy_weight: float = None,
    ) -> torch.Tensor:
        """Soft K-means loss + entropy regularization (uses ClusteringUtils.cluster_pytorch)."""
        eps = 1e-8
        temperature = 0.1
        device = features.device
        n_samples = features.size(0)
        n_clusters = n_clusters or ClusteringUtils.auto_n_clusters(n_samples)
        entropy_weight = entropy_weight if entropy_weight is not None else 0.2
        features_norm = F.normalize(features, p=2, dim=1, eps=eps)
        _, centers = ClusteringUtils.cluster_pytorch(
            features_norm, n_clusters=n_clusters, temperature=temperature, n_iterations=8, normalize=False
        )
        centers_norm = F.normalize(centers, p=2, dim=1, eps=eps)
        dists_sq = torch.cdist(features_norm, centers_norm, p=2) ** 2
        soft_assign = F.softmax(-dists_sq / (temperature + eps), dim=1)
        kmeans_loss = (soft_assign * dists_sq).sum() / n_samples
        entropy_loss = -torch.mean((soft_assign * torch.log(soft_assign + eps)).sum(dim=1))
        loss = kmeans_loss + entropy_weight * entropy_loss
        return loss if torch.isfinite(loss) else torch.tensor(0.0, device=device)
