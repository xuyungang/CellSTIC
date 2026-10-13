"""Loss utilities for sparse CCC and multimodal feature training."""

from typing import Optional

import torch
import torch.nn.functional as F

from .clustering_utils import ClusteringUtils


class LossUtils:
    """Losses used by the active sparse training pipeline."""

    @staticmethod
    def sampled_gaes_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Stable weighted GAE loss for aligned sparse edge/channel samples.

        ``sample_weights`` are expected to sum to one over the complete epoch
        sample set.  This allows losses to be accumulated over mini-batches
        without changing the original positive and hard-negative objectives.
        Soft ancestor targets interpolate between those two endpoint losses.
        """
        if logits.ndim != 1:
            raise ValueError(f"logits must be 1D, got {tuple(logits.shape)}")
        if labels.shape != logits.shape or sample_weights.shape != logits.shape:
            raise ValueError(
                "logits, labels, and sample_weights must have identical shapes, "
                f"got {tuple(logits.shape)}, {tuple(labels.shape)}, and "
                f"{tuple(sample_weights.shape)}"
            )
        if logits.numel() == 0:
            return logits.sum() * 0.0
        if (
            not torch.isfinite(labels).all()
            or torch.any(labels < 0)
            or torch.any(labels > 1)
        ):
            raise ValueError("labels must be finite values in [0, 1]")
        if torch.any(sample_weights < 0) or not torch.isfinite(sample_weights).all():
            raise ValueError("sample_weights must be finite and non-negative")

        probabilities = torch.sigmoid(logits)
        positive_loss = -F.logsigmoid(logits)
        negative_loss = -(1.0 + probabilities) * F.logsigmoid(-logits)
        per_sample = labels * positive_loss + (1.0 - labels) * negative_loss
        loss = torch.sum(per_sample * sample_weights)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite sampled GAE loss")
        return loss

    @staticmethod
    def reconstruction_loss(
        predicted_features: torch.Tensor, target_features: torch.Tensor
    ) -> torch.Tensor:
        """Return reconstruction MSE for tensors with exactly matching shapes.

        A mismatch means that the decoder architecture or observation alignment is
        invalid.  Failing here prevents unsupervised feature axes or observations
        from being silently discarded.
        """
        if predicted_features.ndim != 2 or target_features.ndim != 2:
            raise ValueError(
                "Reconstruction inputs must be 2D feature matrices, got "
                f"predicted={tuple(predicted_features.shape)}, "
                f"target={tuple(target_features.shape)}"
            )
        if predicted_features.shape != target_features.shape:
            raise ValueError(
                "Reconstruction shape mismatch: "
                f"predicted={tuple(predicted_features.shape)}, "
                f"target={tuple(target_features.shape)}"
            )
        return F.mse_loss(predicted_features, target_features)

    @staticmethod
    def unsupervised_clustering_loss(
        features: torch.Tensor,
        n_clusters: Optional[int] = None,
        entropy_weight: Optional[float] = None,
        temperature: float = 0.1,
        n_iterations: int = 8,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Soft K-means loss + entropy regularization (uses ClusteringUtils.cluster_pytorch)."""
        eps = 1e-8
        if features.ndim != 2 or features.size(0) == 0:
            raise ValueError(
                f"features must be a non-empty 2D matrix, got {tuple(features.shape)}"
            )
        n_samples = features.size(0)
        n_clusters = (
            ClusteringUtils.auto_n_clusters(n_samples)
            if n_clusters is None
            else n_clusters
        )
        entropy_weight = entropy_weight if entropy_weight is not None else 0.2
        if not 2 <= n_clusters <= n_samples:
            raise ValueError(
                f"n_clusters must be between 2 and n_samples ({n_samples}), "
                f"got {n_clusters}"
            )
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if n_iterations < 1:
            raise ValueError(f"n_iterations must be >= 1, got {n_iterations}")
        if entropy_weight < 0:
            raise ValueError(f"entropy_weight must be >= 0, got {entropy_weight}")
        features_norm = F.normalize(features, p=2, dim=1, eps=eps)
        _, centers = ClusteringUtils.cluster_pytorch(
            features_norm,
            n_clusters=n_clusters,
            temperature=temperature,
            n_iterations=n_iterations,
            generator=generator,
            normalize=False,
        )
        centers_norm = F.normalize(centers, p=2, dim=1, eps=eps)
        dists_sq = torch.cdist(features_norm, centers_norm, p=2) ** 2
        soft_assign = F.softmax(-dists_sq / (temperature + eps), dim=1)
        kmeans_loss = (soft_assign * dists_sq).sum() / n_samples
        entropy_loss = -torch.mean(
            (soft_assign * torch.log(soft_assign + eps)).sum(dim=1)
        )
        loss = kmeans_loss + entropy_weight * entropy_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite unsupervised clustering loss")
        return loss
