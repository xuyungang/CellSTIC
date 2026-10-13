"""Negative sampling from a bounded sparse candidate set."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .sparse_graph import as_sparse_typed_graph


class NegativeSampler:
    @staticmethod
    def sample_channel_balanced(
        graph,
        candidate_pairs: np.ndarray,
        channel_counts: np.ndarray,
        *,
        random_seed: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample edges that are negative in each requested target channel."""
        typed_graph = as_sparse_typed_graph(graph)
        candidates = np.asarray(candidate_pairs, dtype=np.int64).reshape(-1, 2)
        counts = np.asarray(channel_counts, dtype=np.int64).reshape(-1)
        if len(counts) != len(typed_graph):
            raise ValueError(
                "channel_counts must contain one value per graph channel, got "
                f"{len(counts)} counts for {len(typed_graph)} channels"
            )
        if np.any(counts < 0):
            raise ValueError("channel_counts must be non-negative")
        if counts.sum() > 0 and len(candidates) == 0:
            raise ValueError("No negative spatial candidates are available")
        if len(candidates) and (
            candidates.min() < 0 or candidates.max() >= typed_graph.n_nodes
        ):
            raise ValueError("candidate_pairs contains an out-of-range node index")
        rng = np.random.default_rng(random_seed)
        position_parts = []
        channel_parts = []
        for channel, count in enumerate(counts):
            if count == 0:
                continue
            occupied = np.asarray(
                typed_graph[channel][candidates[:, 0], candidates[:, 1]]
            ).reshape(-1) > 0
            available = np.flatnonzero(~occupied)
            if len(available) == 0:
                raise ValueError(
                    f"No negative spatial candidates are available for channel {channel}"
                )
            indices = rng.choice(
                available,
                int(count),
                replace=len(available) < count,
            )
            position_parts.append(candidates[indices])
            channel_parts.append(np.full(int(count), channel, dtype=np.int64))
        if not position_parts:
            return (
                np.empty((0, 2), dtype=np.int64),
                np.empty(0, dtype=np.int64),
            )
        positions = np.vstack(position_parts)
        channels = np.concatenate(channel_parts)
        permutation = rng.permutation(len(positions))
        return positions[permutation], channels[permutation]
