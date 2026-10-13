"""Spatial candidate construction utilities for CellSTIC."""

from typing import Union

import numpy as np
import torch
from scipy.spatial import cKDTree


class EdgeFilterUtils:
    """Construct distance-limited spatial candidate pairs."""

    @staticmethod
    def compute_spot_distance_threshold(
        spatial_coords: Union[torch.Tensor, np.ndarray],
        n_spots: int = 4,
    ) -> float:
        """Return ``n_spots`` times the median nearest-neighbor distance."""
        coords = (
            spatial_coords.detach().cpu().numpy()
            if isinstance(spatial_coords, torch.Tensor)
            else np.asarray(spatial_coords)
        )
        if coords.ndim != 2 or coords.shape[1] not in (2, 3):
            raise ValueError(
                f"spatial_coords must have shape (n, 2) or (n, 3), got {coords.shape}"
            )
        if not np.all(np.isfinite(coords)):
            raise ValueError("spatial_coords contains non-finite values")
        if n_spots < 1:
            raise ValueError(f"n_spots must be >= 1, got {n_spots}")
        if len(coords) < 2:
            return np.inf
        nearest_dist = cKDTree(np.asarray(coords, dtype=np.float64)).query(coords, k=2)[
            0
        ][:, 1]
        if np.any(~np.isfinite(nearest_dist)) or len(nearest_dist) == 0:
            return np.inf
        spot_distance = float(np.median(nearest_dist))
        return n_spots * spot_distance

    @staticmethod
    def spatial_candidate_pairs(
        spatial_coords: Union[torch.Tensor, np.ndarray],
        max_distance: float,
        min_distance: float = 0.0,
        max_neighbors: int = 64,
    ) -> np.ndarray:
        """Return directed non-self pairs within a radius without an ``n x n`` matrix."""
        coords = (
            spatial_coords.detach().cpu().numpy()
            if isinstance(spatial_coords, torch.Tensor)
            else np.asarray(spatial_coords)
        )
        if coords.ndim != 2 or coords.shape[1] not in (2, 3):
            raise ValueError(
                f"spatial_coords must have shape (n, 2) or (n, 3), got {coords.shape}"
            )
        if not np.all(np.isfinite(coords)):
            raise ValueError("spatial_coords contains non-finite values")
        if max_neighbors < 1:
            raise ValueError(f"max_neighbors must be >= 1, got {max_neighbors}")
        if np.isnan(max_distance) or np.isnan(min_distance):
            raise ValueError("min_distance and max_distance must not be NaN")
        if min_distance < 0 or min_distance > max_distance:
            raise ValueError(
                "Require 0 <= min_distance <= max_distance, got "
                f"{min_distance} and {max_distance}"
            )
        if len(coords) < 2 or max_distance <= 0:
            return np.empty((0, 2), dtype=np.int64)
        neighbor_count = min(int(max_neighbors) + 1, len(coords))
        distances, indices = cKDTree(coords).query(
            coords,
            k=neighbor_count,
            distance_upper_bound=float(max_distance),
            workers=-1,
        )
        if neighbor_count == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        rows = np.repeat(np.arange(len(coords), dtype=np.int64), neighbor_count)
        cols = np.asarray(indices).reshape(-1).astype(np.int64, copy=False)
        flat_distances = np.asarray(distances).reshape(-1)
        valid = (
            (cols < len(coords))
            & (rows != cols)
            & (flat_distances >= min_distance)
            & (flat_distances <= max_distance)
        )
        if not np.any(valid):
            return np.empty((0, 2), dtype=np.int64)
        return np.column_stack((rows[valid], cols[valid]))
