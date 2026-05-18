"""Communication strength vs spatial distance line chart visualization."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from anndata import AnnData

def _calculate_signal_distance_data(
    pos_edge_probs_np: np.ndarray,
    edge_type_map: Dict[str, int],
    adata: AnnData,
    lr_filter: Optional[List[str]] = None,
    threshold: float = 0.0,
    max_distance: Optional[float] = None,
    distance_unit: str = 'pixel',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate spatial distances and corresponding interaction scores for all cell pairs.
    """
    if 'spatial' in adata.obsm:
        coords = adata.obsm['spatial'][:, :2]
    elif 'x' in adata.obs and 'y' in adata.obs:
        coords = np.ascontiguousarray(adata.obs[['x', 'y']].values, dtype=np.float32)
    else:
        raise ValueError("No spatial coordinates found in adata")

    distances = cdist(coords, coords)

    if lr_filter is not None:
        valid_edge_indices = [edge_type_map[k] for k in lr_filter if k in edge_type_map]
        if valid_edge_indices:
            aggregated_scores = pos_edge_probs_np[:, :, valid_edge_indices].sum(axis=2)
        else:
            aggregated_scores = pos_edge_probs_np.sum(axis=2)
    else:
        aggregated_scores = pos_edge_probs_np.sum(axis=2)

    if threshold > 0.0:
        aggregated_scores = aggregated_scores.copy()
        aggregated_scores[aggregated_scores < threshold] = 0.0

    np.fill_diagonal(distances, np.nan)
    np.fill_diagonal(aggregated_scores, np.nan)

    n_cells = distances.shape[0]
    mask = ~np.eye(n_cells, dtype=bool)

    distances_flat = distances[mask]
    scores_flat = aggregated_scores[mask]

    if max_distance is not None:
        valid_mask = distances_flat <= max_distance
        distances_flat = distances_flat[valid_mask]
        scores_flat = scores_flat[valid_mask]

    valid_mask = ~(np.isnan(distances_flat) | np.isnan(scores_flat))
    distances_flat = distances_flat[valid_mask]
    scores_flat = scores_flat[valid_mask]

    return distances_flat, scores_flat


_NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Helvetica", "Arial", "sans-serif"],
    "font.size": 12,
    "axes.linewidth": 0.75,
    "axes.edgecolor": "#000000",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "none",
    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,
    "lines.linewidth": 2.0,
}


class StrengthDistanceVisualizer:
    """Tool class for plotting communication strength vs spatial distance."""

    def __init__(self):
        pass

    def plot_strength_vs_distance(
        self,
        pos_edge_probs_np: np.ndarray,
        edge_type_map: Dict[str, int],
        adata: AnnData,
        save_path: Path,
        lr_filter: Optional[List[str]] = None,
        threshold: float = 0.0,
        ccc_ground: Optional[np.ndarray] = None,
        max_distance: Optional[float] = None,
        n_bins: int = 50,
        figsize: Tuple[int, int] = (8, 6),
        dpi: int = 600,
        distance_unit: str = 'pixel',
        linewidth: float = 2.0,
    ) -> None:
        """
        Plot communication strength vs spatial distance as a line chart.
        X-axis: spatial distance (binned), Y-axis: mean communication strength per bin.
        If ccc_ground is provided, draws two lines: Predicted and Ground truth.

        save_path: Output file; suffix sets format (e.g. ``.svg`` for vector output).
        SVG uses ``svg.fonttype: none`` in rcParams so text stays editable in vector tools.

        API unchanged; styling updated toward a cleaner, Nature-like publication figure.
        """
        from matplotlib.ticker import MaxNLocator

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Predicted scores
        dist_flat, pred_scores = _calculate_signal_distance_data(
            pos_edge_probs_np=pos_edge_probs_np,
            edge_type_map=edge_type_map,
            adata=adata,
            lr_filter=lr_filter,
            threshold=threshold,
            max_distance=max_distance,
            distance_unit=distance_unit,
        )

        # Ground truth scores (same cell pairs, same aggregation logic)
        ground_scores = None
        if ccc_ground is not None:
            ccc_ground = np.asarray(ccc_ground)
            if ccc_ground.shape == pos_edge_probs_np.shape:
                ground_scores, _ = self._compute_aggregated_scores_flat(
                    score_matrix=ccc_ground,
                    edge_type_map=edge_type_map,
                    adata=adata,
                    lr_filter=lr_filter,
                    threshold=0.0,
                    max_distance=max_distance,
                )
                if ground_scores is not None and len(ground_scores) != len(dist_flat):
                    ground_scores = None

        if len(dist_flat) == 0:
            print("Warning: No valid (distance, score) pairs for strength vs distance plot; skip.")
            return

        # Bin distances
        dist_min, dist_max = float(dist_flat.min()), float(dist_flat.max())
        if dist_max <= dist_min:
            bins = np.array([dist_min, dist_min + 1e-6], dtype=float)
        else:
            bins = np.linspace(dist_min, dist_max, int(n_bins) + 1)

        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_indices = np.digitize(dist_flat, bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(bin_centers) - 1)

        # Mean and SEM per bin
        pred_means = np.full(len(bin_centers), np.nan, dtype=float)
        pred_sems = np.full(len(bin_centers), np.nan, dtype=float)
        ground_means = np.full(len(bin_centers), np.nan, dtype=float) if ground_scores is not None else None
        ground_sems = np.full(len(bin_centers), np.nan, dtype=float) if ground_scores is not None else None

        for i in range(len(bin_centers)):
            mask = bin_indices == i
            n_i = int(np.sum(mask))
            if n_i > 0:
                pred_vals = pred_scores[mask]
                pred_means[i] = float(np.mean(pred_vals))
                if n_i > 1:
                    pred_sems[i] = float(np.std(pred_vals, ddof=1) / np.sqrt(n_i))
                else:
                    pred_sems[i] = 0.0

                if ground_means is not None and ground_scores is not None:
                    ground_vals = ground_scores[mask]
                    ground_means[i] = float(np.mean(ground_vals))
                    if n_i > 1:
                        ground_sems[i] = float(np.std(ground_vals, ddof=1) / np.sqrt(n_i))
                    else:
                        ground_sems[i] = 0.0

        valid_pred = np.isfinite(pred_means)
        valid_ground = ground_means is not None and np.any(np.isfinite(ground_means))

        plt.rcParams.update(_NATURE_RC)
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
                "font.size": 8,
                "axes.labelsize": 8.5,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "legend.fontsize": 7,
                "axes.linewidth": 0.8,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "svg.fonttype": "none",
            }
        )

        fig, ax = plt.subplots(figsize=figsize, dpi=max(int(dpi), 600), facecolor="white")
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.set_axisbelow(True)

        # Nature-like colors
        color_pred = "#4E79A7"
        color_ground = "#E15759"

        # Predicted: line + subtle SEM ribbon
        if np.any(valid_pred):
            x_pred = bin_centers[valid_pred]
            y_pred = pred_means[valid_pred]
            e_pred = pred_sems[valid_pred]

            ax.plot(
                x_pred,
                y_pred,
                color=color_pred,
                linewidth=linewidth,
                label="Predicted",
                zorder=3,
                solid_capstyle="round",
            )
            if np.any(np.isfinite(e_pred)):
                ax.fill_between(
                    x_pred,
                    y_pred - np.nan_to_num(e_pred, nan=0.0),
                    y_pred + np.nan_to_num(e_pred, nan=0.0),
                    color=color_pred,
                    alpha=0.14,
                    linewidth=0,
                    zorder=2,
                )

        # Ground truth: line + subtle SEM ribbon
        if valid_ground and ground_means is not None and ground_sems is not None:
            valid_g = np.isfinite(ground_means)
            x_g = bin_centers[valid_g]
            y_g = ground_means[valid_g]
            e_g = ground_sems[valid_g]

            ax.plot(
                x_g,
                y_g,
                color=color_ground,
                linewidth=linewidth,
                linestyle="--",
                label="Ground truth",
                zorder=3,
                solid_capstyle="round",
            )
            if np.any(np.isfinite(e_g)):
                ax.fill_between(
                    x_g,
                    y_g - np.nan_to_num(e_g, nan=0.0),
                    y_g + np.nan_to_num(e_g, nan=0.0),
                    color=color_ground,
                    alpha=0.12,
                    linewidth=0,
                    zorder=1,
                )

        # Labels
        ax.set_xlabel(f"Spatial distance ({distance_unit})", fontsize=8.5, color="#222222", labelpad=4)
        ax.set_ylabel("Mean communication strength", fontsize=8.5, color="#222222", labelpad=4)

        # Ticks / limits
        ax.tick_params(axis="both", which="major", labelsize=7, colors="#222222", width=0.8, length=3, pad=2)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_ylim(bottom=0)

        # Light grid only on y-axis
        ax.grid(axis="y", linestyle="-", linewidth=0.5, alpha=0.7, color="#D9D9D9")
        ax.grid(axis="x", visible=False)

        # Clean spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.spines["left"].set_color("#222222")
        ax.spines["bottom"].set_color("#222222")

        # Legend
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 0:
            leg = ax.legend(
                loc="best",
                frameon=False,
                handlelength=1.6,
                handletextpad=0.5,
                labelspacing=0.35,
                borderaxespad=0.3,
            )
            for txt in leg.get_texts():
                txt.set_color("#222222")

        plt.tight_layout(pad=0.5)
        fig.savefig(
            save_path,
            dpi=max(int(dpi), 600),
            bbox_inches="tight",
            pad_inches=0.03,
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig)
        print(f"Strength vs distance plot saved to {save_path}")
    @staticmethod
    def _compute_aggregated_scores_flat(
        score_matrix: np.ndarray,
        edge_type_map: Dict[str, int],
        adata: AnnData,
        lr_filter: Optional[List[str]] = None,
        threshold: float = 0.0,
        max_distance: Optional[float] = None,
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Compute flattened (scores, distances) from a (n,n,m) score matrix.
        Returns (scores_flat, distances_flat). lr_filter: list of edge_type_map keys.
        """
        if 'spatial' in adata.obsm:
            coords = adata.obsm['spatial'][:, :2]
        elif 'x' in adata.obs and 'y' in adata.obs:
            coords = np.ascontiguousarray(adata.obs[['x', 'y']].values, dtype=np.float32)
        else:
            return None, np.array([])

        distances = cdist(coords, coords)
        if lr_filter is not None:
            valid_edge_indices = [edge_type_map[k] for k in lr_filter if k in edge_type_map]
            aggregated = score_matrix[:, :, valid_edge_indices].sum(axis=2) if valid_edge_indices else score_matrix.sum(axis=2)
        else:
            aggregated = score_matrix.sum(axis=2)

        if threshold > 0.0:
            aggregated = np.where(aggregated >= threshold, aggregated, 0.0).astype(np.float32)
        np.fill_diagonal(distances, np.nan)
        np.fill_diagonal(aggregated, np.nan)

        n = distances.shape[0]
        mask = ~np.eye(n, dtype=bool)
        dist_flat = distances[mask]
        scores_flat = aggregated[mask]
        if max_distance is not None:
            valid = dist_flat <= max_distance
            dist_flat = dist_flat[valid]
            scores_flat = scores_flat[valid]
        valid_mask = ~(np.isnan(dist_flat) | np.isnan(scores_flat))
        dist_flat = dist_flat[valid_mask]
        scores_flat = scores_flat[valid_mask]
        return scores_flat, dist_flat
