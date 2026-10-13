"""Spatial strength maps for saved communication hierarchy channels."""
from __future__ import annotations

from pathlib import Path
from textwrap import fill

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd

from scipy import sparse


TEXT_COLOR = "#263A46"
FRAME_COLOR = "#8492A1"
SPATIAL_CMAP = LinearSegmentedColormap.from_list(
    "hierarchy_spatial_blue", ["#EFF3F7", "#B5C9DE", "#5D88B2", "#244E78"]
)
RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
    "font.size": 8,
    "axes.linewidth": .8,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.color": TEXT_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "axes.titlecolor": TEXT_COLOR,
    "axes.edgecolor": FRAME_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
}


def _clean_matrix(matrix, n_obs, threshold):
    """Filter a copy of directed weights without inserting sparse diagonals."""
    if hasattr(matrix, "detach"):
        matrix = matrix.detach().cpu().numpy()
    matrix = sparse.csr_matrix(matrix, dtype=np.float64).copy()
    if matrix.shape != (n_obs, n_obs):
        raise ValueError(f"Each channel must have shape ({n_obs}, {n_obs})")
    matrix.sum_duplicates()
    if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
        raise ValueError("Communication weights must be finite and nonnegative")
    diagonal = matrix.diagonal()
    if np.any(diagonal):
        matrix = matrix - sparse.diags(diagonal, format="csr")
    matrix.data[matrix.data < threshold] = 0
    matrix.eliminate_zeros()
    return matrix


class HierarchySpatialVisualizer:
    """Draw hierarchy-channel strengths as spatial heatmaps in a single row."""

    def plot(self, *, adata, channel_matrices, level=2, threshold=0.0,
             save_path=".", channel_titles=None):
        """Export SVG/PDF/PNG maps and per-observation source data.

        ``channel_matrices`` maps channel names to sender-by-receiver matrices;
        sparse, dense and tensor inputs are accepted. Exclude self edges and
        weights below ``threshold``, sum outgoing plus incoming scores, and
        plot log1p strength using one shared color scale. No spatial smoothing
        or per-channel scaling is applied. Inputs are left unchanged.

        ``save_path`` is an output directory. ``channel_titles`` optionally
        provides compact display labels while source data retain full names.
        Return the SVG path without displaying figures or requiring IPython.
        """
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold must be finite and nonnegative")
        channels = sorted(channel_matrices.items())
        if not channels:
            raise ValueError("Supply at least one hierarchy channel")
        target = Path(save_path)
        target.mkdir(parents=True, exist_ok=True)
        coordinates = np.asarray(adata.obsm["spatial"], dtype=float)[:, :2]
        if coordinates.shape != (adata.n_obs, 2) or not len(coordinates) or not np.isfinite(coordinates).all():
            raise ValueError("Expected nonempty, finite spatial coordinates of shape (n_obs, 2)")
        frames = []
        for order, (name, matrix) in enumerate(channels, 1):
            matrix = _clean_matrix(matrix, adata.n_obs, threshold)
            outgoing = np.asarray(matrix.sum(axis=1)).ravel()
            incoming = np.asarray(matrix.sum(axis=0)).ravel()
            strength = outgoing + incoming
            frames.append(pd.DataFrame({
                "obs_name": adata.obs_names.astype(str), "level": level,
                "channel": name, "channel_order": order,
                "x": coordinates[:, 0], "y": coordinates[:, 1],
                "outgoing_strength": outgoing, "incoming_strength": incoming,
                "strength": strength, "log1p_strength": np.log1p(strength),
            }))
        source = pd.concat(frames, ignore_index=True)
        low, high = coordinates.min(axis=0), coordinates.max(axis=0)
        center = (low + high) / 2
        half_span = max(float(np.max(high - low)), 1.0) * .535
        vmin, vmax = float(source.log1p_strength.min()), float(source.log1p_strength.max())
        norm = Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1)
        with mpl.rc_context(RC):
            fig = plt.figure(figsize=(1.6 * len(channels) + .7, 1.85))
            grid = fig.add_gridspec(1, len(channels) + 1,
                                   width_ratios=[1] * len(channels) + [.04],
                                   left=.035, right=.92, bottom=.08, top=.77, wspace=.16)
            for column, frame in enumerate(frames):
                axis = fig.add_subplot(grid[0, column])
                points = axis.scatter(frame.x, frame.y, c=frame.log1p_strength,
                                      cmap=SPATIAL_CMAP, norm=norm, s=1.6,
                                      alpha=1, edgecolors="none", linewidths=0, rasterized=True)
                axis.set_xlim(center[0] - half_span, center[0] + half_span)
                axis.set_ylim(center[1] - half_span, center[1] + half_span)
                axis.set_aspect("equal", adjustable="box")
                axis.set_box_aspect(1)
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_visible(True)
                    spine.set_color(FRAME_COLOR)
                    spine.set_linewidth(.6)
                name = channels[column][0]
                axis.set_title((channel_titles or {}).get(name, fill(name, 28)),
                               fontsize=6.3, fontweight="medium", linespacing=1.15, pad=6)
            bar = fig.colorbar(points, cax=fig.add_subplot(grid[0, -1]))
            bar.set_label("log1p signal", fontsize=5.6, labelpad=3)
            bar.ax.tick_params(width=.5, length=1.8, pad=1.5, labelsize=4.9)
            bar.outline.set_linewidth(.6)
            bar.outline.set_edgecolor(FRAME_COLOR)
            try:
                for suffix in ("png", "pdf", "svg"):
                    fig.savefig(target / f"level_{level}_spatial_strength.{suffix}",
                                dpi=600, bbox_inches="tight", pad_inches=.03,
                                facecolor="white")
            finally:
                plt.close(fig)
        source.to_csv(target / f"level_{level}_spatial_strength_source.csv", index=False)
        return target / f"level_{level}_spatial_strength.svg"
