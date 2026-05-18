"""Domain-domain, cell type-cell type, and simple spatial communication heatmaps."""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from anndata import AnnData

from utils.viz.matplotlib_svg import configure_matplotlib_svg_for_illustrator, path_wants_svg

from .differential_analysis import _get_spatial_coords, _set_nature_style

_NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Helvetica", "Arial", "sans-serif"],
    "font.size": 14,
    "axes.linewidth": 0.75,
    "axes.edgecolor": "#000000",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "none",
}


class AggregatedHeatmapVisualizer:
    """Domain-domain / cell type-cell type heatmaps from aggregated cell-cell graph."""

    def __init__(self):
        pass

    @staticmethod
    def sanitize_filename(name: str) -> str:
        s = str(name)
        s = re.sub(r"[^\w\-_\.]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s or "item"
    
    @staticmethod
    def lr_tuple_to_key(ligand: str, receptor: str) -> str:
        return f"{ligand}:{receptor}"

    @staticmethod
    def build_group_onehot(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        One-hot (n_cells, n_groups) and sorted group names.

        Ordering rules:
        - Drop cells labeled "unknown" (excluded from aggregation and display).
        - For remaining labels:
            * Numeric strings are sorted by numeric value (0, 1, 2, ..., 9, 10, 11, ...).
            * Non-numeric strings are sorted lexicographically after all numeric labels.
        """
        labels = labels.astype(str)

        # 1) Exclude unknown; those cells do not belong to any group
        valid_mask = labels != "unknown"
        labels_valid = labels[valid_mask]

        group_names = np.unique(labels_valid)

        # 2) Split numeric vs non-numeric labels
        numeric_flags = []
        numeric_values = []
        for g in group_names:
            try:
                v = float(g)
                numeric_flags.append(True)
                numeric_values.append(v)
            except (ValueError, TypeError):
                numeric_flags.append(False)
                numeric_values.append(None)

        numeric_flags = np.array(numeric_flags, dtype=bool)

        # Sort numeric label indices by value
        numeric_idx = np.where(numeric_flags)[0]
        non_numeric_idx = np.where(~numeric_flags)[0]

        if numeric_idx.size > 0:
            numeric_sorted = numeric_idx[np.argsort([numeric_values[i] for i in numeric_idx])]
        else:
            numeric_sorted = np.array([], dtype=int)

        # Non-numeric labels: lexicographic order
        if non_numeric_idx.size > 0:
            non_numeric_labels = group_names[non_numeric_idx]
            order = np.argsort(non_numeric_labels.astype(str))
            non_numeric_sorted = non_numeric_idx[order]
        else:
            non_numeric_sorted = np.array([], dtype=int)

        # Numeric first, then non-numeric (unknown already removed)
        ordered_idx = np.concatenate([numeric_sorted, non_numeric_sorted])
        group_names = group_names[ordered_idx]

        group_to_idx = {g: i for i, g in enumerate(group_names)}

        # 3) Build one-hot: valid cells only; unknown rows stay all zeros
        idx_valid = np.array([group_to_idx[g] for g in labels_valid], dtype=int)
        n = len(labels)  # total cells including unknown
        k = len(group_names)
        onehot = np.zeros((n, k), dtype=np.float32)
        # Assign only at valid rows
        row_indices = np.nonzero(valid_mask)[0]
        onehot[row_indices, idx_valid] = 1.0
        return onehot, group_names
    
    def compute_domain_domain_matrix(
        self, graph_2d: np.ndarray, domain_labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Aggregate cell-cell graph to domain-domain: M = D^T @ G @ D (D one-hot)."""
        D, domains = self.build_group_onehot(domain_labels)
        mat = D.T @ graph_2d @ D
        return mat, domains
    
    def plot_heatmap_matrix(
        self,
        mat: np.ndarray,
        labels: np.ndarray,
        save_path: Path,
        cmap: str = "GnBu",
        log1p: bool = True,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        font_size: Optional[float] = None,
    ) -> None:
        """
        Plot (n_domains, n_domains) heatmap; optional log1p, vmin/vmax.

        Nature-like styling:
        - restrained sequential colormap
        - no visible cell borders
        - lighter axes/colorbar styling
        - publication-friendly typography
        """
        from matplotlib.ticker import MaxNLocator

        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Transform
        m = np.log1p(mat) if log1p else mat.copy()

        # Nature-like rc setup
        base_fs = 8.0 if font_size is None else float(font_size)
        label_fs = base_fs + 0.5
        tick_fs = max(1.0, base_fs - 1.0)

        plt.rcParams.update(_NATURE_RC)
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
                "font.size": base_fs,
                "axes.labelsize": label_fs,
                "xtick.labelsize": tick_fs,
                "ytick.labelsize": tick_fs,
                "axes.linewidth": 0.8,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "svg.fonttype": "none",
            }
        )

        # More publication-like figure size
        fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=600, facecolor="white")
        ax.set_facecolor("white")

        n_rows, n_cols = m.shape
        x = np.arange(n_cols + 1)
        y = np.arange(n_rows + 1)

        # Use pcolormesh without visible borders for a cleaner journal-style heatmap
        im = ax.pcolormesh(
            x,
            y,
            m,
            cmap=cmap,          # default changed to GnBu: softer, more Nature-like than viridis
            vmin=vmin,
            vmax=vmax,
            shading="flat",
            edgecolors="none",
            linewidth=0.0,
            antialiased=False,
        )

        ax.invert_yaxis()

        # Ticks centered on cells
        ax.set_xticks(np.arange(n_cols) + 0.5)
        ax.set_yticks(np.arange(n_rows) + 0.5)

        # Adaptive label rotation
        if len(labels) <= 8:
            rot = 0
            ha = "center"
        elif len(labels) <= 16:
            rot = 35
            ha = "right"
        else:
            rot = 45
            ha = "right"

        ax.set_xticklabels(labels, rotation=rot, ha=ha, color="#222222")
        ax.set_yticklabels(labels, color="#222222")

        # Clean spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.spines["left"].set_color("#222222")
        ax.spines["bottom"].set_color("#222222")

        ax.tick_params(axis="x", colors="#222222", width=0.8, length=2.5, pad=2)
        ax.tick_params(axis="y", colors="#222222", width=0.8, length=2.5, pad=2)

        # Keep square-like cells if possible
        ax.set_aspect("equal")

        # Colorbar: slimmer and lighter
        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cbar.ax.tick_params(labelsize=tick_fs, colors="#222222", width=0.7, length=2.5)
        cbar.outline.set_linewidth(0.7)
        cbar.outline.set_edgecolor("#222222")
        cbar.locator = MaxNLocator(nbins=5)
        cbar.update_ticks()

        if log1p:
            cbar.set_label(r"$\log(1+\mathrm{strength})$", fontsize=label_fs, color="#222222", labelpad=4)
        else:
            cbar.set_label("Strength", fontsize=label_fs, color="#222222", labelpad=4)

        fig.tight_layout(pad=0.4)
        if path_wants_svg(save_path):
            configure_matplotlib_svg_for_illustrator()
        fig.savefig(
            save_path,
            bbox_inches="tight",
            dpi=600,
            facecolor="white",
            edgecolor="none",
            pad_inches=0.02,
        )
        plt.close(fig)

    def plot_rectangular_heatmap(
        self,
        mat: np.ndarray,
        row_labels: List[str],
        col_labels: List[str],
        title: str,
        save_path: Path,
        cmap: str = "GnBu",
        log1p: bool = True,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        interpolation: str = "nearest",
        font_size: Optional[float] = None,
    ) -> None:
        """
        Plot (n_rows, n_cols) heatmap with row and column labels.
        Style matches Nature-like publication figures.

        Notes
        -----
        - API is unchanged.
        - `interpolation` is kept for backward compatibility, but has no effect when using pcolormesh.
        """
        from matplotlib.ticker import MaxNLocator

        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Data transform
        m = np.log1p(mat) if log1p else mat.copy()

        # Nature-like rc
        base_fs = 8.0 if font_size is None else float(font_size)
        title_fs = base_fs + 1.0
        label_fs = base_fs + 0.5
        tick_fs = max(1.0, base_fs - 1.0)

        plt.rcParams.update(_NATURE_RC)
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
                "font.size": base_fs,
                "axes.titlesize": title_fs,
                "axes.labelsize": label_fs,
                "xtick.labelsize": tick_fs,
                "ytick.labelsize": tick_fs,
                "axes.linewidth": 0.8,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "svg.fonttype": "none",
            }
        )

        # More publication-like defaults
        # Slightly adaptive size, but less inflated than the original
        fig_h = max(3.8, min(12, len(row_labels) * 0.22 + 1.8))
        fig_w = max(4.6, min(14, len(col_labels) * 0.28 + 2.2))

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=600, facecolor="white")
        ax.set_facecolor("white")

        n_rows, n_cols = m.shape
        x = np.arange(n_cols + 1)
        y = np.arange(n_rows + 1)

        # For Nature-like heatmaps, avoid visible tile borders
        # If user leaves the old default "viridis", switch internally to a softer sequential map
        cmap_to_use = "GnBu" if cmap == "viridis" else cmap

        im = ax.pcolormesh(
            x,
            y,
            m,
            cmap=cmap_to_use,
            vmin=vmin,
            vmax=vmax,
            shading="flat",
            edgecolors="none",
            linewidth=0.0,
            antialiased=False,
        )

        ax.invert_yaxis()

        # Ticks centered on cells
        ax.set_xticks(np.arange(n_cols) + 0.5)
        ax.set_yticks(np.arange(n_rows) + 0.5)

        # Adaptive x tick rotation
        # Make x tick labels readable by default:
        # even for few labels, avoid fully-horizontal text.
        if len(col_labels) <= 8:
            rot = 60
            ha = "center"
        elif len(col_labels) <= 20:
            rot = 60
            ha = "center"
        else:
            # For very dense column labels, use near-vertical text
            # and lean it toward the left for better readability.
            rot = 90
            ha = "center"

        # Adaptive font sizes for dense heatmaps
        scale = base_fs / 8.0
        if len(col_labels) <= 12:
            x_fs = 7.0 * scale
        elif len(col_labels) <= 30:
            x_fs = 6.5 * scale
        else:
            x_fs = 5.8 * scale

        if len(row_labels) <= 15:
            y_fs = 7.0 * scale
        elif len(row_labels) <= 40:
            y_fs = 6.5 * scale
        else:
            y_fs = 5.8 * scale

        x_fs = max(4.0, x_fs)
        y_fs = max(4.0, y_fs)

        ax.set_xticklabels(
            col_labels,
            rotation=rot,
            rotation_mode="default",
            ha=ha,
            color="#222222",
            fontsize=x_fs,
        )
        ax.set_yticklabels(row_labels, color="#222222", fontsize=y_fs)

        # Clean axes
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.spines["left"].set_color("#222222")
        ax.spines["bottom"].set_color("#222222")

        x_tick_pad = 6 if abs(rot) >= 60 else 2
        ax.tick_params(axis="x", colors="#222222", width=0.8, length=2.5, pad=x_tick_pad)
        ax.tick_params(axis="y", colors="#222222", width=0.8, length=2.5, pad=2)

        # Use title if provided
        if title:
            ax.set_title(title, fontsize=title_fs, color="#222222", pad=6)

        # Keep heatmap cells visually clean
        ax.set_xlim(0, n_cols)
        ax.set_ylim(n_rows, 0)

        # Slim colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.025)
        cbar.ax.tick_params(labelsize=tick_fs, colors="#222222", width=0.7, length=2.5)
        cbar.outline.set_linewidth(0.7)
        cbar.outline.set_edgecolor("#222222")
        cbar.locator = MaxNLocator(nbins=5)
        cbar.update_ticks()

        if log1p:
            cbar.set_label(r"$log(1+\mathrm{strength})$", fontsize=label_fs, color="#222222", labelpad=4)
        else:
            cbar.set_label("Strength", fontsize=label_fs, color="#222222", labelpad=4)

        fig.tight_layout(pad=0.4)
        if path_wants_svg(save_path):
            configure_matplotlib_svg_for_illustrator()
        fig.savefig(
            save_path,
            bbox_inches="tight",
            dpi=600,
            facecolor="white",
            edgecolor="none",
            pad_inches=0.02,
        )
        plt.close(fig)

    def plot_simple_communication_heatmap(
        self,
        adata: AnnData,
        pos_edge_probs_np: np.ndarray,
        edge_type_map: Dict[str, int],
        save_dir: Path,
        lr_filter: List[str],
        threshold: float = 0.0,
        score_method: str = 'sum',
        cmap: str = 'coolwarm',
        point_size: Optional[float] = None,
        font_size: Optional[float] = None,
    ) -> None:
        """
        Plot simple communication intensity heatmap: total + per LR pair (no domain).
        Saves total.svg (aggregated across lr_filter) and {lr_key}.svg per key.
        lr_filter: list of edge_type_map keys (e.g. "CCL7:CCR2", "root" for virtual nodes).
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        n_edges = pos_edge_probs_np.shape[2]

        # Total heatmap (aggregate all LR in lr_filter)
        if lr_filter:
            valid_indices = []
            for lr_key in lr_filter:
                idx = edge_type_map.get(lr_key)
                if idx is not None and idx < n_edges:
                    valid_indices.append(idx)
            if valid_indices:
                total_matrix = pos_edge_probs_np[:, :, valid_indices].sum(axis=2).astype(np.float32)
                if threshold > 0:
                    total_matrix = np.where(total_matrix >= threshold, total_matrix, 0.0)
                self._plot_single_simple_heatmap(
                    adata=adata,
                    pos_edge_probs_np=total_matrix[:, :, np.newaxis],
                    edge_type_map={"total": 0},
                    save_path=save_dir / "total.svg",
                    lr_pair=("total", "total"),
                    threshold=0.0,
                    score_method=score_method,
                    cmap=cmap,
                    point_size=point_size,
                    edge_idx=0,
                    font_size=font_size,
                )

        for lr_key in lr_filter:
            idx = edge_type_map.get(lr_key)
            if idx is None or idx >= n_edges:
                print(f"Warning: {lr_key} not in edge_type_map, skipping heatmap")
                continue
            save_path = save_dir / f"{self.sanitize_filename(lr_key.replace(':', '_'))}.svg"
            self._plot_single_simple_heatmap(
                adata=adata,
                pos_edge_probs_np=pos_edge_probs_np,
                edge_type_map=edge_type_map,
                save_path=save_path,
                lr_pair=(lr_key, lr_key),
                threshold=threshold,
                score_method=score_method,
                cmap=cmap,
                point_size=point_size,
                edge_idx=idx,
                font_size=font_size,
            )


    def _plot_single_simple_heatmap(
            self,
            adata: AnnData,
            pos_edge_probs_np: np.ndarray,
            edge_type_map: Dict[str, int],
            save_path: Path,
            lr_pair: Tuple[str, str],
            threshold: float = 0.0,
            score_method: str = 'sum',
            cmap: str = 'coolwarm',
            point_size: Optional[float] = None,
            edge_idx: Optional[int] = None,
            font_size: Optional[float] = None,
        ) -> None:
            """
            Plot one simple spatial communication intensity map for a single LR channel.
            edge_idx overrides lookup when provided.

            Nature-like styling:
            - cleaner scatter rendering
            - restrained typography and axes
            - slimmer colorbar
            - sequential colormap preferred for non-negative communication scores
            """
            from matplotlib.ticker import MaxNLocator

            n_edges = pos_edge_probs_np.shape[2]
            if edge_idx is not None:
                if edge_idx >= n_edges:
                    return
            else:
                lig, rec = lr_pair
                edge_idx = None
                for key in [f"{lig}:{rec}", f"{lig}_{rec}", f"{lig}-{rec}"]:
                    if key in edge_type_map:
                        idx = edge_type_map[key]
                        edge_idx = idx if idx < n_edges else None
                        break
                if edge_idx is None and lig == rec and lig in edge_type_map:
                    idx = edge_type_map[lig]
                    edge_idx = idx if idx < n_edges else None

            if edge_idx is None:
                print(f"Warning: {lr_pair[0]}:{lr_pair[1]} not in edge_type_map, skipping heatmap")
                return

            coords = _get_spatial_coords(adata)
            edge_matrix = pos_edge_probs_np[:, :, edge_idx].copy()

            if threshold > 0:
                edge_matrix[edge_matrix < threshold] = 0.0

            n_c = edge_matrix.shape[0]
            diag = np.diag(edge_matrix)
            out_sum = edge_matrix.sum(axis=1) - diag
            in_sum = edge_matrix.sum(axis=0) - diag
            if score_method == "sum":
                scores = out_sum + in_sum
            elif score_method == "mean":
                scores = (out_sum + in_sum) / max(1, 2 * (n_c - 1))
            elif score_method == "incoming":
                scores = in_sum / max(1, n_c - 1)
            elif score_method == "outgoing":
                scores = out_sum / max(1, n_c - 1)
            else:
                scores = out_sum + in_sum
            scores = np.asarray(scores, dtype=np.float64)
            scores = np.log1p(scores)

            # Nature-like style (font sizes can be overridden)
            base_fs = 8.0 if font_size is None else float(font_size)
            label_fs = base_fs + 0.5
            tick_fs = max(1.0, base_fs - 1.0)

            _set_nature_style(int(round(base_fs)))
            plt.rcParams.update(
                {
                    "font.family": "sans-serif",
                    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
                    "font.size": base_fs,
                    "axes.labelsize": label_fs,
                    "xtick.labelsize": tick_fs,
                    "ytick.labelsize": tick_fs,
                    "axes.linewidth": 0.8,
                    "pdf.fonttype": 42,
                    "ps.fonttype": 42,
                    "svg.fonttype": "none",
                }
            )

            fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=600, facecolor='white')
            ax.set_facecolor('white')

            score_min = float(np.nanmin(scores)) if scores.size else 0.0
            score_max = float(np.nanmax(scores)) if scores.size else 1.0
            if not np.isfinite(score_min):
                score_min = 0.0
            if not np.isfinite(score_max):
                score_max = 1.0
            if score_max <= score_min:
                score_max = score_min + 1e-6

            # For non-negative communication intensity, a sequential map is more publication-friendly.
            # Keep API unchanged: user can still pass any cmap.
            # If using the old default "coolwarm", internally switch to a more suitable Nature-like map.
            cmap_to_use = "magma" if cmap == "coolwarm" else cmap
            try:
                colormap = plt.colormaps[cmap_to_use]
            except (AttributeError, KeyError):
                colormap = plt.cm.get_cmap(cmap_to_use)

            n_cells = len(coords)

            if point_size is None:
                if n_cells > 0:
                    x_range = float(coords[:, 0].max() - coords[:, 0].min()) if n_cells > 1 else 1.0
                    y_range = float(coords[:, 1].max() - coords[:, 1].min()) if n_cells > 1 else 1.0
                    area = x_range * y_range if x_range > 0 and y_range > 0 else 1.0
                    avg_spacing = np.sqrt(area / max(n_cells, 1))

                    # Slightly more conservative sizing for publication figures
                    if n_cells < 1000:
                        point_size = max(8, min(avg_spacing * 1.0, 24))
                    elif n_cells < 5000:
                        point_size = max(5, min(avg_spacing * 0.8, 16))
                    elif n_cells < 10000:
                        point_size = max(3.5, min(avg_spacing * 0.65, 10))
                    else:
                        point_size = max(2.5, min(avg_spacing * 0.5, 7))
                else:
                    point_size = 5.0

            scatter = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=scores,
                cmap=colormap,
                s=point_size,
                alpha=1.0,
                edgecolors='none',
                linewidths=0,
                vmin=score_min,
                vmax=score_max,
                rasterized=True,   # helps for dense plots in vector export
            )

            # Slim, light colorbar
            cbar = plt.colorbar(scatter, ax=ax, pad=0.02, fraction=0.045, shrink=0.9)
            cbar.ax.tick_params(labelsize=tick_fs, colors='#222222', width=0.7, length=2.5)
            cbar.outline.set_linewidth(0.7)
            cbar.outline.set_edgecolor('#222222')
            cbar.locator = MaxNLocator(nbins=5)
            cbar.update_ticks()

            score_method_display = str(score_method).replace("_", r"\_")
            cbar.set_label(
                rf'$log(1+\mathrm{{{score_method_display}}})$',
                fontsize=label_fs,
                color='#222222',
                labelpad=4,
            )

            # Nature-like axes: clean and light
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.8)
            ax.spines['bottom'].set_linewidth(0.8)
            ax.spines['left'].set_color('#222222')
            ax.spines['bottom'].set_color('#222222')

            ax.set_xlabel('Spatial x', fontsize=label_fs, color='#222222', labelpad=4)
            ax.set_ylabel('Spatial y', fontsize=label_fs, color='#222222', labelpad=4)
            ax.tick_params(
                axis='both',
                labelsize=tick_fs,
                width=0.8,
                length=2.5,
                colors='#222222',
                direction='out',
                pad=2,
            )

            ax.set_aspect('equal', adjustable='box')

            # Optional light margins so edge points are not clipped
            if n_cells > 0:
                x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
                y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
                x_pad = max((x_max - x_min) * 0.02, 1e-6)
                y_pad = max((y_max - y_min) * 0.02, 1e-6)
                ax.set_xlim(x_min - x_pad, x_max + x_pad)
                ax.set_ylim(y_min - y_pad, y_max + y_pad)

            plt.tight_layout(pad=0.4)

            save_path.parent.mkdir(parents=True, exist_ok=True)
            if path_wants_svg(save_path):
                configure_matplotlib_svg_for_illustrator()
            plt.savefig(
                save_path,
                dpi=600,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none',
                pad_inches=0.03,
            )
            plt.close(fig)
            print(f"Simple communication heatmap saved to {save_path}")

    def plot_domain_domain_heatmaps(
        self,
        adata: AnnData,
        pos_edge_probs_np: np.ndarray,
        edge_type_map: Dict[str, int],
        domain_key: str,
        lr_filter: List[str],
        threshold: float,
        save_dir: Path,
        font_size: Optional[float] = None,
    ) -> None:
        """Domain-domain heatmaps: total + per-LR (domain_domain_*.svg). lr_filter: list of edge_type_map keys."""
        domains = adata.obs[domain_key].astype(str).to_numpy()
        n_types = pos_edge_probs_np.shape[2]

        lr_items: List[Tuple[str, int]] = [
            (key, edge_type_map[key])
            for key in lr_filter
            if key in edge_type_map and 0 <= edge_type_map[key] < n_types
        ]

        if not lr_items:
            print("Warning: No LR channels resolved for domain-domain heatmaps; skip.")
            return

        # Compute all domain-domain matrices first to determine global vmin/vmax
        all_mats = []
        
        # Total across all selected LR
        total_g = np.zeros((pos_edge_probs_np.shape[0], pos_edge_probs_np.shape[1]), dtype=np.float32)
        for lr_key, idx in lr_items:
            g = pos_edge_probs_np[:, :, idx].astype(np.float32, copy=False)
            if threshold > 0:
                g = np.where(g >= threshold, g, 0.0).astype(np.float32, copy=False)
            total_g += g
        mat_total, dom_names = self.compute_domain_domain_matrix(total_g, domains)
        all_mats.append(mat_total)
        
        # Per LR matrices
        per_lr_mats = []
        for lr_key, idx in lr_items:
            g = pos_edge_probs_np[:, :, idx].astype(np.float32, copy=False)
            if threshold > 0:
                g = np.where(g >= threshold, g, 0.0).astype(np.float32, copy=False)
            mat, dom_names2 = self.compute_domain_domain_matrix(g, domains)
            per_lr_mats.append((lr_key, idx, mat, dom_names2))
            all_mats.append(mat)
        
        # Compute global vmin/vmax across all matrices (after log1p transform)
        all_mats_log1p = [np.log1p(mat) for mat in all_mats]
        global_vmin = min(np.min(mat) for mat in all_mats_log1p if mat.size > 0)
        global_vmax = max(np.max(mat) for mat in all_mats_log1p if mat.size > 0)
        
        # Plot total heatmap (coolwarm, same as cell_type_cell_type)
        self.plot_heatmap_matrix(
            mat_total,
            dom_names,
            save_path=save_dir / "domain_domain_total.svg",
            cmap="GnBu",
            log1p=True,
            vmin=global_vmin,
            vmax=global_vmax,
            font_size=font_size,
        )

        # Plot per LR heatmaps with unified colorbar range (coolwarm, same as cell_type_cell_type)
        for lr_key, idx, mat, dom_names2 in per_lr_mats:
            lr_name = lr_key.replace(":", "-")
            self.plot_heatmap_matrix(
                mat,
                dom_names2,
                save_path=save_dir / f"domain_domain_{self.sanitize_filename(lr_name)}.svg",
                cmap="GnBu",
                log1p=True,
                vmin=global_vmin,
                vmax=global_vmax,
                font_size=font_size,
            )
    
    def plot_cell_type_cell_type_heatmaps(
        self,
        adata: AnnData,
        pos_edge_probs_np: np.ndarray,
        edge_type_map: Dict[str, int],
        cell_type_key: str,
        cell_type_filter: Optional[List[str]],
        lr_filter: List[str],
        threshold: float,
        save_dir: Path,
        font_size: Optional[float] = None,
    ) -> None:
        """Cell type-cell type heatmaps: total + per-LR (cell_type_cell_type_*.svg).

        Parameters
        ----------
        cell_type_filter
            Optional list of cell-type names to include. When provided, the heatmaps
            are computed only on cells whose `adata.obs[cell_type_key]` is in this list.
        lr_filter
            List of `edge_type_map` keys (e.g. "CCL7:CCR2") to aggregate/plot.
        """
        cell_types = adata.obs[cell_type_key].astype(str).to_numpy()

        # Optional cell-type filtering: subset both adata and graph to selected types.
        if cell_type_filter:
            wanted = {str(ct) for ct in cell_type_filter}
            mask = np.isin(cell_types, list(wanted))
            if not np.any(mask):
                print(
                    f"Warning: cell_type_filter={sorted(wanted)} yielded no matching cells; "
                    "skip cell type-cell type heatmaps."
                )
                return
            adata = adata[mask].copy()
            pos_edge_probs_np = pos_edge_probs_np[mask][:, mask, :]
            cell_types = cell_types[mask]
        n_types = pos_edge_probs_np.shape[2]

        lr_items: List[Tuple[str, int]] = [
            (key, edge_type_map[key])
            for key in lr_filter
            if key in edge_type_map and 0 <= edge_type_map[key] < n_types
        ]

        if not lr_items:
            print("Warning: No LR channels resolved for cell type-cell type heatmaps; skip.")
            return

        # Compute all cell type-cell type matrices first to determine global vmin/vmax
        all_mats = []
        
        # Total across all selected LR
        total_g = np.zeros((pos_edge_probs_np.shape[0], pos_edge_probs_np.shape[1]), dtype=np.float32)
        for lr_key, idx in lr_items:
            g = pos_edge_probs_np[:, :, idx].astype(np.float32, copy=False)
            if threshold > 0:
                g = np.where(g >= threshold, g, 0.0).astype(np.float32, copy=False)
            total_g += g
        mat_total, cell_type_names = self.compute_domain_domain_matrix(total_g, cell_types)
        all_mats.append(mat_total)
        
        # Per LR matrices
        per_lr_mats = []
        for lr_key, idx in lr_items:
            g = pos_edge_probs_np[:, :, idx].astype(np.float32, copy=False)
            if threshold > 0:
                g = np.where(g >= threshold, g, 0.0).astype(np.float32, copy=False)
            mat, cell_type_names2 = self.compute_domain_domain_matrix(g, cell_types)
            per_lr_mats.append((lr_key, idx, mat, cell_type_names2))
            all_mats.append(mat)
        
        # Compute global vmin/vmax across ALL matrices (total + per-LR) so same values get same color
        all_mats_log1p = [np.log1p(mat) for mat in all_mats]
        global_vmin = min(np.min(mat) for mat in all_mats_log1p if mat.size > 0) if all_mats_log1p else 0.0
        global_vmax = max(np.max(mat) for mat in all_mats_log1p if mat.size > 0) if all_mats_log1p else 1.0
        global_vmin = float(global_vmin)
        global_vmax = float(global_vmax)
        if global_vmax <= global_vmin:
            global_vmax = global_vmin + 1e-6

        # Plot total heatmap (coolwarm, pcolormesh so each cell is one solid block; matches run_simple_heatmaps)
        self.plot_heatmap_matrix(
            mat_total,
            cell_type_names,
            save_path=save_dir / "cell_type_cell_type_total.svg",
            cmap="GnBu",
            log1p=True,
            vmin=global_vmin,
            vmax=global_vmax,
            font_size=font_size,
        )

        # Plot per LR heatmaps with unified colorbar range, coolwarm cmap, each cell one solid block
        for lr_key, idx, mat, cell_type_names2 in per_lr_mats:
            lr_name = lr_key.replace(":", "-")
            self.plot_heatmap_matrix(
                mat,
                cell_type_names2,
                save_path=save_dir / f"cell_type_cell_type_{self.sanitize_filename(lr_name)}.svg",
                cmap="GnBu",
                log1p=True,
                vmin=global_vmin,
                vmax=global_vmax,
                font_size=font_size,
            )

        # Plot rectangular heatmap: rows = cell type pairs, columns = LR pairs
        n_ct = len(cell_type_names)
        row_labels = [f"{src}→{tgt}" for src in cell_type_names for tgt in cell_type_names]
        col_labels = [lr_key.replace(":", "-") for lr_key, _ in lr_items]
        # Stack (n_ct, n_ct) per LR -> (n_ct*n_ct, n_lr)
        pair_lr_matrix = np.stack(
            [mat.reshape(-1) for _, _, mat, _ in per_lr_mats],
            axis=1,
        ).astype(np.float32)
        # Filter rows with at least one non-zero
        valid_rows = np.any(pair_lr_matrix > 0, axis=1)
        if np.any(valid_rows):
            pair_lr_matrix = pair_lr_matrix[valid_rows]
            row_labels_filtered = [lb for lb, v in zip(row_labels, valid_rows) if v]
            self.plot_rectangular_heatmap(
                pair_lr_matrix,
                row_labels=row_labels_filtered,
                col_labels=col_labels,
                title="",
                save_path=save_dir / "cell_type_pair_lr.svg",
                cmap="GnBu",
                log1p=True,
                vmin=global_vmin,
                vmax=global_vmax,
                font_size=font_size,
            )
