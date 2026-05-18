import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import torch
import pandas as pd
import scanpy as sc
import anndata as ad
from typing import Optional

from utils.viz.matplotlib_svg import configure_matplotlib_svg_for_illustrator, path_wants_svg
from sklearn.neighbors import KernelDensity

from utils.tools.seed_utils import active_base_seed


class UMAPVisualizer:
    """
    UMAP visualization utility for spatial gene expression data.
    """

    @staticmethod
    def generate_visualization(
        node_features: torch.Tensor,
        adata_source: ad.AnnData,
        save_path: str = "umap_visualization.png",
        palette: Optional[str] = None,
        add_kde_contour: bool = False,
        add_cluster_labels: bool = False,
        point_size: Optional[float] = None,
        n_neighbors: int = 40,
        color_key: str = "cluster",
    ) -> None:
        """
        Generate UMAP visualization of node features using scanpy (similar to SpatialGlue style).
        Args:
            node_features: Learned node features (n_nodes, feature_dim)
            adata_source: AnnData object with cluster/community labels in obs[color_key]
            save_path: Path to save the UMAP plot
            palette: Color palette to use (optional)
            add_kde_contour: Whether to add KDE contour lines for each cluster (default: False)
            add_cluster_labels: Whether to add cluster labels at cluster centers (default: False)
            point_size: Size of points in the scatter plot. If None, uses default size (40 for scanpy, 80 for enhanced points)
            color_key: obs column for coloring (default 'cluster', e.g. community labels from Louvain)
        Returns:
            None
        """
        if color_key not in adata_source.obs:
            raise KeyError(f"color_key '{color_key}' not found in adata.obs. Available: {list(adata_source.obs.columns)}")
        # Convert tensor to numpy and create AnnData
        features_np = node_features.detach().cpu().numpy()
        adata = ad.AnnData(features_np, obs=adata_source.obs.copy())
        adata.obsm['features'] = features_np
        adata.obs['color_key'] = pd.Categorical(adata.obs[color_key].astype(str))
        
        # Compute neighbors and UMAP using scanpy (similar to SpatialGlue style)
        # Set random_state for reproducibility
        sc.pp.neighbors(adata, use_rep='features', n_neighbors=n_neighbors, random_state=active_base_seed())
        sc.tl.umap(adata, random_state=active_base_seed())
        
        # Setup font and scanpy parameters - Nature style
        # Try Arial first (Nature standard), fallback to DejaVu Sans
        font_name = 'Arial'
        try:
            # Test if Arial is available
            from matplotlib import font_manager
            available_fonts = [f.name for f in font_manager.fontManager.ttflist]
            if 'Arial' not in available_fonts:
                font_name = 'DejaVu Sans'
        except:
            font_name = 'DejaVu Sans'
        
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 9
        plt.rcParams['figure.titlesize'] = 12
        
        # Increase DPI for publication quality
        sc.settings.set_figure_params(dpi=600, facecolor='white', frameon=False, 
                                     vector_friendly=True, fontsize=10, figsize=(8, 6))
        
        # Create figure with Nature-style dimensions and high DPI
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=600)
        fig.patch.set_facecolor('white')
        
        # Get color palette
        n_categories = len(adata.obs['color_key'].cat.categories)
        # When too many categories, hide legend to avoid huge figure size (PIL limit 2^16 per side)
        show_legend = n_categories <= 50
        palette = palette or ('tab10' if n_categories <= 10 else ('tab20' if n_categories <= 20 else 'Set3'))
        
        # Use original cluster labels directly (no conversion to A1, A2, A3...)
        cluster_categories = adata.obs['color_key'].cat.categories
        adata.obs['legend_label'] = adata.obs['color_key'].astype(str)
        adata.obs['legend_label'] = pd.Categorical(adata.obs['legend_label'], categories=[str(cat) for cat in cluster_categories])
        
        # Determine point size: Nature style - moderate size with slight transparency
        scanpy_point_size = point_size if point_size is not None else 30
        enhanced_point_size = point_size if point_size is not None else 30
        
        # Plot UMAP with Nature-style points
        legend_loc = 'right margin' if show_legend else 'none'
        sc.pl.umap(adata, color='legend_label', ax=ax, title='', s=scanpy_point_size, show=False,
                   frameon=False, palette=palette, legend_loc=legend_loc,
                   legend_fontsize=9, legend_fontweight='normal', return_fig=False)
        
        # Enhance scatter points - Nature style: no edges, slight transparency for overlap visibility
        for collection in ax.collections:
            # Remove edge colors (no borders) for clean look
            collection.set_edgecolors('none')
            collection.set_linewidths(0)
            # Slight transparency for better overlap visualization (Nature style)
            collection.set_alpha(0.85)
            # Set point sizes - moderate size for Nature style
            if hasattr(collection, '_sizes') and len(collection._sizes) > 0:
                # Keep original sizes, just adjust slightly
                collection.set_sizes([s * 1.0 for s in collection._sizes])
            else:
                collection.set_sizes([enhanced_point_size])
        
        # Add KDE contours for each cluster if requested
        if add_kde_contour:
            for cluster_label in adata.obs['legend_label'].cat.categories:
                # Get UMAP coordinates for this cluster
                mask = adata.obs['legend_label'] == cluster_label
                x = adata.obsm["X_umap"][mask, 0]
                y = adata.obsm["X_umap"][mask, 1]
                
                if len(x) > 0:
                    # Create grid for KDE
                    x_min, x_max = x.min() - 0.5, x.max() + 0.5
                    y_min, y_max = y.min() - 0.5, y.max() + 0.5
                    xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
                    
                    xy_train = np.vstack([x, y]).T
                    xy_test = np.vstack([xx.ravel(), yy.ravel()]).T
                    
                    # Kernel density estimation
                    kde = KernelDensity(bandwidth=0.5, metric='euclidean')
                    kde.fit(xy_train)
                    
                    # Density
                    Z = np.exp(kde.score_samples(xy_test))
                    Z = Z.reshape(xx.shape)
                    
                    # Draw contour (Nature style: subtle, professional)
                    ax.contour(
                        xx, yy, Z,
                        levels=[Z.max() * 0.1],
                        colors='#666666',  # Subtle gray
                        linestyles='--',
                        linewidths=1.0,
                        alpha=0.6
                    )
        
        # Add cluster labels at cluster centers if requested
        if add_cluster_labels:
            for cluster_label in adata.obs['legend_label'].cat.categories:
                mask = adata.obs['legend_label'] == cluster_label
                x = np.median(adata.obsm["X_umap"][mask, 0])
                y = np.median(adata.obsm["X_umap"][mask, 1])
                ax.text(
                    x, y, cluster_label,
                    fontsize=9,
                    ha="center",
                    va="center",
                    color="black",
                    fontname=font_name,
                    fontweight='normal',
                    bbox=dict(
                        facecolor='white',
                        edgecolor='#CCCCCC',
                        alpha=0.8,
                        boxstyle="round,pad=0.3",
                        linewidth=0.5
                    )
                )
        
        # Remove title for cleaner look
        ax.set_title('', pad=0)
        
        # Nature style: Clean axes with minimal ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Nature style: Remove all spines for clean look
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Remove grid for cleaner background
        ax.grid(False)
        ax.set_facecolor('white')
        
        # Get axis limits for arrow placement
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        # Calculate arrow parameters based on data range and aspect ratio
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        aspect_ratio = (fig.get_size_inches()[0] / fig.get_size_inches()[1]) * (y_range / x_range)
        
        # Arrow length (8% of data range for Nature style - more subtle)
        dx = x_range * 0.08 * aspect_ratio
        dy = y_range * 0.08
        
        # Arrow head parameters (proportional to arrow length, Nature style)
        head_width = dy * 0.12
        head_length = dx * 0.12
        
        # Draw X-axis arrow (Nature style: clean, professional)
        ax.arrow(
            xlim[0], ylim[0],
            dx, 0,
            head_width=head_width,
            head_length=head_length,
            fc='black',
            ec='black',
            linewidth=1.2,
            length_includes_head=True
        )
        
        # Draw Y-axis arrow (Nature style: clean, professional)
        ax.arrow(
            xlim[0], ylim[0],
            0, dy,
            head_width=head_width,
            head_length=head_length,
            fc='black',
            ec='black',
            linewidth=1.2,
            length_includes_head=True
        )
        
        # Add axis labels (Nature style: clear, professional typography)
        label_offset = dx * 0.2
        ax.text(
            xlim[0] + dx/2,
            ylim[0] - label_offset,
            'UMAP 1',
            ha='center',
            va='top',
            fontsize=11,
            fontname=font_name,
            fontweight='normal',
            color='black'
        )
        ax.text(
            xlim[0] - label_offset,
            ylim[0] + dy/2,
            'UMAP 2',
            rotation=90,
            ha='right',
            va='center',
            fontsize=11,
            fontname=font_name,
            fontweight='normal',
            color='black'
        )
        
        # Recreate legend with Nature style - clean and professional (skip when too many categories)
        legend = ax.get_legend() if show_legend else None
        if legend and show_legend:
            # Get handles and labels from existing legend
            handles, labels = ax.get_legend_handles_labels()
            if not handles:
                # Fallback: try to get from legend directly
                try:
                    handles = list(legend.legendHandles)
                    labels = [t.get_text() for t in legend.get_texts()]
                except:
                    # If that fails, create handles from categories
                    categories = adata.obs['legend_label'].cat.categories
                    handles = []
                    labels = list(categories)
                    # Get colors from palette
                    import matplotlib.cm as cm
                    if isinstance(palette, list):
                        colors = palette[:len(categories)]
                    else:
                        cmap = cm.get_cmap(palette)
                        colors = [cmap(i / max(len(categories) - 1, 1)) for i in range(len(categories))]
                    for cat, color in zip(categories, colors):
                        handle = mlines.Line2D([0], [0], marker='o', color='w', 
                                               markerfacecolor=color, markersize=7,
                                               markeredgecolor='none', markeredgewidth=0,
                                               alpha=0.85)
                        handles.append(handle)
            
            # Remove old legend and create new one with Nature style
            legend.remove()
            legend = ax.legend(
                handles, labels,
                loc='center left',
                bbox_to_anchor=(1.02, 0.5),
                frameon=True,
                fontsize=9,
                labelspacing=0.8,  # Nature style: moderate spacing
                handletextpad=0.5,
                columnspacing=1.0,
                borderpad=0.5,
                framealpha=1.0
            )
        
        # Enhance legend style - Nature publication quality
        if legend:
            legend.get_frame().set_linewidth(0.8)
            legend.get_frame().set_edgecolor('#333333')  # Subtle gray border
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(1.0)
            for text in legend.get_texts():
                text.set_fontweight('normal')
                text.set_fontname(font_name)
                text.set_fontsize(9)
                text.set_color('black')
        
        # Save figure with Nature publication quality
        # When many categories we hid legend so figure size is bounded; cap dpi to avoid PIL limit (2^16 px)
        save_dpi = 600
        if n_categories > 50:
            plt.tight_layout(pad=0.5)
            # Cap dpi so width/height in pixels stay < 65535 (bbox can grow with tight_layout)
            w_in, h_in = fig.get_size_inches()
            max_px = 65000
            if w_in * save_dpi > max_px or h_in * save_dpi > max_px:
                save_dpi = int(max_px / max(w_in, h_in))
            ax.text(0.99, 0.01, f'N={n_categories} clusters', transform=ax.transAxes,
                    fontsize=8, ha='right', va='bottom', color='gray')
        else:
            plt.tight_layout(rect=[0, 0, 0.95, 1])  # Leave space for legend on right
        if path_wants_svg(save_path):
            configure_matplotlib_svg_for_illustrator()
        plt.savefig(save_path, dpi=save_dpi, bbox_inches='tight', facecolor='white',
                    edgecolor='none', pad_inches=0.1)
        plt.close()
        
        print(f"UMAP visualization saved to {save_path}")
