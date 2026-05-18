import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import numpy as np
from scipy.spatial.distance import cdist
import scanpy as sc
import anndata as ad
from typing import Optional

from utils.viz.matplotlib_svg import configure_matplotlib_svg_for_illustrator, path_wants_svg

from .palette_utils import get_custom_palette


class SpatialVisualizer:
    """Spatial domain visualization utility for spatial transcriptomics data."""

    @staticmethod
    def generate_spatial_domain_visualization(
        adata_source: ad.AnnData,
        save_path: str = "spatial_domain_visualization.png",
        palette: Optional[str] = None,
        size_multiplier: float = 1.0,
        figsize: tuple = (8, 6),
    ) -> None:
        """
        Generate spatial domain visualization in grid-based style (pixelated heatmap).
        
        Args:
            adata_source: AnnData object with spatial coordinates in obsm['spatial'] and cluster labels in obs['cluster']
            save_path: Path to save the visualization
            palette: Color palette to use (optional)
        """
        adata_viz = adata_source.copy()
        
        # Extract spatial coordinates from adata_source
        if 'spatial' not in adata_viz.obsm:
            if 'x' in adata_viz.obs and 'y' in adata_viz.obs:
                adata_viz.obsm['spatial'] = np.ascontiguousarray(
                    adata_viz.obs[['x', 'y']].values, dtype=np.float32
                )
            else:
                raise ValueError("No spatial coordinates found in adata_source")
        
        # Get spatial coordinates and cluster labels
        coords = adata_viz.obsm['spatial']
        clusters = adata_viz.obs['cluster'].values
        
        # Get unique clusters and their categories
        cluster_categories = adata_viz.obs['cluster'].cat.categories
        n_categories = len(cluster_categories)
        print(f"Visualizing {adata_viz.n_obs} points with {n_categories} domains in grid style")
        
        # Create cluster to index mapping
        cluster_to_idx = {cat: idx for idx, cat in enumerate(cluster_categories)}
        cluster_indices = np.array([cluster_to_idx[clust] for clust in clusters])
        
        # Direct visualization: each point gets its own square
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        
        # Calculate data range
        x_range = x_max - x_min
        y_range = y_max - y_min
        n_points = len(coords)
        
        # Calculate square size based on actual minimum nearest neighbor distance
        # This ensures no gaps between points
        if n_points > 1 and x_range > 0 and y_range > 0:
            # Calculate minimum nearest neighbor distance efficiently
            # For large datasets, use sampling; for small datasets, calculate all distances
            if n_points > 10000:
                # For very large datasets, sample points to estimate min distance
                sample_size = min(5000, n_points)
                sample_indices = np.random.choice(n_points, size=sample_size, replace=False)
                sample_coords = coords[sample_indices]
                # Calculate distances within sample
                distances_sample = cdist(sample_coords, sample_coords)
                np.fill_diagonal(distances_sample, np.inf)  # Exclude self-distances
                min_nearest_dist = np.min(distances_sample[distances_sample > 0])
            elif n_points > 2000:
                # For medium datasets, calculate distances for a subset of points
                # Check nearest neighbors for each point (k=10 should be enough)
                min_dists = []
                for i in range(0, n_points, max(1, n_points // 100)):  # Sample every Nth point
                    point = coords[i:i+1]
                    distances = cdist(point, coords)
                    distances[0, i] = np.inf  # Exclude self
                    min_dists.append(np.min(distances[distances > 0]))
                min_nearest_dist = np.min(min_dists)
            else:
                # For small datasets, calculate all pairwise distances
                distances = cdist(coords, coords)
                np.fill_diagonal(distances, np.inf)  # Exclude self-distances
                min_nearest_dist = np.min(distances[distances > 0])
            
            # Use minimum nearest neighbor distance as base size
            # Default size_multiplier=1.0 gives size = min_nearest_dist * 1.2 (ensures no gaps)
            # User can increase size_multiplier to make points larger
            base_size = min_nearest_dist * 1.2  # 1.2x ensures no gaps between adjacent points
            square_size = base_size * size_multiplier
            print(f"  Calculated min nearest neighbor distance: {min_nearest_dist:.4f}")
            print(f"  Square size (base * {size_multiplier}): {square_size:.4f}")
        else:
            # Fallback: use a small fraction of the data range
            if x_range > 0 and y_range > 0:
                square_size = min(x_range, y_range) / max(10, np.sqrt(n_points)) * size_multiplier
            else:
                square_size = 1.0 * size_multiplier
        
        print(f"  Direct visualization: {n_points} points, each point gets its own square")
        
        # Get color palette - Nature style: use custom palette
        if palette is None:
            colors = get_custom_palette(n_categories)
        elif isinstance(palette, str):
            cmap = plt.get_cmap(palette)
            colors = [mcolors.to_hex(cmap(i / max(n_categories - 1, 1))) for i in range(n_categories)]
        else:
            colors = [mcolors.to_hex(c) if not isinstance(c, str) else c for c in palette]
        
        # Setup font and matplotlib parameters - Nature style
        font_name = 'Arial'
        
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['figure.titlesize'] = 12
        
        # Create figure with Nature publication quality
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=600)
        fig.patch.set_facecolor('white')
        
        # Draw each point as a square directly at its spatial coordinates
        for i in range(n_points):
            cluster_idx = cluster_indices[i]
            x_coord = coords[i, 0]
            y_coord = coords[i, 1]
            
            # Calculate square position (centered at point coordinates)
            x_start = x_coord - square_size / 2
            y_start = y_coord - square_size / 2
            
            # Get color for this cluster
            color_hex = colors[cluster_idx]
            color_rgb = mcolors.to_rgb(color_hex)
            
            # Draw rectangle without edge for seamless Nature style
            # Use high-quality rendering for publication
            rect = Rectangle((x_start, y_start), 
                            square_size, square_size,
                            facecolor=color_rgb, 
                            edgecolor='none', 
                            linewidth=0,
                            antialiased=True)  # Enable antialiasing for smooth edges
            ax.add_patch(rect)
        
        # Set display limits to include all points with Nature-style padding
        # Use square_size to calculate padding (minimal padding for clean look)
        padding = square_size / 2
        display_x_min = x_min - padding
        display_x_max = x_max + padding
        display_y_min = y_min - padding
        display_y_max = y_max + padding
        
        # Nature style: Clean axes - no labels, ticks, or spines
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        # Remove all spines for clean Nature style
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Set axis limits and background - Nature style: clean white background
        ax.set_xlim(display_x_min, display_x_max)
        ax.set_ylim(display_y_min, display_y_max)
        ax.set_aspect('equal')
        ax.set_facecolor('white')
        
        # No legend (matching Nature publication style)
        
        # Nature style: Optimized margins for publication quality
        # Use tight layout with minimal padding
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        
        # Save figure with Nature publication quality settings
        if path_wants_svg(save_path):
            configure_matplotlib_svg_for_illustrator()
        plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', pad_inches=0.05)
        plt.close()
        
        print(f"Spatial domain visualization (grid style) saved to {save_path}")
