"""Analysis: CCC visualization, community detection, differential, network metrics, radar/heatmap, etc."""

from .domain_viz import DomainVisualizer, require_domain_obs
from .cell_type_communication import CellTypeCommunicationComputer
from .differential_analysis import DifferentialAnalyzer, detect_communities_louvain
from .aggregated_heatmap_viz import AggregatedHeatmapVisualizer
from .lr_spatial_viz import LigandReceptorSpatialVisualizer, get_colour_scheme
from .strength_distance_viz import StrengthDistanceVisualizer
from .spot_level_metrics import compute_spot_level_metrics
from .alluvial_viz import AlluvialVisualizer
from .sender_receiver_stacked_bar_viz import SenderReceiverStackedBarVisualizer
from .alluvial_icicle_viz import (
    plot_alluvial_and_icicle_per_cell_type_pair,
    plot_alluvial_and_icicle_per_domain,
)

__all__ = [
    "DomainVisualizer",
    "require_domain_obs",
    "CellTypeCommunicationComputer",
    "DifferentialAnalyzer",
    "AggregatedHeatmapVisualizer",
    "LigandReceptorSpatialVisualizer",
    "StrengthDistanceVisualizer",
    "compute_spot_level_metrics",
    "detect_communities_louvain",
    "get_colour_scheme",
    "AlluvialVisualizer",
    "SenderReceiverStackedBarVisualizer",
    "plot_alluvial_and_icicle_per_cell_type_pair",
    "plot_alluvial_and_icicle_per_domain",
]
