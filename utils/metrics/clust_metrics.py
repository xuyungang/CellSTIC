"""
Clustering evaluation metrics utility for SpaGEM.

This module provides comprehensive clustering evaluation metrics for labeled data.
"""

import numpy as np
import torch
from typing import Dict, Optional, Union
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    mutual_info_score,
    homogeneity_score,
    v_measure_score,
)


class ClusteringMetrics:
    """
    Utility class for calculating clustering evaluation metrics.
    
    Supports labeled data evaluation:
    - External metrics (AMI, NMI, ARI, Homogeneity, Mutual_info, V-measure)
    """
    
    @staticmethod
    def calculate_metrics(
        features: Union[torch.Tensor, np.ndarray],
        true_labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
        cluster_labels: Optional[Union[torch.Tensor, np.ndarray]] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive clustering evaluation metrics.
        
        For labeled data (when true_labels is provided), computes external metrics:
        - AMI (Adjusted Mutual Information): Adjusted version of MI accounting for chance
        - NMI (Normalized Mutual Information): Normalized information-theoretic measure
        - ARI (Adjusted Rand Index): Measures similarity between clusters
        - Homogeneity: Each cluster contains only members of a single class
        - Mutual_info: Raw mutual information between labels
        - V-measure: Harmonic mean of homogeneity and completeness
        
        Args:
            features: Feature matrix (n_samples, n_features)
            true_labels: True labels for external metrics evaluation (optional)
            cluster_labels: Pre-computed cluster labels (n_samples,) - if provided, skips clustering
            
        Returns:
            Dictionary containing metric names and their values
        """
        # Convert to numpy
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        features = np.asarray(features)
        
        # Use pre-computed cluster labels if provided, otherwise perform clustering
        if isinstance(cluster_labels, torch.Tensor):
            cluster_labels = cluster_labels.detach().cpu().numpy()
        cluster_labels = np.asarray(cluster_labels)
        if cluster_labels.shape[0] != features.shape[0]:
            raise ValueError(f"cluster_labels length ({cluster_labels.shape[0]}) must match features size ({features.shape[0]})")
        
        # Convert true_labels if provided
        if true_labels is not None:
            if isinstance(true_labels, torch.Tensor):
                true_labels = true_labels.detach().cpu().numpy()
            true_labels = np.asarray(true_labels)
        
        # Check for valid clustering
        unique_labels = np.unique(cluster_labels)
        n_clusters_actual = len(unique_labels)
        n_samples = len(cluster_labels)
        
        if n_clusters_actual < 2:
            raise ValueError("At least 2 clusters are required for metric calculation")
        
        if n_samples < n_clusters_actual:
            raise ValueError("Number of samples must be greater than number of clusters")
        
        metrics = {}
        
        # External metrics (labeled data)
        if true_labels is not None:
            true_labels = np.asarray(true_labels)
            
            # AMI (Adjusted Mutual Information): Range [0, 1], higher is better
            try:
                metrics['AMI'] = float(adjusted_mutual_info_score(true_labels, cluster_labels))
            except Exception as e:
                metrics['AMI'] = np.nan
            
            # NMI (Normalized Mutual Information): Range [0, 1], higher is better
            try:
                metrics['NMI'] = float(normalized_mutual_info_score(true_labels, cluster_labels))
            except Exception as e:
                metrics['NMI'] = np.nan
            
            # ARI (Adjusted Rand Index): Range [-1, 1], higher is better
            try:
                metrics['ARI'] = float(adjusted_rand_score(true_labels, cluster_labels))
            except Exception as e:
                metrics['ARI'] = np.nan
            
            # Homogeneity: Range [0, 1], higher is better
            try:
                metrics['Homogeneity'] = float(homogeneity_score(true_labels, cluster_labels))
            except Exception as e:
                metrics['Homogeneity'] = np.nan
            
            # Mutual_info: Raw mutual information, higher is better
            try:
                metrics['Mutual_info'] = float(mutual_info_score(true_labels, cluster_labels))
            except Exception as e:
                metrics['Mutual_info'] = np.nan
            
            # V-measure: Range [0, 1], higher is better
            try:
                metrics['V-measure'] = float(v_measure_score(true_labels, cluster_labels))
            except Exception as e:
                metrics['V-measure'] = np.nan
        
        # Print metrics directly after calculation
        ClusteringMetrics.print_metrics(metrics)
        
        return metrics
    
    @staticmethod
    def print_metrics(
        metrics: Dict[str, float],
        title: str = "Clustering Metrics",
        precision: int = 4
    ) -> None:
        """
        Print clustering metrics in a formatted, readable way.
        
        Args:
            metrics: Dictionary of metric names and values
            title: Title for the metrics report
            precision: Number of decimal places to display
        """
        print(f"\n{'='*70}")
        print(f"{title:^70}")
        print(f"{'='*70}")
        
        # External metrics (if available)
        if any(key in metrics for key in ['AMI', 'NMI', 'ARI', 'Homogeneity', 'Mutual_info', 'V-measure']):
            print(f"\n🏷️  External Metrics (Labeled Data):")
            if 'AMI' in metrics and not np.isnan(metrics['AMI']):
                print(f"  • AMI (Adjusted Mutual Info):    {metrics['AMI']:.{precision}f} [range: 0 to 1, higher better]")
            if 'NMI' in metrics and not np.isnan(metrics['NMI']):
                print(f"  • NMI (Normalized Mutual Info): {metrics['NMI']:.{precision}f} [range: 0 to 1, higher better]")
            if 'ARI' in metrics and not np.isnan(metrics['ARI']):
                print(f"  • ARI (Adjusted Rand Index):     {metrics['ARI']:.{precision}f} [range: -1 to 1, higher better]")
            if 'Homogeneity' in metrics and not np.isnan(metrics['Homogeneity']):
                print(f"  • Homogeneity:                  {metrics['Homogeneity']:.{precision}f} [range: 0 to 1, higher better]")
            if 'Mutual_info' in metrics and not np.isnan(metrics['Mutual_info']):
                print(f"  • Mutual_info:                   {metrics['Mutual_info']:.{precision}f} [higher better]")
            if 'V-measure' in metrics and not np.isnan(metrics['V-measure']):
                print(f"  • V-measure:                     {metrics['V-measure']:.{precision}f} [range: 0 to 1, higher better]")
        
        print(f"{'='*70}\n")
