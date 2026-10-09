"""
Color palette utilities for visualization.

This module provides color palette generation functions for various visualization needs.
"""


def get_custom_palette(n_categories: int) -> list:
    """
    Get custom color palette with high contrast and distinctiveness.
    Uses 18 carefully selected colors from a professional color scheme.
    
    Args:
        n_categories: Number of categories/clusters to color
        
    Returns:
        List of color hex codes
    """
    # 18-color custom palette with high contrast
    custom_colors = [
        '#3366CC',  # 1-ctx: dark blue
        '#FF9933',  # 2-acb: orange
        '#009966',  # 3-ctx: dark green
        '#E63946',  # 4-cp: red
        '#9966CC',  # 5-ctx/aca: purple
        '#885649',  # 6-ctx: dark brown
        '#F17AAB',  # 7-ctx: light pink
        '#BCC959',  # 8-aca: light yellow-green
        '#66CCFF',  # 9-vl: light blue
        '#C8C8CD',  # 10-ctx: light gray
        '#FFB870',  # 11-ccg/aco: light orange
        '#99E699',  # 12-ls: light green
        '#F78888',  # 13-cp: light red
        '#C5B0E5',  # 14-ctx: light purple
        '#C9A087',  # 15-aca: light brown
        '#FFD6E0',  # 16-ctx: pale pink
        '#F0F389',  # 17-ctx: pale yellow
        '#CCEEFF',  # 18-lpo: very light blue
    ]
    
    # Return colors based on number of categories needed
    if n_categories <= len(custom_colors):
        return custom_colors[:n_categories]
    else:
        # If more categories than colors, cycle through the palette
        return (custom_colors * ((n_categories // len(custom_colors)) + 1))[:n_categories]
