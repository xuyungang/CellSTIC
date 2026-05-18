"""
Map developmental stage labels to plot coordinates.

Embryonic days stored as numeric strings (e.g. "12.5") use float x positions and
``E{value}`` tick labels. Non-numeric labels (e.g. ``develop_44``) use ordinal
positions 0..n-1; pass ``full_order`` (typically the analysis ``stages`` list)
to fix left-to-right order.
"""

from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


def stage_axis_from_present(
    present_stages: Iterable,
    full_order: Optional[List[str]] = None,
) -> Tuple[Dict[str, float], List[str], List[float], List[str], str]:
    """
    Build x-coordinates and tick labels for stage-based plots.

    Returns:
        stage_to_x: maps each stage string to its x coordinate
        ordered_stages: stage strings in left-to-right order (for pivots / sorting)
        xticks: positions for ``ax.set_xticks``
        xticklabels: strings for ``ax.set_xticklabels`` (includes ``E`` prefix when numeric)
        xlabel: suggested x-axis label (``Stage (E)`` or ``Stage``)
    """
    present = {str(s) for s in present_stages if pd.notna(s) and str(s) not in ("nan", "")}
    if not present:
        return {}, [], [], [], "Stage"

    nums = {s: pd.to_numeric(s, errors="coerce") for s in present}
    all_numeric = all(v == v for v in nums.values())  # no NaN

    if all_numeric:
        ordered_stages = sorted(present, key=lambda s: float(nums[s]))
        stage_to_x = {s: float(nums[s]) for s in ordered_stages}
        xs = [stage_to_x[s] for s in ordered_stages]
        labels = [f"E{x:g}" for x in xs]
        return stage_to_x, ordered_stages, xs, labels, "Stage (E)"

    if full_order:
        ordered = [str(x) for x in full_order if str(x) in present]
        ordered.extend(sorted(present - set(ordered)))
    else:
        ordered = sorted(present)

    stage_to_x = {s: float(i) for i, s in enumerate(ordered)}
    xs = [float(i) for i in range(len(ordered))]
    labels = list(ordered)
    return stage_to_x, ordered, xs, labels, "Stage"
