"""
Filter utilities for time-sequence analysis.

Shared by all scripts under utils/time/ for annotation_filter and lr_filter.
"""

from typing import List, Optional


def lr_match(lr_stem: str, lr_filter: Optional[List[str]]) -> bool:
    """
    Check if LR stem matches filter. Normalizes underscore/hyphen for comparison.

    Args:
        lr_stem: CSV filename stem (e.g. F2_F2r).
        lr_filter: List of LR pairs (e.g. ["F2-F2r", "Lpar3-Adgre5"]). None/empty = match all.

    Returns:
        True if lr_filter is empty/None or lr_stem matches any entry.
    """
    if not lr_filter:
        return True
    stem_norm = lr_stem.replace("_", "-")
    for f in lr_filter:
        if f.replace("_", "-") == stem_norm:
            return True
    return False


def annotation_pass(organ: str, annotation_filter: Optional[List[str]]) -> bool:
    """
    Check if organ passes annotation filter.

    Args:
        organ: Organ name (e.g. Brain, Liver).
        annotation_filter: List of organs to include. None/empty = pass all.

    Returns:
        True if organ should be processed.
    """
    if not annotation_filter:
        return True
    return organ in annotation_filter
