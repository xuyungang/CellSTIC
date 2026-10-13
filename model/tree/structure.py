"""Shared branching rule for explicitly requested hierarchy heights."""

from numbers import Integral
from typing import List


def children_counts_for_height(n_leaves: int, height: int) -> List[int]:
    """Return balanced branching counts, including the leaf transition.

    A height-four tree with 27 leaves has branching [3, 3, 3], hence
    1, 3, 9, and 27 channels from root to leaves.
    """
    if isinstance(height, bool) or not isinstance(height, Integral) or height < 2:
        raise ValueError(f"tree height must be an integer >= 2, got {height!r}")
    if n_leaves < 1:
        raise ValueError("n_leaves must be positive")
    branch = 1
    while branch ** (height - 1) < n_leaves:
        branch += 1
    parent_count = branch ** (height - 2)
    last_count = (n_leaves + parent_count - 1) // parent_count
    return [branch] * (height - 2) + [last_count]
