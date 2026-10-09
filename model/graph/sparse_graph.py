"""Sparse graph containers used by the CellSTIC graph pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Tuple

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class SparseTypedGraph:
    """A stack of sparse ``(source, target)`` adjacency channels.

    Keeping channels as CSR matrices avoids allocating ``n_cells ** 2`` entries
    while retaining the old logical ``(n_cells, n_cells, n_types)`` shape.
    """

    channels: Tuple[sparse.csr_matrix, ...]
    n_nodes: int

    def __post_init__(self) -> None:
        if not isinstance(self.n_nodes, (int, np.integer)) or self.n_nodes < 1:
            raise ValueError(
                f"n_nodes must be a positive integer, got {self.n_nodes!r}"
            )
        normalized_channels = []
        for index, source in enumerate(self.channels):
            channel = sparse.csr_matrix(source, copy=True)
            if channel.shape != (self.n_nodes, self.n_nodes):
                raise ValueError(
                    f"channels[{index}] must have shape "
                    f"({self.n_nodes}, {self.n_nodes}), got {channel.shape}"
                )
            channel.sum_duplicates()
            channel.eliminate_zeros()
            channel.sort_indices()
            if not np.all(np.isfinite(channel.data)):
                raise ValueError(f"channels[{index}] contains non-finite values")
            normalized_channels.append(channel)
        object.__setattr__(self, "channels", tuple(normalized_channels))

    @classmethod
    def from_channels(
        cls, channels: Iterable[sparse.spmatrix], n_nodes: int
    ) -> "SparseTypedGraph":
        return cls(tuple(channels), n_nodes)

    @classmethod
    def from_dense(cls, values: np.ndarray) -> "SparseTypedGraph":
        array = np.asarray(values)
        if array.ndim != 3 or array.shape[0] != array.shape[1]:
            raise ValueError("dense typed graph must have shape (n, n, n_types)")
        return cls.from_channels(
            (sparse.csr_matrix(array[:, :, index]) for index in range(array.shape[2])),
            array.shape[0],
        )

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.n_nodes, self.n_nodes, len(self.channels)

    @property
    def nnz(self) -> int:
        return sum(channel.nnz for channel in self.channels)

    def copy(self) -> "SparseTypedGraph":
        return SparseTypedGraph(
            tuple(channel.copy() for channel in self.channels), self.n_nodes
        )

    def __len__(self) -> int:
        return len(self.channels)

    def __iter__(self) -> Iterator[sparse.csr_matrix]:
        return iter(self.channels)

    def __getitem__(self, index: int) -> sparse.csr_matrix:
        return self.channels[index]

    def union(self) -> sparse.csr_matrix:
        """Return a binary CSR adjacency containing edges from every channel."""
        result = sparse.csr_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        for channel in self.channels:
            result = result.maximum(channel)
        if result.nnz:
            result.data[:] = 1.0
        return result


def as_sparse_typed_graph(graph) -> SparseTypedGraph:
    """Normalize legacy dense typed graphs to the sparse representation."""
    if isinstance(graph, SparseTypedGraph):
        return graph
    return SparseTypedGraph.from_dense(np.asarray(graph))


def sparse_values(
    matrix: sparse.spmatrix, rows: np.ndarray, cols: np.ndarray
) -> np.ndarray:
    """Read pairwise values from a sparse matrix as a flat NumPy array."""
    if len(rows) == 0:
        return np.empty(0, dtype=matrix.dtype)
    return np.asarray(matrix.tocsr()[rows, cols]).reshape(-1)
