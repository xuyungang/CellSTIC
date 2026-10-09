"""Data summaries used by the dataset tutorials."""

from __future__ import annotations

import numpy as np

import pandas as pd

from scipy import sparse

from .result_io import clean_matrix, lr_matrices

def group_communication(result, matrix, group_key):
    """Sum directed weights by sender and receiver annotation, without averaging."""
    labels = result.obs[group_key]
    if labels.isna().any():
        raise ValueError(f"Missing labels in {group_key}")
    codes, names = pd.factorize(labels.astype(str), sort=True)
    membership = sparse.csr_matrix((np.ones(result.n_obs), (np.arange(result.n_obs), codes)),
                                   shape=(result.n_obs, len(names)))
    matrix = clean_matrix(matrix)
    grouped = (membership.T @ matrix @ membership).toarray()
    return pd.DataFrame(grouped, index=pd.Index(names, name="sender"), columns=pd.Index(names, name="receiver"))

def receiver_scores(result, pairs=None, *, rank=False):
    """Incoming column sums, optionally ranked among strictly positive receivers."""
    matrices = lr_matrices(result)
    values = {}
    for pair in (pairs if pairs is not None else matrices):
        vector = np.asarray(matrices[pair].sum(axis=0)).ravel()
        if rank:
            positive = vector > 0
            vector[positive] = pd.Series(vector[positive]).rank(pct=True).to_numpy()
        values[pair] = vector
    return values

def lr_summary(result):
    """Report the size and total weight of every leaf channel."""
    return pd.DataFrame([dict(lr_pair=pair, edges=m.nnz, strength=float(m.sum()),
                              senders=int(np.count_nonzero(np.diff(m.indptr))),
                              receivers=int(np.count_nonzero(np.diff(m.tocsc().indptr))))
                         for pair, m in lr_matrices(result).items()])

def expression_values(path, names, *, obs_names=None):
    """Read a small gene panel by observation ID without loading all RNA counts."""
    import anndata as ad
    source = ad.read_h5ad(path, backed="r")
    try:
        ids = source.obs_names if obs_names is None else pd.Index(obs_names)
        positions = source.obs_names.get_indexer(ids)
        if (positions < 0).any() or not source.obs_names.is_unique:
            raise ValueError("Expression and result observations cannot be aligned")
        # Slice columns first: backed sparse matrices require ordered row slices.
        selected = source[:, list(names)].X
        selected = selected.toarray() if sparse.issparse(selected) else np.asarray(selected)
        return {name: np.log1p(selected[positions, i]) for i, name in enumerate(names)}
    finally:
        source.file.close()
