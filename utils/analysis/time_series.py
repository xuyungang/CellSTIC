"""Stage-wise communication analysis for independently fitted spatial samples.

Rows and columns of every graph are senders and receivers. Stage order comes
from the supplied sample table. Only leaf LR graphs enter totals; ancestor
channels and calibration statistics are excluded. No expression arrays or
paper-specific plotting modules are loaded.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .result_io import load_result, lr_matrices
from .network_metrics import network_metrics


class TimeSeriesAnalysis:
    """Summarize an ordered sample table with sample, stage, group and path columns.

    ``control=True`` marks a reference sample in the exported tables.
    ``annotation_key`` names the biological annotation used for strength
    summaries. The caller supplies the categorical stage order.
    """

    def __init__(self, samples, *, annotation_key):
        self.samples = pd.DataFrame(samples).copy()
        required = {"sample", "stage", "group", "path"}
        if not required.issubset(self.samples):
            raise ValueError(f"Sample table requires {sorted(required)}")
        if self.samples.empty or self.samples["sample"].duplicated().any():
            raise ValueError("Supply nonempty, uniquely named samples")
        if self.samples.duplicated(["group", "stage"]).any():
            raise ValueError("Each group/stage must be unique; summarize replicates separately")
        if "control" not in self.samples:
            self.samples["control"] = False
        self.samples["stage"] = self.samples["stage"].astype(str)
        self.annotation_key = annotation_key
        self.tables = {}

    def summarize(self, *, selected_pairs):
        """Compute counts, strengths, densities, hub fractions, CCDFs and distance fits.

        Cell-type strength sums incoming plus outgoing weights over all LR pairs,
        so each directed edge contributes twice overall. Density excludes self
        edges and uses n*(n-1). CCDF denominators include isolated observations.
        Cubic distance fits use all positive edges and stay within their observed
        distance range; fewer than four distinct distances are retained as points.
        Zero-total graphs have undefined hub fractions (NaN), not fabricated zeros.
        """
        groups = set(self.samples["group"])
        if set(selected_pairs) != groups:
            raise ValueError("selected_pairs must map every sample group to one LR pair")
        rows = {name: [] for name in ("samples", "lr", "celltypes", "hubs", "ccdf", "distance")}
        for record in self.samples.itertuples(index=False):
            a = load_result(record.path, leaf_only=True, embeddings=False)
            matrices = lr_matrices(a)
            identity = dict(sample=record.sample, stage=record.stage, group=record.group, control=record.control)
            rows["samples"].append(dict(identity, observations=a.n_obs))
            for pair, matrix in matrices.items():
                nonself = matrix.nnz - np.count_nonzero(matrix.diagonal())
                rows["lr"].append(dict(identity, lr_pair=pair, edge_count=matrix.nnz,
                    strength=float(matrix.sum()), density=nonself / (a.n_obs * (a.n_obs - 1)) if a.n_obs > 1 else np.nan))
            labels = a.obs[self.annotation_key]
            if labels.isna().any():
                raise ValueError(f"Missing {self.annotation_key} in {record.sample}")
            total = sum(matrices.values(), sparse.csr_matrix((a.n_obs, a.n_obs)))
            strength = np.asarray(total.sum(axis=0)).ravel() + np.asarray(total.sum(axis=1)).ravel()
            frame = pd.DataFrame({"cell_type": labels.astype(str).to_numpy(), "strength": strength})
            for label, values in frame.groupby("cell_type", sort=True):
                rows["celltypes"].append(dict(identity, cell_type=label, count=len(values),
                    strength=float(values.strength.sum()), mean_strength=float(values.strength.mean())))
            pair = selected_pairs[record.group]
            if pair not in matrices:
                raise KeyError(f"{record.sample} is missing selected LR {pair}")
            matrix = matrices[pair]
            out = np.asarray(matrix.sum(axis=1)).ravel()
            incoming = np.asarray(matrix.sum(axis=0)).ravel()
            incident = out + incoming
            for k in (5, 10, 15):
                fraction = np.sort(incident)[-k:].sum() / incident.sum() if incident.sum() else np.nan
                rows["hubs"].append(dict(identity, lr_pair=pair, k=k, fraction=fraction))
            vectors = {"out_degree": np.diff(matrix.indptr), "in_degree": np.diff(matrix.tocsc().indptr),
                       "out_strength": out, "in_strength": incoming}
            for metric, values in vectors.items():
                points, counts = np.unique(values[values > 0], return_counts=True)
                survival = np.cumsum(counts[::-1])[::-1] / a.n_obs
                rows["ccdf"].extend(dict(identity, lr_pair=pair, metric=metric, x=x, survival=y)
                                     for x, y in zip(points, survival))
            coo = matrix.tocoo()
            xy = np.asarray(a.obsm["spatial"])[:, :2]
            distances = np.linalg.norm(xy[coo.row] - xy[coo.col], axis=1)
            if len(np.unique(distances)) >= 4:
                fit = np.polynomial.Polynomial.fit(distances, coo.data, deg=3)
                xs = np.linspace(distances.min(), distances.max(), 200)
                ys, kind = fit(xs), "cubic fit"
            else:
                xs, ys, kind = distances, coo.data, "observed edge"
            rows["distance"].extend(dict(identity, lr_pair=pair, kind=kind, distance=x, score=y)
                                       for x, y in zip(xs, ys))
        columns = {
            "ccdf": ["sample", "stage", "group", "control", "lr_pair", "metric", "x", "survival"],
            "distance": ["sample", "stage", "group", "control", "lr_pair", "kind", "distance", "score"],
        }
        self.tables = {name: pd.DataFrame(values, columns=columns.get(name)) for name, values in rows.items()}
        return self.tables

    def export(self, directory):
        """Write tables by category below the supplied ``cellstic/time`` directory.

        Source tables share folders with their corresponding figures. Input
        filesystem paths are never included in the exported sample table.
        """
        directory = Path(directory)
        folders = {
            "samples": "samples",
            "lr": "lr_counts",
            "celltypes": "celltype_strength",
            "network_metrics": "network_metrics",
            "hubs": "topology",
            "ccdf": "topology",
            "distance": "topology",
        }
        for name, table in self.tables.items():
            target = directory / folders[name]
            target.mkdir(parents=True, exist_ok=True)
            table.to_csv(target / f"time_series_{name}.csv", index=False)

    def summarize_networks(self):
        """Recompute efficiency, path length and modularity for each whole sample.

        Sum all retained leaf-LR scores before making the undirected graph.
        The metric definitions are recorded with the results for reproducibility.
        No archived manuscript metric constants enter this calculation.
        """
        rows = []
        for record in self.samples.itertuples(index=False):
            result = load_result(record.path, leaf_only=True, embeddings=False)
            total = sum(lr_matrices(result).values(), sparse.csr_matrix((result.n_obs, result.n_obs)))
            rows.append(dict(sample=record.sample, stage=record.stage, group=record.group,
                             control=record.control, **network_metrics(total),
                             graph="sum of leaf LR scores; maximum reciprocal weight; no self edges",
                             path_definition="unweighted; mean over reachable distinct pairs",
                             efficiency_definition="unweighted; all distinct pairs including unreachable",
                             modularity_definition="weighted Louvain; resolution=1; seed=42"))
        self.tables["network_metrics"] = pd.DataFrame(rows)
        return self.tables["network_metrics"]
