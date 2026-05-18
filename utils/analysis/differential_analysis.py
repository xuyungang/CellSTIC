"""Differential interaction: domain subclustering, DEG heatmap; Louvain community detection on communication graph."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from anndata import AnnData
from scipy.sparse import csr_matrix

from utils.metrics import SpatialVisualizer, UMAPVisualizer, get_custom_palette
from utils.tools.seed_utils import active_base_seed
from utils.viz.matplotlib_svg import configure_matplotlib_svg_for_illustrator, path_wants_svg


def _set_nature_style(font_size: int = 9):
    """Set matplotlib rcParams for Nature-style figures."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
        'axes.unicode_minus': False, 'font.size': font_size,
        'axes.linewidth': 0.75, 'axes.edgecolor': '#000000',
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'savefig.edgecolor': 'none',
    })


def _get_spatial_coords(adata: AnnData) -> np.ndarray:
    """Extract spatial coordinates (n, 2)."""
    if 'spatial' in adata.obsm:
        return adata.obsm['spatial'][:, :2]
    if 'x' in adata.obs and 'y' in adata.obs:
        return np.ascontiguousarray(adata.obs[['x', 'y']].values, dtype=np.float32)
    raise ValueError("No spatial coordinates in adata")


def detect_communities_louvain(
    adata: AnnData,
    pos_edge_probs_np: np.ndarray,
    edge_type_map: Dict[str, int],
    annotation_key: str = "community",
    lr_filter: Optional[List[str]] = None,
    threshold: float = 0.0,
    resolution_range: Tuple[float, float] = (0.01, 10.0),
    target_n_communities: Optional[int] = None,
) -> Tuple[Any, Any, int]:
    """Louvain (CPM) on aggregated LR graph; write labels to adata.obs[annotation_key].

    If target_n_communities is set, resolution is searched in resolution_range (log-spaced)
    and the partition with n_communities closest to target is chosen; else use midpoint of resolution_range.
    """
    try:
        import igraph as ig  # type: ignore
        import louvain  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "detect_communities_louvain requires optional deps `igraph` and `louvain`.\n"
            "Install them first, e.g.: pip install python-igraph louvain"
        ) from e

    if pos_edge_probs_np.ndim != 3:
        raise ValueError("pos_edge_probs_np must be a 3D array [n_cells, n_cells, n_lr].")

    n1, n2, n_lr = pos_edge_probs_np.shape
    if n1 != n2:
        raise ValueError("pos_edge_probs_np must be square in the first two dimensions.")
    if n1 != adata.n_obs:
        raise ValueError(
            f"Shape mismatch: pos_edge_probs_np has {n1} nodes but adata has {adata.n_obs} observations."
        )

    for k, v in edge_type_map.items():
        if not (0 <= v < n_lr):
            raise ValueError(f"edge_type_map[{k!r}]={v} is out of range for n_lr={n_lr}.")

    if lr_filter is not None:
        valid_edge_indices = [edge_type_map[k] for k in lr_filter if k in edge_type_map]
        if len(valid_edge_indices) == 0:
            raise ValueError("None of the lr_filter entries were found in edge_type_map.")
        aggregated_graph = pos_edge_probs_np[:, :, valid_edge_indices].sum(axis=2)
    else:
        aggregated_graph = pos_edge_probs_np.sum(axis=2)

    aggregated_graph = np.asarray(aggregated_graph, dtype=float)

    if threshold > 0.0:
        aggregated_graph = aggregated_graph.copy()
        aggregated_graph[aggregated_graph < threshold] = 0.0

    np.fill_diagonal(aggregated_graph, 0.0)
    aggregated_graph = np.maximum(aggregated_graph, aggregated_graph.T)

    rows, cols = np.triu_indices_from(aggregated_graph, k=1)
    weights = aggregated_graph[rows, cols]
    mask = weights > 0
    rows, cols, weights = rows[mask], cols[mask], weights[mask]

    graph = ig.Graph(n=aggregated_graph.shape[0], edges=list(zip(rows, cols)), directed=False)
    if len(weights) == 0:
        partition = None
        community_labels = np.arange(aggregated_graph.shape[0], dtype=int)
        unique_communities = np.sort(np.unique(community_labels))
        adata.obs[annotation_key] = pd.Categorical(
            community_labels,
            categories=unique_communities,
            ordered=True,
        )
        n_communities = len(unique_communities)
        print(
            f"Community detection completed: {n_communities} communities detected "
            f"(no edges; each node its own community) and assigned to adata.obs['{annotation_key}']"
        )
        return graph, partition, n_communities

    graph.es["weight"] = weights.tolist()

    r_min, r_max = resolution_range[0], resolution_range[1]
    r_mid = (r_min * r_max) ** 0.5
    if target_n_communities is not None:
        n_trials = 20
        resolutions = np.logspace(np.log10(r_min), np.log10(r_max), num=n_trials).tolist()
        print(
            f"Performing Louvain community detection (resolution search in [{r_min:.4g}, {r_max:.4g}], "
            f"target_n_communities={target_n_communities})..."
        )
        best_partition = None
        best_score = None
        best_resolution = r_min
        print("  resolution -> n_communities (|target-n|)")
        for r in resolutions:
            part = louvain.find_partition(
                graph,
                louvain.CPMVertexPartition,
                resolution_parameter=float(r),
                weights="weight",
                seed=active_base_seed(),
            )
            n_com = len(np.unique(part.membership))
            score = -abs(n_com - target_n_communities)
            dist = abs(n_com - target_n_communities)
            mark = " *" if (best_score is None or score > best_score) else ""
            print(f"    {r:.4g} -> {n_com} (|target-n|={dist}){mark}")
            if best_score is None or score > best_score:
                best_score = score
                best_partition = part
                best_resolution = r
        partition = best_partition
        resolution = best_resolution
        n_selected = len(np.unique(partition.membership))
        print(f"  Selected resolution={resolution:.4g} -> {n_selected} communities")
        if n_selected > target_n_communities * 2 and resolution <= r_min * 1.1:
            print(
                f"  Hint: n_communities ({n_selected}) >> target ({target_n_communities}); "
                f"try smaller resolution_range (e.g. ({r_min/10:.2g}, {r_min:.2g})) to get fewer communities."
            )
    else:
        print(f"Performing Louvain community detection (resolution={r_mid:.4g}, midpoint of range)...")
        partition = louvain.find_partition(
            graph,
            louvain.CPMVertexPartition,
            resolution_parameter=r_mid,
            weights="weight",
            seed=active_base_seed(),
        )

    community_labels = np.array(partition.membership, dtype=int)
    unique_communities = np.sort(np.unique(community_labels))
    adata.obs[annotation_key] = pd.Categorical(
        community_labels,
        categories=unique_communities,
        ordered=True,
    )

    n_communities = len(unique_communities)
    print(
        f"Community detection completed: {n_communities} communities detected "
        f"and assigned to adata.obs['{annotation_key}']"
    )
    return graph, partition, n_communities

class DifferentialAnalyzer:
    """Domain subclustering (UMAP, spatial, DEG heatmap) and differential gene heatmap."""
    
    def plot_domain1_subclustering_analysis(
        self,
        adata: AnnData,
        pos_edge_probs_np: np.ndarray,
        edge_type_map: Dict[str, int],
        save_dir: Path,
        domain_key: str = 'annotation',
        domain1: str = '0',
        lr_pair: Tuple[str, str] = ('Penk', 'Oprk1'),
        threshold: float = 0.0,
        resolution_range: Tuple[float, float] = (0.01, 10.0),
        target_n_communities: Optional[int] = None,
        min_cells_per_cluster: int = 10,
    ) -> None:
        """Subcluster domain1 by Louvain; if target_n_communities set, search resolution in resolution_range. Only clusters with >= min_cells_per_cluster cells are kept for UMAP/spatial/DEG."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        domains = adata.obs[domain_key].astype(str).to_numpy()
        domain1_mask = domains == str(domain1)
        domain1_indices = np.where(domain1_mask)[0]

        if len(domain1_indices) == 0:
            print(f"Warning: Domain {domain1} not found, skipping subclustering analysis")
            return

        print(f"Found {len(domain1_indices)} cells in domain {domain1}")

        domain1_adata = adata[domain1_indices].copy()
        domain1_graph = pos_edge_probs_np[np.ix_(
            domain1_indices, domain1_indices,
            np.arange(pos_edge_probs_np.shape[2])
        )]

        if 'out' not in domain1_adata.obsm:
            raise ValueError("Features not found in adata.obsm['out']")
        features = domain1_adata.obsm['out']

        lr_key = f"{lr_pair[0]}:{lr_pair[1]}"
        if lr_key not in edge_type_map:
            lr_filter = None
            print(f"Warning: {lr_key} not in edge_type_map; using all LR pairs for community detection")
        else:
            lr_filter = [lr_key]

        graph, partition, n_communities = detect_communities_louvain(
            domain1_adata,
            domain1_graph,
            edge_type_map,
            annotation_key='cluster',
            lr_filter=lr_filter,
            threshold=threshold,
            resolution_range=resolution_range,
            target_n_communities=target_n_communities,
        )

        unique_clusters = sorted(domain1_adata.obs['cluster'].astype(str).unique())
        print(f"Clustering completed: {len(unique_clusters)} subclusters identified")

        cluster_counts = domain1_adata.obs['cluster'].astype(str).value_counts()
        valid_clusters = [c for c in unique_clusters if cluster_counts[c] >= min_cells_per_cluster]
        dropped = len(unique_clusters) - len(valid_clusters)
        if dropped > 0:
            print(f"Dropped {dropped} cluster(s) with < {min_cells_per_cluster} cells; {len(valid_clusters)} cluster(s) retained for analysis.")
        if len(valid_clusters) == 0:
            print(f"Warning: No cluster has >= {min_cells_per_cluster} cells, skipping subclustering analysis")
            return
        mask = domain1_adata.obs['cluster'].astype(str).isin(valid_clusters)
        domain1_adata = domain1_adata[mask].copy()
        domain1_adata.obs['cluster'] = pd.Categorical(
            domain1_adata.obs['cluster'].astype(str),
            categories=valid_clusters,
            ordered=True,
        )
        features = domain1_adata.obsm['out']
        unique_clusters = valid_clusters

        # Get color palette
        n_categories = len(unique_clusters)
        palette = get_custom_palette(n_categories)
        
        # (1) UMAP visualization colored by community (Louvain cluster), features from obsm['out']
        print("Generating UMAP visualization for subclusters...")
        umap_path = save_dir / "subcluster_umap.svg"
        UMAPVisualizer.generate_visualization(
            node_features=torch.tensor(features, dtype=torch.float32),
            adata_source=domain1_adata,
            save_path=str(umap_path),
            palette=palette,
            add_kde_contour=False,
            add_cluster_labels=True,
            point_size=100,
            color_key="cluster", 
        )
        
        # (2) Generate spatial distribution visualization
        print("Generating spatial distribution visualization for subclusters...")
        spatial_path = save_dir / "subcluster_spatial.svg"
        SpatialVisualizer.generate_spatial_domain_visualization(
            adata_source=domain1_adata,
            save_path=str(spatial_path),
            palette=palette,
            figsize=(6, 4.5),
        )
        
        # (3) Differential gene expression analysis and heatmap visualization
        if len(unique_clusters) >= 2:
            print("Performing differential gene expression analysis...")
            deg_path = save_dir / "subcluster_differential_genes_heatmap.svg"
            self._plot_differential_genes_heatmap(
                adata=domain1_adata,
                cluster_key='cluster',
                save_path=deg_path,
                n_top_genes=20,
                min_cells_per_cluster=min_cells_per_cluster,
            )
        else:
            print(f"Warning: Only {len(unique_clusters)} cluster(s) found, skipping differential gene analysis")
        
        print(f"Subclustering analysis completed. Results saved to {save_dir}")
    
        
    def _plot_differential_genes_heatmap(
        self,
        adata: AnnData,
        cluster_key: str = 'cluster',
        save_path: Path = None,
        n_top_genes: int = 30,
        min_cells_per_cluster: int = 10,
    ) -> None:
        """
        Publication-grade differential-gene heatmap with the same public API.

        Design goals
        ------------
        - Statistical rigor:
            * Wilcoxon rank-sum + BH-FDR + tie correction + fraction expressing
            * exact log2FC computed on linear normalized expression
            * detection-rate filters using pct_in / pct_out / pct_diff
        - Figure quality:
            * balanced marker panel across clusters
            * hierarchical ordering of clusters and genes
            * readable labels, gene-block separators, compact Nature-like layout
        - Reproducibility:
            * self-contained styling helpers
            * deterministic selection/order
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from pathlib import Path
        import scanpy as sc
        from scipy import sparse
        from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
        from scipy.spatial.distance import pdist
        from matplotlib.colors import LinearSegmentedColormap

        # ----------------------------
        # style / helper functions
        # ----------------------------
        def _apply_pub_style():
            mpl.rcParams.update({
                "font.family": "sans-serif",
                "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "font.size": 7,
                "axes.labelsize": 7,
                "axes.titlesize": 7,
                "xtick.labelsize": 6,
                "ytick.labelsize": 6,
                "axes.linewidth": 0.5,
                "xtick.major.width": 0.5,
                "ytick.major.width": 0.5,
                "xtick.major.size": 2.2,
                "ytick.major.size": 2.2,
                "savefig.transparent": False,
                "figure.facecolor": "white",
                "axes.facecolor": "white",
            })

        def _get_palette(n: int):
            base = [
                "#4E79A7", "#E15759", "#59A14F", "#F28E2B", "#B07AA1",
                "#76B7B2", "#EDC948", "#9C755F", "#FF9DA7", "#BAB0AC",
                "#1F77B4", "#D62728", "#2CA02C", "#9467BD", "#8C564B",
                "#17BECF", "#BCBD22", "#7F7F7F"
            ]
            if n <= len(base):
                return base[:n]
            cmap = plt.get_cmap("tab20")
            return [mpl.colors.to_hex(cmap(i / max(1, n - 1))) for i in range(n)]

        def _save_figure(fig, out_path: Path, dpi: int = 600, pad_inches: float = 0.02):
            out_path = Path(out_path)
            if out_path.suffix == "":
                out_path = out_path.with_suffix(".svg")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if path_wants_svg(out_path):
                configure_matplotlib_svg_for_illustrator()
            fig.savefig(
                out_path,
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=pad_inches,
                facecolor="white",
                edgecolor="none",
            )

        def _looks_like_counts_matrix(X) -> bool:
            try:
                if sparse.issparse(X):
                    data = X.data
                else:
                    data = np.asarray(X).ravel()
                if data.size == 0:
                    return False
                if np.nanmin(data) < 0:
                    return False
                sample = data[: min(20000, data.size)]
                frac_integer = np.mean(np.isclose(sample, np.round(sample), atol=1e-8))
                return (frac_integer > 0.95) and (np.nanmax(sample) > 20)
            except Exception:
                return False

        def _matrix_mean_and_pct(X):
            if sparse.issparse(X):
                mean = np.asarray(X.mean(axis=0)).ravel()
                pct = np.asarray(X.getnnz(axis=0)).ravel() / max(1, X.shape[0])
            else:
                X = np.asarray(X)
                mean = np.asarray(X.mean(axis=0)).ravel()
                pct = np.asarray((X > 0).mean(axis=0)).ravel()
            return mean, pct

        def _prepare_deg_input(adata_in: AnnData):
            """
            Returns
            -------
            adata_test : AnnData
                log1p-normalized matrix used for rank_genes_groups
            X_linear_norm :
                linear normalized expression used for exact log2FC / pct calculations
            source_note : str
                provenance note
            """
            if "counts" in adata_in.layers:
                adata_test = adata_in.copy()
                adata_test.X = adata_test.layers["counts"].copy()
                sc.pp.normalize_total(adata_test, target_sum=1e4)
                X_linear_norm = adata_test.X.copy()
                sc.pp.log1p(adata_test)
                return adata_test, X_linear_norm, "layers['counts'] -> normalize_total -> log1p"

            if adata_in.raw is not None:
                raw_adata = adata_in.raw.to_adata()
                raw_adata = raw_adata[adata_in.obs_names].copy()
                raw_adata.obs = adata_in.obs.copy()

                if _looks_like_counts_matrix(raw_adata.X):
                    adata_test = raw_adata.copy()
                    sc.pp.normalize_total(adata_test, target_sum=1e4)
                    X_linear_norm = adata_test.X.copy()
                    sc.pp.log1p(adata_test)
                    return adata_test, X_linear_norm, "adata.raw counts -> normalize_total -> log1p"

                adata_test = raw_adata.copy()
                if "log1p" in adata_test.uns:
                    if sparse.issparse(adata_test.X):
                        X_linear_norm = adata_test.X.copy()
                        X_linear_norm.data = np.expm1(X_linear_norm.data)
                    else:
                        X_linear_norm = np.expm1(np.asarray(adata_test.X))
                    return adata_test, X_linear_norm, "adata.raw already log1p-normalized"
                else:
                    sc.pp.normalize_total(adata_test, target_sum=1e4)
                    X_linear_norm = adata_test.X.copy()
                    sc.pp.log1p(adata_test)
                    return adata_test, X_linear_norm, "adata.raw -> normalize_total -> log1p"

            adata_test = adata_in.copy()
            if "log1p" in adata_test.uns:
                if sparse.issparse(adata_test.X):
                    X_linear_norm = adata_test.X.copy()
                    X_linear_norm.data = np.expm1(X_linear_norm.data)
                else:
                    X_linear_norm = np.expm1(np.asarray(adata_test.X))
                return adata_test, X_linear_norm, "adata.X already log1p-normalized"
            else:
                sc.pp.normalize_total(adata_test, target_sum=1e4)
                X_linear_norm = adata_test.X.copy()
                sc.pp.log1p(adata_test)
                return adata_test, X_linear_norm, "adata.X -> normalize_total -> log1p"

        def _balanced_gene_panel(cluster_tables, n_top_genes: int, max_total_genes: int):
            """Guarantee representation of each cluster before global filling."""
            picked = []
            used = set()

            quota = min(6, max(3, min(n_top_genes, max_total_genes) // max(2, len(cluster_tables))))
            quota = max(2, quota)

            for df in cluster_tables:
                if df is None or df.empty:
                    continue
                cluster_name = str(df["cluster"].iloc[0])
                count = 0
                for _, row in df.iterrows():
                    gene = row["names"]
                    if gene in used:
                        continue
                    picked.append({
                        "names": gene,
                        "source_cluster": cluster_name,
                        "marker_score": row["marker_score"],
                        "scores": row["scores"],
                        "pvals_adj": row["pvals_adj"],
                        "log2fc_exact": row["log2fc_exact"],
                        "pct_in": row["pct_in"],
                        "pct_out": row["pct_out"],
                        "pct_diff": row["pct_diff"],
                    })
                    used.add(gene)
                    count += 1
                    if count >= quota:
                        break

            if len(picked) < max_total_genes:
                pooled = pd.concat([df for df in cluster_tables if df is not None and not df.empty], ignore_index=True)
                pooled = pooled.sort_values(
                    ["marker_score", "log2fc_exact", "pct_diff", "pvals_adj"],
                    ascending=[False, False, False, True],
                )
                for _, row in pooled.iterrows():
                    gene = row["names"]
                    if gene in used:
                        continue
                    picked.append({
                        "names": gene,
                        "source_cluster": row["cluster"],
                        "marker_score": row["marker_score"],
                        "scores": row["scores"],
                        "pvals_adj": row["pvals_adj"],
                        "log2fc_exact": row["log2fc_exact"],
                        "pct_in": row["pct_in"],
                        "pct_out": row["pct_out"],
                        "pct_diff": row["pct_diff"],
                    })
                    used.add(gene)
                    if len(picked) >= max_total_genes:
                        break

            return pd.DataFrame(picked)

        def _zscore_rows(M):
            M = np.asarray(M, dtype=float)
            out = np.zeros_like(M, dtype=float)
            for j in range(M.shape[1]):
                col = M[:, j]
                mu = np.nanmean(col)
                sd = np.nanstd(col)
                out[:, j] = (col - mu) / (sd + 1e-10)
            return np.clip(out, -2.2, 2.2)

        # ----------------------------
        # sanity checks
        # ----------------------------
        if save_path is None:
            raise ValueError("save_path must not be None")

        if adata.X is None or adata.n_vars == 0:
            print("Warning: No gene expression data found, skipping differential gene analysis")
            return

        if cluster_key not in adata.obs:
            print(f"Warning: Cluster key '{cluster_key}' not found, skipping differential gene analysis")
            return

        clusters = adata.obs[cluster_key].astype(str)
        unique_clusters = sorted(clusters.unique())
        cluster_counts = clusters.value_counts()
        valid_clusters = [c for c in unique_clusters if cluster_counts[c] >= min_cells_per_cluster]

        if len(valid_clusters) < 2:
            print(f"Warning: Need at least 2 clusters with >= {min_cells_per_cluster} cells, skipping")
            return

        print(f"Analyzing differential genes for {len(valid_clusters)} clusters...")

        # ----------------------------
        # prepare working AnnData
        # ----------------------------
        adata_work = adata.copy()
        adata_work.obs["cluster"] = pd.Categorical(clusters, categories=valid_clusters)
        adata_work = adata_work[adata_work.obs["cluster"].isin(valid_clusters)].copy()

        adata_test, X_linear_norm, matrix_source_note = _prepare_deg_input(adata_work)
        adata_test.obs["cluster"] = adata_work.obs["cluster"].copy()

        # ----------------------------
        # differential ranking
        # ----------------------------
        rank_key = "rank_genes_groups_wilcoxon"
        sc.tl.rank_genes_groups(
            adata_test,
            groupby="cluster",
            groups=valid_clusters,
            reference="rest",
            method="wilcoxon",
            corr_method="benjamini-hochberg",
            tie_correct=True,
            pts=True,
            n_genes=adata_test.n_vars,
            key_added=rank_key,
            use_raw=False,
        )

        var_names = np.asarray(adata_test.var_names)
        cluster_tables = []
        full_cluster_dfs = {}  # cluster -> full DataFrame (names, log2fc_exact, pvals_adj) for heatmaps

        # ----------------------------
        # per-cluster marker filtering
        # ----------------------------
        for cluster in valid_clusters:
            mask_in = (adata_test.obs["cluster"].astype(str).to_numpy() == cluster)
            mask_out = ~mask_in

            X_in = X_linear_norm[mask_in]
            X_out = X_linear_norm[mask_out]

            mean_in, pct_in = _matrix_mean_and_pct(X_in)
            mean_out, pct_out = _matrix_mean_and_pct(X_out)
            pct_diff = pct_in - pct_out
            exact_log2fc = np.log2((mean_in + 1e-9) / (mean_out + 1e-9))

            effect_df = pd.DataFrame({
                "names": var_names,
                "mean_in": mean_in,
                "mean_out": mean_out,
                "pct_in": pct_in,
                "pct_out": pct_out,
                "pct_diff": pct_diff,
                "log2fc_exact": exact_log2fc,
            })

            try:
                cluster_df = sc.get.rank_genes_groups_df(
                    adata_test,
                    group=cluster,
                    key=rank_key,
                ).copy()
            except Exception as e:
                print(f"Warning: Failed to get DEGs for cluster {cluster}: {e}")
                continue

            cluster_df = cluster_df.merge(effect_df, on="names", how="left")
            cluster_df.insert(0, "cluster", cluster)
            full_cluster_dfs[cluster] = cluster_df.copy()

            strict = cluster_df[
                (cluster_df["pvals_adj"] < 0.05) &
                (cluster_df["log2fc_exact"] >= 0.50) &
                (cluster_df["pct_in"] >= 0.10) &
                (cluster_df["pct_diff"] >= 0.10)
            ].copy()

            if strict.shape[0] < min(6, n_top_genes):
                strict = cluster_df[
                    (cluster_df["pvals_adj"] < 0.05) &
                    (cluster_df["log2fc_exact"] >= 0.25) &
                    (cluster_df["pct_in"] >= 0.05) &
                    (cluster_df["pct_diff"] >= 0.05)
                ].copy()

            if strict.empty:
                print(f"Warning: No robust markers found for cluster {cluster}")
                continue

            strict["marker_score"] = (
                -np.log10(strict["pvals_adj"].clip(lower=1e-300)) +
                1.25 * np.clip(strict["log2fc_exact"], 0, None) +
                1.50 * np.clip(strict["pct_diff"], 0, None) +
                0.35 * strict["pct_in"]
            )

            strict = strict.sort_values(
                ["marker_score", "log2fc_exact", "pct_diff", "scores"],
                ascending=[False, False, False, False],
            )
            cluster_tables.append(strict)

        if len(cluster_tables) == 0:
            print("Warning: No differential genes found, skipping heatmap")
            return

        # ----------------------------
        # select final gene panel
        # ----------------------------
        max_total_genes = min(max(16, len(valid_clusters) * min(4, n_top_genes)), 40)
        selected_df = _balanced_gene_panel(cluster_tables, n_top_genes=n_top_genes, max_total_genes=max_total_genes)

        if selected_df.empty:
            print("Warning: No selected genes found after ranking, skipping heatmap")
            return

        all_deg_genes = selected_df["names"].tolist()

        # ----------------------------
        # export stats table
        # ----------------------------
        stats_out = pd.concat(cluster_tables, ignore_index=True)
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        selected_stats = stats_out[stats_out["names"].isin(all_deg_genes)].copy()
        selected_stats["selected_for_heatmap"] = selected_stats["names"].isin(all_deg_genes)
        selected_stats["matrix_source_note"] = matrix_source_note
        selected_stats.to_csv(save_path.parent / f"{save_path.stem}_selected_genes_stats.csv", index=False)

        # ----------------------------
        # build cluster-mean expression matrix
        # ----------------------------
        gene_idx = [adata_test.var_names.get_loc(g) for g in all_deg_genes if g in adata_test.var_names]
        if len(gene_idx) == 0:
            print("Warning: No selected genes found in adata, skipping heatmap")
            return

        gene_names = [adata_test.var_names[i] for i in gene_idx]
        cluster_n = {cl: int((adata_test.obs["cluster"] == cl).sum()) for cl in valid_clusters}

        expr_rows = []
        for cluster in valid_clusters:
            cluster_mask = (adata_test.obs["cluster"] == cluster).to_numpy()
            X_cluster = adata_test.X[cluster_mask][:, gene_idx]
            if sparse.issparse(X_cluster):
                X_cluster = X_cluster.toarray()
            expr_rows.append(np.asarray(X_cluster.mean(axis=0)).ravel())

        expr_matrix = np.vstack(expr_rows)
        expr_matrix_norm = _zscore_rows(expr_matrix)

        if len(valid_clusters) > 2:
            row_dist = pdist(expr_matrix_norm, metric="euclidean")
            row_linkage = linkage(row_dist, method="ward")
        else:
            row_linkage = None

        try:
            numeric_labels = [int(c) for c in valid_clusters]
            cluster_order = np.argsort(numeric_labels)
        except Exception:
            # Fallback: stable order as in valid_clusters
            cluster_order = np.arange(len(valid_clusters))

        cluster_labels_ordered = [valid_clusters[i] for i in cluster_order]
        expr_matrix_norm = expr_matrix_norm[cluster_order, :]

        # ----------------------------
        # gene order
        # 1) preserve source-cluster blocks
        # 2) cluster genes within each block by expression pattern
        # ----------------------------
        cluster_rank_map = {cl: i for i, cl in enumerate(cluster_labels_ordered)}
        selected_df = selected_df[selected_df["names"].isin(gene_names)].copy()
        selected_df["source_rank"] = selected_df["source_cluster"].map(cluster_rank_map)
        selected_df = selected_df.sort_values(
            ["source_rank", "marker_score", "log2fc_exact"],
            ascending=[True, False, False],
        )

        gene_order = []
        gene_labels_ordered = []
        gene_source_cluster = []
        boundaries = []

        for cl in cluster_labels_ordered:
            genes_in_block = selected_df.loc[selected_df["source_cluster"] == cl, "names"].tolist()
            genes_in_block = [g for g in genes_in_block if g in gene_names]
            if len(genes_in_block) == 0:
                continue

            idxs = [gene_names.index(g) for g in genes_in_block]
            block = expr_matrix_norm[:, idxs].T  # genes x clusters

            if len(idxs) > 2:
                col_dist = pdist(block, metric="correlation")
                if np.all(np.isfinite(col_dist)) and np.nanmax(col_dist) > 0:
                    col_linkage = linkage(col_dist, method="average")
                    inner_order = leaves_list(col_linkage)
                    idxs = [idxs[i] for i in inner_order]
                    genes_in_block = [genes_in_block[i] for i in inner_order]

            gene_order.extend(idxs)
            gene_labels_ordered.extend(genes_in_block)
            gene_source_cluster.extend([cl] * len(genes_in_block))
            boundaries.append(len(gene_order))

        expr_matrix_ordered = expr_matrix_norm[:, gene_order]
        n_clusters = len(cluster_labels_ordered)
        n_genes = len(gene_labels_ordered)

        # ----------------------------
        # log2FC and -log10(FDR) matrices (same cluster x gene order)
        # ----------------------------
        log2fc_matrix = np.full((n_clusters, n_genes), np.nan, dtype=float)
        neg_log10_fdr_matrix = np.full((n_clusters, n_genes), np.nan, dtype=float)
        for i, cluster in enumerate(cluster_labels_ordered):
            if cluster not in full_cluster_dfs:
                continue
            df = full_cluster_dfs[cluster]
            for j, gene in enumerate(gene_labels_ordered):
                row = df.loc[df["names"] == gene]
                if len(row) == 0:
                    continue
                log2fc_matrix[i, j] = row["log2fc_exact"].values[0]
                padj = np.clip(row["pvals_adj"].values[0], 1e-300, 1.0)
                neg_log10_fdr_matrix[i, j] = -np.log10(padj)

        # ----------------------------
        # plotting
        # ----------------------------
        _apply_pub_style()

        mm_to_in = 1.0 / 25.4
        single_col = 89 * mm_to_in
        double_col = 180 * mm_to_in

        fig_width = min(double_col, max(single_col, 0.18 * n_genes + 1.8))
        fig_height = max(2.2, 0.34 * n_clusters + 1.15)

        tick_fs = 6 if n_genes <= 22 else 5
        if n_genes <= 20:
            step_x = 1
        elif n_genes <= 34:
            step_x = 2
        else:
            step_x = 3

        xticks = np.arange(0, n_genes, step_x)
        xlabels = [gene_labels_ordered[i] for i in xticks]

        # Use the same cluster palette as UMAP / spatial subcluster plots
        cluster_colors = get_custom_palette(len(valid_clusters))
        cluster_color_map = {cl: cluster_colors[i] for i, cl in enumerate(valid_clusters)}

        cmap_expr = LinearSegmentedColormap.from_list(
            "nature_expr",
            ["#2C6DB2", "#F7F7F7", "#B33A3A"],
            N=256,
        )
        cmap_expr = cmap_expr.copy()
        cmap_expr.set_bad("#FFFFFF")

        cmap_log2fc = LinearSegmentedColormap.from_list(
            "log2fc", ["#2166AC", "#F7F7F7", "#B2182B"], N=256
        )
        cmap_log2fc = cmap_log2fc.copy()
        cmap_log2fc.set_bad("#FFFFFF")

        cmap_neglog10fdr = LinearSegmentedColormap.from_list(
            "neglog10fdr", ["#F7F7F7", "#EF8A62", "#B2182B"], N=256
        )
        cmap_neglog10fdr = cmap_neglog10fdr.copy()
        cmap_neglog10fdr.set_bad("#FFFFFF")

        def _draw_one_heatmap_figure(data, cmap, vmin, vmax, cbar_label, cbar_ticks=None):
            fig = plt.figure(figsize=(fig_width, fig_height), dpi=300, facecolor="white")
            gs = fig.add_gridspec(
                nrows=3,
                ncols=3,
                width_ratios=[0.07, 1.0, 0.07],
                height_ratios=[0.22, 0.06, 1.0],
                left=0.08,
                right=0.96,
                bottom=0.20 if n_genes > 18 else 0.14,
                top=0.96,
                hspace=0.04,
                wspace=0.04,
            )
            ax_d = fig.add_subplot(gs[0, 1])
            if row_linkage is not None:
                dendrogram(
                    row_linkage,
                    ax=ax_d,
                    orientation="top",
                    no_labels=True,
                    color_threshold=0,
                    above_threshold_color="#666666",
                    link_color_func=lambda k: "#666666",
                )
            ax_d.set_xticks([])
            ax_d.set_yticks([])
            for s in ax_d.spines.values():
                s.set_visible(False)

            ax_top = fig.add_subplot(gs[1, 1])
            ax_top.set_xlim(-0.5, n_genes - 0.5)
            ax_top.set_ylim(0, 1)
            ax_top.axis("off")
            for j, cl in enumerate(gene_source_cluster):
                ax_top.add_patch(
                    plt.Rectangle(
                        (j - 0.5, 0), 1, 1,
                        facecolor=cluster_color_map.get(cl, "#BBBBBB"),
                        edgecolor="none",
                    )
                )
            block_starts = [0] + boundaries[:-1]
            for start, end, cl in zip(block_starts, boundaries, cluster_labels_ordered):
                if end <= start:
                    continue
                center = (start + end - 1) / 2
                if end - start >= 2:
                    ax_top.text(center, 1.12, cl, ha="center", va="bottom", fontsize=5.5)
            for b in boundaries[:-1]:
                ax_top.axvline(b - 0.5, color="white", lw=1.2)

            ax_left = fig.add_subplot(gs[2, 0])
            ax_left.set_xlim(0, 1)
            ax_left.set_ylim(-0.5, n_clusters - 0.5)
            ax_left.invert_yaxis()
            ax_left.axis("off")
            for i, cl in enumerate(cluster_labels_ordered):
                ax_left.add_patch(
                    plt.Rectangle((0, i - 0.5), 1, 1, facecolor=cluster_color_map.get(cl, "#BBBBBB"), edgecolor="none")
                )

            ax = fig.add_subplot(gs[2, 1])
            im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
            for i in range(n_clusters + 1):
                ax.axhline(i - 0.5, color="white", lw=0.45, zorder=3)
            for b in boundaries[:-1]:
                ax.axvline(b - 0.5, color="white", lw=1.2, zorder=3)
                ax_top.axvline(b - 0.5, color="white", lw=1.2)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xlabels, rotation=60, ha="right", rotation_mode="anchor", fontsize=tick_fs)
            ax.set_yticks(np.arange(n_clusters))
            ax.set_yticklabels([f"{cl}  (n={cluster_n[cl]})" for cl in cluster_labels_ordered], fontsize=6)
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_linewidth(0.5)
            ax.tick_params(axis="both", length=2, width=0.5, pad=1.5)
            ax.set_xlabel("")
            ax.set_ylabel("")

            ax_cb = fig.add_subplot(gs[2, 2])
            cb = plt.colorbar(im, cax=ax_cb)
            cb.set_label(cbar_label, fontsize=6)
            if cbar_ticks is not None:
                cb.set_ticks(cbar_ticks)
            cb.ax.tick_params(labelsize=6, width=0.5, length=2)
            return fig

        save_path = Path(save_path)
        if save_path.suffix == "":
            save_path = save_path.with_suffix(".svg")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        stem, suffix = save_path.stem, save_path.suffix

        # Figure 1: expression z-score
        fig1 = _draw_one_heatmap_figure(
            expr_matrix_ordered, cmap_expr, -2.2, 2.2,
            "Cluster-mean\nlog1p expression\n(z-score)",
            cbar_ticks=[-2, -1, 0, 1, 2],
        )
        _save_figure(fig1, save_path, dpi=600, pad_inches=0.02)
        plt.close(fig1)
        print(f"Differential genes heatmap (expression) saved to {save_path}")

        # Figure 2: log2FC
        log2fc_abs_max = np.nanmax(np.abs(log2fc_matrix)) if np.any(np.isfinite(log2fc_matrix)) else 2.0
        log2fc_v = max(0.5, min(4.0, log2fc_abs_max))
        path_log2fc = save_path.parent / f"{stem}_log2fc{suffix}"
        fig2 = _draw_one_heatmap_figure(
            log2fc_matrix, cmap_log2fc, -log2fc_v, log2fc_v,
            "log₂(FC)",
            cbar_ticks=None,
        )
        _save_figure(fig2, path_log2fc, dpi=600, pad_inches=0.02)
        plt.close(fig2)
        print(f"Differential genes heatmap (log2FC) saved to {path_log2fc}")

        # Figure 3: -log10(FDR)
        neglog10fdr_max = np.nanmax(neg_log10_fdr_matrix) if np.any(np.isfinite(neg_log10_fdr_matrix)) else 10.0
        neglog10fdr_v = min(15.0, max(2.0, neglog10fdr_max))
        path_neglog10fdr = save_path.parent / f"{stem}_neglog10fdr{suffix}"
        fig3 = _draw_one_heatmap_figure(
            neg_log10_fdr_matrix, cmap_neglog10fdr, 0, neglog10fdr_v,
            "-log₁₀(FDR)",
            cbar_ticks=None,
        )
        _save_figure(fig3, path_neglog10fdr, dpi=600, pad_inches=0.02)
        plt.close(fig3)
        print(f"Differential genes heatmap (-log10(FDR)) saved to {path_neglog10fdr}")