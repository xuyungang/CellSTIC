"""Ligand–receptor spatial communication maps."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize, to_hex, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch, Polygon, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.ticker import FormatStrFormatter
from anndata import AnnData

from utils.viz.matplotlib_svg import configure_matplotlib_svg_for_illustrator, path_wants_svg

from utils.tools.seed_utils import active_base_seed

try:
    from scipy.spatial import Voronoi, QhullError, cKDTree
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    Voronoi = None
    QhullError = Exception
    cKDTree = None
    _HAS_SCIPY = False


def get_colour_scheme(palette_name: str, num_colours: int) -> List[str]:
    """
    Return a list of `num_colours` hex colors from a matplotlib colormap.
    """
    cmap = plt.get_cmap(palette_name)
    xs = np.linspace(0.0, 1.0, num_colours, endpoint=True)
    return [to_hex(cmap(x), keep_alpha=False) for x in xs]


class LigandReceptorSpatialVisualizer:
    """High-level API for ligand–receptor spatial communication maps."""

    def plot_ligand_receptor_spatial_distribution(
        self,
        adata: AnnData,
        ligand_receptor_pairs: Optional[List[Tuple[str, str]]] = None,
        edge_type_map: Optional[Dict[str, int]] = None,
        pos_edge_probs: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
        threshold: float = 0.0,
        lr_filter: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        max_outgoing_edges_per_node: int = 3,
        max_incoming_edges_per_node: int = 3,
        show_intensity_axis: bool = True,
    ) -> None:
        """
        Plot per–ligand–receptor-pair spatial communication maps.

        Style target:
        - one irregular tissue-like polygon per point;
        - thicker black curved communication lines;
        - red polygon fill according to local hotspot score;
        - no titles or LR text labels in the panel.

        Edge sparsification:
        - per LR pair, we keep only the strongest edges so that, in the
          undirected sense, each node is incident to at most ~3 edges
          by default (symmetric cap derived from the outgoing/incoming
          limits).
        """
        if pos_edge_probs is not None:
            try:
                pos_edge_probs = (
                    pos_edge_probs.detach().cpu().numpy()
                    if hasattr(pos_edge_probs, "detach")
                    else np.asarray(pos_edge_probs)
                )
            except Exception as e:
                print(f"Warning: Could not convert pos_edge_probs to numpy: {e}")
                pos_edge_probs = None

        coords = self._extract_spatial_coordinates(adata)
        gene_to_idx = {gene: idx for idx, gene in enumerate(adata.var_names)}
        lr_pairs = self._get_ligand_receptor_pairs(
            ligand_receptor_pairs, edge_type_map, gene_to_idx
        )

        if lr_filter is not None:
            def normalize_lr_pair(lig: str, rec: str) -> Tuple[str, str]:
                lig_norm = lig.strip()
                rec_components = sorted([r.strip() for r in rec.split(":") if r.strip()])
                return lig_norm, ":".join(rec_components)

            normalized_lr_filter = set()
            for key in lr_filter:
                if ":" in key:
                    lig, rec = key.split(":", 1)
                else:
                    lig, rec = key.strip(), key.strip()
                normalized_lr_filter.add(normalize_lr_pair(lig, rec))

            lr_pairs = [
                lr for lr in lr_pairs
                if normalize_lr_pair(lr[0], lr[1]) in normalized_lr_filter
            ]

        if not lr_pairs:
            raise ValueError("No valid ligand-receptor pairs found after filtering")
        if pos_edge_probs is None or edge_type_map is None:
            raise ValueError("pos_edge_probs and edge_type_map are required for visualization")

        if save_path is None:
            save_dir = Path(".")
        else:
            save_path_obj = Path(save_path)
            if save_path_obj.suffix.lower() in {".png", ".jpg", ".jpeg", ".tiff", ".pdf", ".svg"}:
                save_dir = save_path_obj.parent
            else:
                save_dir = save_path_obj
        save_dir.mkdir(parents=True, exist_ok=True)

        lr_to_edge_idx = self._create_lr_to_edge_idx(edge_type_map, lr_pairs)
        n_points = coords.shape[0]

        for lig, rec in lr_pairs:
            if (lig, rec) not in lr_to_edge_idx:
                continue

            edge_idx = lr_to_edge_idx[(lig, rec)]
            edge_probs = pos_edge_probs[:, :, edge_idx]
            if edge_probs.shape[0] != n_points or edge_probs.shape[1] != n_points:
                continue

            edges_list: List[Tuple[int, int]] = []
            scores_list: List[float] = []
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    s_ij = float(edge_probs[i, j])
                    s_ji = float(edge_probs[j, i])
                    s = max(s_ij, s_ji)
                    if s > threshold:
                        edges_list.append((i, j))
                        scores_list.append(s)

            if not edges_list:
                continue

            edges_arr = np.asarray(edges_list, dtype=int)
            scores_arr = np.asarray(scores_list, dtype=float)

            # Per-node degree cap: greedily keep highest-scoring edges
            # such that each node participates in at most K edges
            # (K derived from the outgoing/incoming limits).
            max_deg = max(1, int(min(max_outgoing_edges_per_node, max_incoming_edges_per_node)))
            if edges_arr.size > 0 and max_deg < np.iinfo(np.int32).max:
                edges_arr, scores_arr = self._limit_edges_per_node(
                    n_nodes=n_points,
                    edges=edges_arr,
                    edge_score=scores_arr,
                    max_degree=max_deg,
                )

            vmin_used = float(threshold)
            vmax_used = (
                float(np.percentile(scores_arr, 95))
                if scores_arr.size > 0 else float(threshold + 1.0)
            )
            if vmax_used <= vmin_used:
                vmax_used = vmin_used + 1.0

            safe_name = f"{lig}_{rec}".replace(" ", "_").replace("/", "-").replace(":", "-")
            out_path = save_dir / f"lr_spatial_{safe_name}.svg"

            self._plot_spatial_comm_map(
                coords=coords,
                edges=edges_arr,
                edge_score=scores_arr,
                figsize=figsize or (7.0, 6.6),
                bg_color="white",
                cmap="Reds",
                cmap_min=0.08,   # lower bound is light red, not pure white
                vmin=vmin_used,
                vmax=vmax_used,
                invert_y=False,
                save=str(out_path),
                dpi=300,
                show_intensity_axis=show_intensity_axis,
            )

    def plot_ligand_receptor_spatial_distribution_by_region(
        self,
        adata: AnnData,
        ligand_receptor_pairs: Optional[List[Tuple[str, str]]] = None,
        edge_type_map: Optional[Dict[str, int]] = None,
        pos_edge_probs: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
        threshold: float = 0.0,
        lr_filter: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        max_outgoing_edges_per_node: int = 3,
        max_incoming_edges_per_node: int = 3,
        region_key: str = "domain",
        regions_to_plot: Optional[List[str]] = None,
    ) -> None:
        """
        Plot per–LR-pair spatial communication maps with cells colored by region.

        Same as plot_ligand_receptor_spatial_distribution (edges, layout, curved lines),
        but cell polygons are colored by region (one color per region) instead of by
        hotspot score. Optionally restrict to a subset of regions via regions_to_plot.

        Expects adata.obs[region_key] to contain region labels (e.g. from load_domain_from_csv).
        """
        if pos_edge_probs is not None:
            try:
                pos_edge_probs = (
                    pos_edge_probs.detach().cpu().numpy()
                    if hasattr(pos_edge_probs, "detach")
                    else np.asarray(pos_edge_probs)
                )
            except Exception as e:
                print(f"Warning: Could not convert pos_edge_probs to numpy: {e}")
                pos_edge_probs = None

        if region_key not in adata.obs:
            raise ValueError(
                f"region_key '{region_key}' not found in adata.obs; "
                "load domain/region first (e.g. via load_domain_from_csv) or set region_key."
            )

        region_labels = adata.obs[region_key].astype(str).to_numpy()

        if regions_to_plot is not None:
            regions_set = set(regions_to_plot)
            mask = np.array([r in regions_set for r in region_labels])
            if not np.any(mask):
                raise ValueError(
                    f"No cells found in regions_to_plot {regions_to_plot}; "
                    f"available: {np.unique(region_labels).tolist()}"
                )
            adata = adata[mask].copy()
            region_labels = region_labels[mask]
            if pos_edge_probs is not None:
                pos_edge_probs = pos_edge_probs[mask][:, mask, :]

        coords = self._extract_spatial_coordinates(adata)
        gene_to_idx = {gene: idx for idx, gene in enumerate(adata.var_names)}
        lr_pairs = self._get_ligand_receptor_pairs(
            ligand_receptor_pairs, edge_type_map, gene_to_idx
        )

        if lr_filter is not None:
            def normalize_lr_pair(lig: str, rec: str) -> Tuple[str, str]:
                lig_norm = lig.strip()
                rec_components = sorted([r.strip() for r in rec.split(":") if r.strip()])
                return lig_norm, ":".join(rec_components)

            normalized_lr_filter = set()
            for key in lr_filter:
                if ":" in key:
                    lig, rec = key.split(":", 1)
                else:
                    lig, rec = key.strip(), key.strip()
                normalized_lr_filter.add(normalize_lr_pair(lig, rec))

            lr_pairs = [
                lr for lr in lr_pairs
                if normalize_lr_pair(lr[0], lr[1]) in normalized_lr_filter
            ]

        if not lr_pairs:
            raise ValueError("No valid ligand-receptor pairs found after filtering")
        if pos_edge_probs is None or edge_type_map is None:
            raise ValueError("pos_edge_probs and edge_type_map are required for visualization")

        if save_path is None:
            save_dir = Path(".")
        else:
            save_path_obj = Path(save_path)
            if save_path_obj.suffix.lower() in {".png", ".jpg", ".jpeg", ".tiff", ".pdf", ".svg"}:
                save_dir = save_path_obj.parent
            else:
                save_dir = save_path_obj
        save_dir.mkdir(parents=True, exist_ok=True)

        lr_to_edge_idx = self._create_lr_to_edge_idx(edge_type_map, lr_pairs)
        n_points = coords.shape[0]

        for lig, rec in lr_pairs:
            if (lig, rec) not in lr_to_edge_idx:
                continue

            edge_idx = lr_to_edge_idx[(lig, rec)]
            edge_probs = pos_edge_probs[:, :, edge_idx]
            if edge_probs.shape[0] != n_points or edge_probs.shape[1] != n_points:
                continue

            edges_list = []
            scores_list = []
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    s_ij = float(edge_probs[i, j])
                    s_ji = float(edge_probs[j, i])
                    s = max(s_ij, s_ji)
                    if s > threshold:
                        edges_list.append((i, j))
                        scores_list.append(s)

            if not edges_list:
                continue

            edges_arr = np.asarray(edges_list, dtype=int)
            scores_arr = np.asarray(scores_list, dtype=float)

            max_deg = max(1, int(min(max_outgoing_edges_per_node, max_incoming_edges_per_node)))
            if edges_arr.size > 0 and max_deg < np.iinfo(np.int32).max:
                edges_arr, scores_arr = self._limit_edges_per_node(
                    n_nodes=n_points,
                    edges=edges_arr,
                    edge_score=scores_arr,
                    max_degree=max_deg,
                )

            vmin_used = float(threshold)
            vmax_used = (
                float(np.percentile(scores_arr, 95))
                if scores_arr.size > 0 else float(threshold + 1.0)
            )
            if vmax_used <= vmin_used:
                vmax_used = vmin_used + 1.0

            safe_name = f"{lig}_{rec}".replace(" ", "_").replace("/", "-").replace(":", "-")
            out_path = save_dir / f"lr_spatial_region_{safe_name}.svg"

            self._plot_spatial_comm_map_by_region(
                coords=coords,
                edges=edges_arr,
                edge_score=scores_arr,
                region_labels=region_labels,
                figsize=figsize or (7.0, 6.6),
                bg_color="white",
                vmin=vmin_used,
                vmax=vmax_used,
                invert_y=False,
                save=str(out_path),
                dpi=300,
            )

    def _extract_spatial_coordinates(self, adata: AnnData) -> np.ndarray:
        """Extract spatial coordinates."""
        if "spatial" in adata.obsm:
            return np.asarray(adata.obsm["spatial"][:, :2], dtype=float)
        if "x" in adata.obs and "y" in adata.obs:
            return np.ascontiguousarray(adata.obs[["x", "y"]].values, dtype=np.float32)
        raise ValueError("No spatial coordinates found")

    def _get_ligand_receptor_pairs(
        self,
        ligand_receptor_pairs: Optional[List[Tuple[str, str]]],
        edge_type_map: Optional[Dict[str, int]],
        gene_to_idx: Dict[str, int],
    ) -> List[Tuple[str, str]]:
        """Get and validate ligand-receptor pair list."""
        if ligand_receptor_pairs is not None:
            lr_pairs = ligand_receptor_pairs
        elif edge_type_map is not None:
            lr_pairs = []
            for lr_name, _ in sorted(edge_type_map.items(), key=lambda x: x[1]):
                sep = ":" if ":" in lr_name else "-" if "-" in lr_name else None
                if sep:
                    lr_pairs.append(tuple(s.strip() for s in lr_name.split(sep, 1)))
        else:
            lr_pairs = []

        valid_lr_pairs = []
        for ligand, receptor in lr_pairs:
            if ligand not in gene_to_idx:
                continue
            receptor_names = [r.strip() for r in receptor.split(":") if r.strip()]
            if receptor_names and all(r in gene_to_idx for r in receptor_names):
                valid_lr_pairs.append((ligand, receptor))

        print(f"Found {len(valid_lr_pairs)} valid ligand-receptor pairs")
        return valid_lr_pairs

    def _create_lr_to_edge_idx(
        self,
        edge_type_map: Dict[str, int],
        lr_pairs: List[Tuple[str, str]],
    ) -> Dict[Tuple[str, str], int]:
        """Create mapping from LR pair to edge type index."""
        lr_to_edge_idx = {}
        for lr_name, idx in edge_type_map.items():
            sep = ":" if ":" in lr_name else "-" if "-" in lr_name else None
            if sep is None:
                continue
            l, r = lr_name.split(sep, 1)
            lr_pair = (l.strip(), r.strip())
            if lr_pair in lr_pairs:
                lr_to_edge_idx[lr_pair] = idx
        return lr_to_edge_idx

    def _compute_node_score(
        self,
        n_nodes: int,
        edges: np.ndarray,
        edge_score: np.ndarray,
    ) -> np.ndarray:
        """Aggregate edge scores to node-level hotspot score using max incident edge score."""
        node_score = np.zeros(n_nodes, dtype=float)
        for (i, j), s in zip(edges, edge_score):
            if s > node_score[i]:
                node_score[i] = s
            if s > node_score[j]:
                node_score[j] = s
        return node_score

    def _limit_edges_per_node(
        self,
        n_nodes: int,
        edges: np.ndarray,
        edge_score: np.ndarray,
        max_degree: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Greedily keep highest-scoring undirected edges with a per-node degree cap.

        The algorithm sorts edges by descending score and accepts an edge (i, j)
        only if both nodes currently have degree < max_degree.
        """
        if edges.size == 0 or max_degree <= 0:
            return edges[:0], edge_score[:0]

        order = np.argsort(edge_score)[::-1]
        edges_sorted = edges[order]
        scores_sorted = edge_score[order]

        degree = np.zeros(n_nodes, dtype=int)
        kept_edges: List[Tuple[int, int]] = []
        kept_scores: List[float] = []

        for (i, j), s in zip(edges_sorted, scores_sorted):
            if degree[i] >= max_degree or degree[j] >= max_degree:
                continue
            kept_edges.append((int(i), int(j)))
            kept_scores.append(float(s))
            degree[i] += 1
            degree[j] += 1

        if not kept_edges:
            return edges[:0], edge_score[:0]

        return np.asarray(kept_edges, dtype=int), np.asarray(kept_scores, dtype=float)

    def _voronoi_finite_polygons_2d(
        self,
        vor: "Voronoi",
        radius: Optional[float] = None,
    ) -> Tuple[List[List[int]], np.ndarray]:
        """
        Reconstruct infinite Voronoi regions in a 2D diagram to finite regions.
        """
        if vor.points.shape[1] != 2:
            raise ValueError("Requires 2D input")

        new_regions: List[List[int]] = []
        new_vertices = vor.vertices.tolist()

        center = vor.points.mean(axis=0)
        if radius is None:
            radius = float(np.ptp(vor.points, axis=0).max() * 2.0)

        all_ridges: Dict[int, List[Tuple[int, int, int]]] = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        for p1, region_idx in enumerate(vor.point_region):
            region = vor.regions[region_idx]

            if len(region) == 0:
                new_regions.append([])
                continue

            if all(v >= 0 for v in region):
                new_regions.append(region)
                continue

            ridges = all_ridges.get(p1, [])
            new_region = [v for v in region if v >= 0]

            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    continue

                tangent = vor.points[p2] - vor.points[p1]
                tangent /= np.linalg.norm(tangent) + 1e-12
                normal = np.array([-tangent[1], tangent[0]])

                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, normal)) * normal
                far_point = vor.vertices[v2] + direction * radius

                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())

            if len(new_region) == 0:
                new_regions.append([])
                continue

            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
            new_region = [v for _, v in sorted(zip(angles, new_region))]
            new_regions.append(new_region)

        return new_regions, np.asarray(new_vertices)

    def _chaikin_smooth_closed(
        self,
        points: np.ndarray,
        refinements: int = 2,
    ) -> np.ndarray:
        """Chaikin smoothing for closed polygons."""
        pts = np.asarray(points, dtype=float)
        if pts.shape[0] < 3:
            return pts

        for _ in range(refinements):
            new_pts = []
            n = pts.shape[0]
            for i in range(n):
                p0 = pts[i]
                p1 = pts[(i + 1) % n]
                q = 0.75 * p0 + 0.25 * p1
                r = 0.25 * p0 + 0.75 * p1
                new_pts.extend([q, r])
            pts = np.asarray(new_pts, dtype=float)
        return pts

    def _make_one_irregular_cell(
        self,
        seed_point: np.ndarray,
        region_polygon: np.ndarray,
        local_radius: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Convert a Voronoi polygon into a more irregular cell-like polygon.

        Compared with a smoother version:
        - stronger anisotropy;
        - more edge perturbation points;
        - less smoothing to preserve irregularity;
        - extra radial harmonic distortion.
        """
        poly = np.asarray(region_polygon, dtype=float)
        if poly.shape[0] < 3:
            return poly

        seed_point = np.asarray(seed_point, dtype=float)

        # 1) clamp region by local spacing
        vec = poly - seed_point
        dist = np.linalg.norm(vec, axis=1)
        max_r = max(local_radius * 1.30, 1e-6)
        scale = np.minimum(1.0, max_r / (dist + 1e-12))
        poly = seed_point + vec * scale[:, None]

        # 2) global shrink: leave a little gap, but not too much
        shrink = 0.84 + 0.08 * rng.random()
        poly = seed_point + shrink * (poly - seed_point)

        # 3) anisotropic deformation to make shapes less uniform
        theta0 = rng.uniform(0, 2 * np.pi)
        major_axis = np.array([np.cos(theta0), np.sin(theta0)])
        minor_axis = np.array([-major_axis[1], major_axis[0]])
        major_scale = 1.02 + 0.18 * rng.random()
        minor_scale = 0.82 + 0.12 * rng.random()

        vv = poly - seed_point
        proj_major = vv @ major_axis
        proj_minor = vv @ minor_axis
        poly = (
            seed_point
            + np.outer(proj_major * major_scale, major_axis)
            + np.outer(proj_minor * minor_scale, minor_axis)
        )

        # 4) insert more irregular points along edges
        pts = []
        n = poly.shape[0]
        for i in range(n):
            p0 = poly[i]
            p1 = poly[(i + 1) % n]
            pts.append(p0)

            edge = p1 - p0
            L = np.linalg.norm(edge) + 1e-12
            tangent = edge / L
            perp = np.array([-tangent[1], tangent[0]])

            # 2 or 3 inner points per edge
            n_inner = 2 + int(rng.random() > 0.45)

            for k in range(1, n_inner + 1):
                t = k / (n_inner + 1)
                base = (1 - t) * p0 + t * p1

                radial = base - seed_point
                radial /= np.linalg.norm(radial) + 1e-12

                amp = local_radius * (0.07 + 0.18 * rng.random())

                side_sign = 1.0 if rng.random() > 0.5 else -1.0
                direction = (
                    (0.75 + 0.20 * rng.random()) * radial
                    + side_sign * (0.18 + 0.35 * rng.random()) * perp
                )
                direction /= np.linalg.norm(direction) + 1e-12

                pts.append(base + amp * direction)

        pts = np.asarray(pts, dtype=float)

        # 5) lighter smoothing: preserve irregular boundaries
        pts = self._chaikin_smooth_closed(pts, refinements=1)

        # 6) extra harmonic radial distortion
        vec = pts - seed_point
        ang = np.arctan2(vec[:, 1], vec[:, 0])
        rr = np.linalg.norm(vec, axis=1)

        f1 = int(rng.integers(3, 6))
        f2 = int(rng.integers(6, 10))
        ph1 = rng.uniform(0, 2 * np.pi)
        ph2 = rng.uniform(0, 2 * np.pi)

        mod = (
            1.0
            + 0.10 * np.sin(f1 * ang + ph1)
            + 0.06 * np.sin(f2 * ang + ph2)
        )
        rr = rr * np.clip(mod, 0.82, 1.22)

        max_final_r = local_radius * 1.36
        rr = np.minimum(rr, max_final_r)

        pts = np.c_[
            seed_point[0] + rr * np.cos(ang),
            seed_point[1] + rr * np.sin(ang),
        ]
        return pts

    def _build_irregular_tissue_patches(
        self,
        coords: np.ndarray,
    ) -> Optional[List[Polygon]]:
        """
        Build one different irregular polygon per point.

        This version makes the blocks more irregular and less uniform.
        """
        if not _HAS_SCIPY or coords.shape[0] < 4:
            return None

        pts = np.asarray(coords, dtype=float).copy()

        x_range = float(pts[:, 0].max() - pts[:, 0].min())
        y_range = float(pts[:, 1].max() - pts[:, 1].min())
        jitter_scale = max(x_range, y_range, 1.0) * 1e-8
        rng0 = np.random.default_rng(active_base_seed())
        pts += rng0.normal(loc=0.0, scale=jitter_scale, size=pts.shape)

        try:
            vor = Voronoi(pts)
            regions, vertices = self._voronoi_finite_polygons_2d(vor)
        except QhullError:
            return None
        except Exception:
            return None

        try:
            tree = cKDTree(coords)
            k = min(5, len(coords))
            dists, _ = tree.query(coords, k=k)
            if dists.ndim == 1:
                local_radius = np.full(len(coords), np.median(dists) * 0.52)
            else:
                neigh = dists[:, 1:] if dists.shape[1] > 1 else dists
                local_radius = np.median(neigh, axis=1) * 0.52
        except Exception:
            local_radius = np.full(
                len(coords),
                max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1])) / 140.0
            )

        if len(local_radius) > 1:
            lo = np.percentile(local_radius, 5)
            hi = np.percentile(local_radius, 95)
            local_radius = np.clip(local_radius, lo, hi)

        patches: List[Polygon] = []
        for idx, region in enumerate(regions):
            rng = np.random.default_rng(active_base_seed() + 1000 + idx)

            if len(region) < 3:
                theta = np.linspace(0, 2 * np.pi, 18, endpoint=False)
                rr = local_radius[idx] * (
                    0.72
                    + 0.18 * np.sin(theta * (3 + (idx % 3)) + idx * 0.31)
                    + 0.08 * np.sin(theta * 7 + idx * 0.17)
                )
                poly = np.c_[
                    coords[idx, 0] + rr * np.cos(theta),
                    coords[idx, 1] + rr * np.sin(theta),
                ]
                patches.append(Polygon(poly, closed=True))
                continue

            region_polygon = vertices[region]
            irr_poly = self._make_one_irregular_cell(
                seed_point=coords[idx],
                region_polygon=region_polygon,
                local_radius=float(local_radius[idx]),
                rng=rng,
            )
            patches.append(Polygon(irr_poly, closed=True))

        return patches

    def _truncate_cmap(
        self,
        cmap: Union[str, LinearSegmentedColormap],
        minval: float = 0.08,
        maxval: float = 1.0,
        n: int = 256,
    ) -> LinearSegmentedColormap:
        """
        Truncate a colormap so that the lower end is not pure white.

        For example, cmap='Reds', minval=0.08 means the displayed minimum
        starts from a very light pink rather than white.
        """
        base = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
        minval = float(np.clip(minval, 0.0, 0.95))
        maxval = float(np.clip(maxval, minval + 1e-6, 1.0))
        colors = base(np.linspace(minval, maxval, n))
        return LinearSegmentedColormap.from_list(
            f"trunc_{base.name}_{minval:.2f}_{maxval:.2f}",
            colors,
        )

    def _style_nature_colorbar(
        self,
        cbar,
        *,
        vmin: float,
        vmax: float,
    ) -> None:
        """
        Style colorbar in a cleaner Nature-like manner:
        - slim bar
        - no outer box
        - few ticks
        - small font
        """
        cbar.outline.set_visible(False)
        for spine in cbar.ax.spines.values():
            spine.set_visible(False)

        if vmax <= vmin:
            ticks = np.array([vmin], dtype=float)
        else:
            ticks = np.linspace(vmin, vmax, 4)

        cbar.set_ticks(ticks)

        value_range = float(vmax - vmin)
        if value_range < 0.1:
            fmt = "%.3f"
        elif value_range < 1:
            fmt = "%.2f"
        elif value_range < 10:
            fmt = "%.1f"
        else:
            fmt = "%.0f"

        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))
        cbar.ax.tick_params(
            axis="y",
            which="major",
            direction="out",
            length=2.2,
            width=0.6,
            color="#4d4d4d",
            labelcolor="#4d4d4d",
            labelsize=8,
            pad=2,
        )

    def _draw_curved_edges(
        self,
        ax,
        coords: np.ndarray,
        edges: np.ndarray,
        edge_score: np.ndarray,
        *,
        vmin: float,
        vmax: float,
        alpha_min: float = 0.35,
        alpha_max: float = 0.95,
        lw_min: float = 0.22,
        lw_max: float = 1.40,
        curve_scale: float = 0.18,
        zorder: float = 4.0,
    ) -> None:
        """
        Draw thicker black curved edges with quadratic Bézier curves.
        """
        if edges.size == 0:
            return

        order = np.argsort(edge_score)
        edges = edges[order]
        edge_score = edge_score[order]

        seg = coords[edges[:, 1]] - coords[edges[:, 0]]
        seg_len = np.linalg.norm(seg, axis=1)
        positive = seg_len > 1e-12
        median_len = float(np.median(seg_len[positive])) if np.any(positive) else 1.0
        median_len = max(median_len, 1e-6)

        for (i, j), s in zip(edges, edge_score):
            p0 = coords[i]
            p1 = coords[j]
            d = p1 - p0
            L = np.linalg.norm(d)
            if L <= 1e-12:
                continue

            tangent = d / L
            perp = np.array([-tangent[1], tangent[0]])

            sign = 1.0 if (((int(i) * 73856093) ^ (int(j) * 19349663)) & 1) else -1.0

            strength = (np.clip(s, vmin, vmax) - vmin) / max(vmax - vmin, 1e-12)
            len_factor = np.clip(L / median_len, 0.55, 1.55)
            offset = sign * perp * (curve_scale * len_factor * L)

            mid = 0.5 * (p0 + p1)
            control = mid + offset

            lw = lw_min + strength * (lw_max - lw_min)
            alpha = alpha_min + strength * (alpha_max - alpha_min)

            path = MplPath(
                [p0, control, p1],
                [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3],
            )

            under = PathPatch(
                path,
                facecolor="none",
                edgecolor="black",
                lw=lw * 1.35,
                alpha=alpha * 0.22,
                capstyle="round",
                joinstyle="round",
                zorder=zorder - 0.05,
            )
            ax.add_patch(under)

            patch = PathPatch(
                path,
                facecolor="none",
                edgecolor="black",
                lw=lw,
                alpha=alpha,
                capstyle="round",
                joinstyle="round",
                zorder=zorder,
            )
            ax.add_patch(patch)

    def _plot_spatial_comm_map(
        self,
        coords: np.ndarray,
        edges: np.ndarray,
        edge_score: np.ndarray,
        *,
        figsize: Tuple[float, float] = (7, 6.6),
        bg_color: str = "white",
        cmap: str = "Reds",
        cmap_min: float = 0.08,
        vmin: float = 0.0,
        vmax: float = 7.0,
        invert_y: bool = False,
        save: Optional[str] = None,
        dpi: int = 300,
        show_intensity_axis: bool = True,
    ) -> None:
        """
        Draw a spatial communication map with:
        - one different irregular tissue block per point;
        - red hotspot fills;
        - thick black curved lines.

        Key change:
        - the lowest color is no longer pure white;
        - low-score cells are rendered in a very light pink.
        """
        coords = np.asarray(coords, dtype=float)
        edges = np.asarray(edges, dtype=int)
        edge_score = np.asarray(edge_score, dtype=float)

        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("coords must be an array of shape (N, 2)")
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("edges must be an array of shape (M, 2)")
        if len(edge_score) != len(edges):
            raise ValueError("edge_score length must match number of edges")

        cmap_obj = self._truncate_cmap(cmap, minval=cmap_min, maxval=1.0)
        norm = Normalize(vmin=vmin, vmax=vmax)
        node_score = self._compute_node_score(len(coords), edges, edge_score)

        fig, ax = plt.subplots(figsize=figsize, facecolor=bg_color)
        ax.set_facecolor(bg_color)

        patches = self._build_irregular_tissue_patches(coords)

        if patches is not None and len(patches) == len(coords):
            # Keep low values as very light red, do not force them to pure white
            cell_values = np.clip(node_score, vmin, vmax)
            cell_colors = cmap_obj(norm(cell_values))

            pc = PatchCollection(
                patches,
                facecolor=cell_colors,
                edgecolor="#2f2f2f",
                linewidths=0.22,
                antialiaseds=True,
                alpha=0.98,
                zorder=1,
            )
            ax.add_collection(pc)
        else:
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=np.clip(node_score, vmin, vmax),
                cmap=cmap_obj,
                norm=norm,
                s=14,
                edgecolors="#2f2f2f",
                linewidths=0.18,
                zorder=1,
            )

        hot_mask = edge_score > vmin
        if np.any(hot_mask):
            self._draw_curved_edges(
                ax=ax,
                coords=coords,
                edges=edges[hot_mask],
                edge_score=edge_score[hot_mask],
                vmin=vmin,
                vmax=vmax,
                alpha_min=0.35,
                alpha_max=0.95,
                lw_min=0.22,
                lw_max=1.0,
                curve_scale=0.2,
                zorder=4.0,
            )

        pad_x = (coords[:, 0].max() - coords[:, 0].min()) * 0.04
        pad_y = (coords[:, 1].max() - coords[:, 1].min()) * 0.04
        ax.set_xlim(coords[:, 0].min() - pad_x, coords[:, 0].max() + pad_x)
        ax.set_ylim(coords[:, 1].min() - pad_y, coords[:, 1].max() + pad_y)

        if invert_y:
            ax.invert_yaxis()

        ax.set_aspect("equal")
        ax.axis("off")

        if show_intensity_axis:
            sm = ScalarMappable(norm=norm, cmap=cmap_obj)
            sm.set_array([])

            cbar = fig.colorbar(
                sm,
                ax=ax,
                fraction=0.032,   # slimmer colorbar
                pad=0.020,        # tighter padding
                aspect=35,        # elongated, journal-style look
                shrink=0.96,
            )
            self._style_nature_colorbar(cbar, vmin=vmin, vmax=vmax)

        plt.tight_layout()

        if save is not None:
            Path(save).parent.mkdir(parents=True, exist_ok=True)
            if path_wants_svg(save):
                configure_matplotlib_svg_for_illustrator()
            plt.savefig(
                save,
                dpi=dpi,
                bbox_inches="tight",
                facecolor=fig.get_facecolor(),
            )

        plt.close(fig)

    def _plot_spatial_comm_map_by_region(
        self,
        coords: np.ndarray,
        edges: np.ndarray,
        edge_score: np.ndarray,
        region_labels: np.ndarray,
        *,
        figsize: Tuple[float, float] = (7, 6.6),
        bg_color: str = "white",
        vmin: float = 0.0,
        vmax: float = 7.0,
        invert_y: bool = False,
        save: Optional[str] = None,
        dpi: int = 300,
        region_palette: str = "tab20",
    ) -> None:
        """
        Draw spatial communication map with cells colored by region (categorical).

        Same layout and curved edges as _plot_spatial_comm_map; cell fill uses
        one color per region and a legend instead of a score colorbar.
        """
        coords = np.asarray(coords, dtype=float)
        edges = np.asarray(edges, dtype=int)
        edge_score = np.asarray(edge_score, dtype=float)
        region_labels = np.asarray(region_labels)

        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("coords must be an array of shape (N, 2)")
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("edges must be an array of shape (M, 2)")
        if len(edge_score) != len(edges):
            raise ValueError("edge_score length must match number of edges")
        if len(region_labels) != len(coords):
            raise ValueError("region_labels length must match number of points")

        unique_regions = sorted(np.unique(region_labels).tolist())
        n_regions = len(unique_regions)
        palette = get_colour_scheme(region_palette, max(n_regions, 1))
        region_to_color = {r: palette[i % len(palette)] for i, r in enumerate(unique_regions)}
        cell_colors = np.array([region_to_color[r] for r in region_labels])

        fig, ax = plt.subplots(figsize=figsize, facecolor=bg_color)
        ax.set_facecolor(bg_color)

        patches = self._build_irregular_tissue_patches(coords)

        if patches is not None and len(patches) == len(coords):
            pc = PatchCollection(
                patches,
                facecolor=cell_colors,
                edgecolor="#2f2f2f",
                linewidths=0.22,
                antialiaseds=True,
                alpha=0.98,
                zorder=1,
            )
            ax.add_collection(pc)
        else:
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=cell_colors,
                s=14,
                edgecolors="#2f2f2f",
                linewidths=0.18,
                zorder=1,
            )

        if edges.size > 0 and np.any(edge_score > vmin):
            hot_mask = edge_score > vmin
            self._draw_curved_edges(
                ax=ax,
                coords=coords,
                edges=edges[hot_mask],
                edge_score=edge_score[hot_mask],
                vmin=vmin,
                vmax=vmax,
                alpha_min=0.35,
                alpha_max=0.95,
                lw_min=0.22,
                lw_max=1.0,
                curve_scale=0.2,
                zorder=4.0,
            )

        pad_x = (coords[:, 0].max() - coords[:, 0].min()) * 0.04
        pad_y = (coords[:, 1].max() - coords[:, 1].min()) * 0.04
        ax.set_xlim(coords[:, 0].min() - pad_x, coords[:, 0].max() + pad_x)
        ax.set_ylim(coords[:, 1].min() - pad_y, coords[:, 1].max() + pad_y)

        if invert_y:
            ax.invert_yaxis()

        ax.set_aspect("equal")
        ax.axis("off")

        legend_handles = [
            Patch(facecolor=region_to_color[r], edgecolor="#2f2f2f", label=str(r))
            for r in unique_regions
        ]
        ax.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            fontsize=8,
        )

        plt.tight_layout()

        if save is not None:
            Path(save).parent.mkdir(parents=True, exist_ok=True)
            if path_wants_svg(save):
                configure_matplotlib_svg_for_illustrator()
            plt.savefig(
                save,
                dpi=dpi,
                bbox_inches="tight",
                facecolor=fig.get_facecolor(),
            )

        plt.close(fig)