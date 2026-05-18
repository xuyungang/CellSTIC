"""Spot-level evaluation: Spearman correlation + hotspot overlap (top-k% Jaccard)."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

_NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Helvetica", "Arial", "sans-serif"],
    "font.size": 12,
    "axes.linewidth": 0.75,
    "axes.edgecolor": "#000000",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "none",
    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,
}


def compute_spot_level_metrics(
    pos_edge_probs_np: np.ndarray,
    ccc_ground: np.ndarray,
    edge_type_map: Dict[str, int],
    lr_filter: List[str],
    score_method: str = "mean",
    threshold: float = 0.0,
    hotspot_top_pct: float = 10.0,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 5),
    dpi: int = 600,
) -> List[Dict[str, float]]:
    """
    Compute Spearman + hotspot overlap (Jaccard) per LR channel.
    Optionally plot grouped bar chart in a Nature-like style.

    Args:
        pos_edge_probs_np: Predicted (n, n, m)
        ccc_ground: Ground truth (n, n, m), same shape
        edge_type_map: channel key -> index
        lr_filter: List of edge_type_map keys (e.g. "CCL7:CCR2", "root" for virtual nodes)
        score_method: 'mean'|'sum' for per-cell scores
        threshold: Min edge value (applied to both pred and gt)
        hotspot_top_pct: Top k% of spots as hotspots (default 10)
        save_path: If provided, save bar chart
        figsize: Figure size for bar chart
        dpi: DPI for bar chart

    Returns:
        List of dicts: [{"lr": "...", "spearman": ..., "hotspot_jaccard": ...}, ...]
    """
    from matplotlib.ticker import MaxNLocator

    def _wrap_label(label: str, max_line_len: int = 14, max_lines: int = 2) -> str:
        label = str(label).replace(":", "–")
        if len(label) <= max_line_len:
            return label

        seps = ["–", "-", "_", "/", "|"]
        parts = [label]
        sep_used = None
        for sep in seps:
            if sep in label:
                parts = label.split(sep)
                sep_used = sep
                break

        if len(parts) > 1:
            lines = []
            current = parts[0]
            for p in parts[1:]:
                token = (sep_used if sep_used is not None else "") + p
                if len(current) + len(token) <= max_line_len:
                    current += token
                else:
                    lines.append(current)
                    current = p
            lines.append(current)

            if len(lines) > max_lines:
                merged = lines[: max_lines - 1]
                tail = "".join(lines[max_lines - 1 :])
                if len(tail) > max_line_len - 1:
                    tail = tail[: max_line_len - 1] + "…"
                merged.append(tail)
                lines = merged

            return "\n".join(lines)

        midpoint = len(label) // 2
        return label[:midpoint] + "\n" + label[midpoint:]

    ccc_ground = np.asarray(ccc_ground)
    if pos_edge_probs_np.shape != ccc_ground.shape:
        raise ValueError(
            f"Shape mismatch: pred {pos_edge_probs_np.shape} vs ground {ccc_ground.shape}"
        )

    n_edges = pos_edge_probs_np.shape[2]
    results = []

    for lr_key in lr_filter:
        edge_idx = edge_type_map.get(lr_key)
        if edge_idx is None or edge_idx >= n_edges:
            results.append(
                {"lr": lr_key, "spearman": np.nan, "hotspot_jaccard": np.nan}
            )
            continue

        pred_mat = pos_edge_probs_np[:, :, edge_idx].copy()
        gt_mat = ccc_ground[:, :, edge_idx].copy()

        if threshold > 0:
            pred_mat = np.where(pred_mat >= threshold, pred_mat, 0.0)
            gt_mat = np.where(gt_mat >= threshold, gt_mat, 0.0)

        def _per_cell_scores(mat: np.ndarray, method: str) -> np.ndarray:
            n = mat.shape[0]
            d = np.diag(mat)
            out_s = mat.sum(axis=1) - d
            in_s = mat.sum(axis=0) - d
            if method == "sum":
                return out_s + in_s
            if method == "mean":
                return (out_s + in_s) / max(1, 2 * (n - 1))
            if method == "incoming":
                return in_s / max(1, n - 1)
            return out_s / max(1, n - 1)

        pred_scores = np.asarray(_per_cell_scores(pred_mat, score_method), dtype=float)
        gt_scores = np.asarray(_per_cell_scores(gt_mat, score_method), dtype=float)

        # Spearman
        valid = np.isfinite(pred_scores) & np.isfinite(gt_scores)
        if np.sum(valid) < 3:
            spearman_val = np.nan
        else:
            r, _ = spearmanr(pred_scores[valid], gt_scores[valid])
            spearman_val = float(r) if np.isfinite(r) else np.nan

        # Hotspot overlap: top k% of spots, Jaccard
        n_spots = len(pred_scores)
        k = max(1, int(np.ceil(n_spots * hotspot_top_pct / 100.0)))

        pred_rank_scores = np.where(np.isfinite(pred_scores), pred_scores, -np.inf)
        gt_rank_scores = np.where(np.isfinite(gt_scores), gt_scores, -np.inf)

        gt_top_indices = set(np.argsort(gt_rank_scores)[-k:])
        pred_top_indices = set(np.argsort(pred_rank_scores)[-k:])

        intersection = len(gt_top_indices & pred_top_indices)
        union = len(gt_top_indices | pred_top_indices)
        jaccard_val = intersection / union if union > 0 else np.nan

        results.append(
            {
                "lr": lr_key,
                "spearman": spearman_val,
                "hotspot_jaccard": float(jaccard_val) if np.isfinite(jaccard_val) else np.nan,
            }
        )

    # Plot grouped bar chart if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        lr_names = [r["lr"] for r in results]
        display_lr = [_wrap_label(x) for x in lr_names]

        spearman_vals = np.array([r["spearman"] for r in results], dtype=float)
        jaccard_vals = np.array([r["hotspot_jaccard"] for r in results], dtype=float)

        # Preserve negative Spearman for plotting
        spearman_plot = np.where(np.isfinite(spearman_vals), spearman_vals, 0.0)
        jaccard_plot = np.where(np.isfinite(jaccard_vals), jaccard_vals, 0.0)

        spearman_labels = [f"{v:.2f}" if np.isfinite(v) else "-" for v in spearman_vals]
        jaccard_labels = [f"{v:.2f}" if np.isfinite(v) else "-" for v in jaccard_vals]

        x = np.arange(len(lr_names), dtype=float)
        width = 0.36

        plt.rcParams.update(_NATURE_RC)
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
                "font.size": 8,
                "axes.labelsize": 8.5,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "legend.fontsize": 7,
                "axes.linewidth": 0.8,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "svg.fonttype": "none",
            }
        )

        fig, ax = plt.subplots(figsize=figsize, dpi=max(int(dpi), 600), facecolor="white")
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.set_axisbelow(True)

        color_spearman = "#4E79A7"
        color_jaccard = "#E15759"

        bars1 = ax.bar(
            x - width / 2,
            spearman_plot,
            width,
            label="Spearman",
            color=color_spearman,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        bars2 = ax.bar(
            x + width / 2,
            jaccard_plot,
            width,
            label="Hotspot Jaccard",
            color=color_jaccard,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )

        # Zero baseline helps interpret negative Spearman
        ax.axhline(0, color="#222222", linewidth=0.8, zorder=2)

        # Bar labels: only show when count is moderate, otherwise too cluttered
        if len(lr_names) <= 20:
            for bar, txt, val in zip(bars1, spearman_labels, spearman_plot):
                y = val + 0.02 if val >= 0 else val - 0.04
                va = "bottom" if val >= 0 else "top"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y,
                    txt,
                    ha="center",
                    va=va,
                    fontsize=6.5,
                    color="#222222",
                    rotation=90,
                )

            for bar, txt, val in zip(bars2, jaccard_labels, jaccard_plot):
                y = val + 0.02
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y,
                    txt,
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color="#222222",
                    rotation=90,
                )

        ax.set_ylabel("Score", fontsize=8.5, color="#222222", labelpad=4)
        ax.set_xticks(x)

        # X-axis LR labels: tilt down-left (45° CCW, anchor at top-right of text box)
        x_fs = 7.0 if len(lr_names) <= 12 else 6.1
        ax.set_xticklabels(
            display_lr,
            rotation=45,
            ha="right",
            va="top",
            rotation_mode="anchor",
            fontsize=x_fs,
            color="#222222",
        )
        ax.tick_params(axis="x", which="major", width=0.8, length=3, colors="#222222", pad=2)
        ax.tick_params(axis="y", which="major", labelsize=7, width=0.8, length=3, colors="#222222", pad=2)

        # Y range: include negative Spearman if present
        finite_s = spearman_vals[np.isfinite(spearman_vals)]
        y_min = min([-0.05] + ([float(finite_s.min())] if finite_s.size else [0.0]))
        y_max = 1.02

        if y_min < 0:
            ax.set_ylim(max(-1.02, y_min - 0.08), y_max)
        else:
            ax.set_ylim(0, y_max)

        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

        # Light grid only on y-axis
        ax.grid(axis="y", linestyle="-", linewidth=0.5, alpha=0.7, color="#D9D9D9")
        ax.grid(axis="x", visible=False)

        # Clean spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.spines["left"].set_color("#222222")
        ax.spines["bottom"].set_color("#222222")

        leg = ax.legend(
            loc="upper right",
            frameon=False,
            handlelength=1.2,
            handletextpad=0.5,
            labelspacing=0.35,
            borderaxespad=0.3,
        )
        for txt in leg.get_texts():
            txt.set_color("#222222")

        plt.tight_layout(pad=0.5)
        fig.savefig(
            save_path,
            dpi=max(int(dpi), 600),
            bbox_inches="tight",
            pad_inches=0.03,
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig)
        print(f"Spot-level metrics bar chart saved to {save_path}")

    return results