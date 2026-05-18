from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
from anndata import AnnData

from utils.viz.matplotlib_svg import configure_matplotlib_svg_for_illustrator, path_wants_svg


class SenderReceiverStackedBarVisualizer:
    """
    Nature-style sender/receiver stacked bars for LR communication.

    Layout:
    - Mirrored around y=0 on a single axis:
      sender strength is stacked upward,
      receiver strength is stacked downward.

    Bars correspond to LR pairs; stacks correspond to cell types.
    """

    def _setup_nature_style(self) -> None:
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": [
                    "Arial",
                    "Helvetica",
                    "DejaVu Sans",
                    "Liberation Sans",
                    "sans-serif",
                ],
                "font.size": 8,
                "axes.titlesize": 9,
                "axes.labelsize": 8.5,
                "xtick.labelsize": 6.5,
                "ytick.labelsize": 7.5,
                "legend.fontsize": 7,
                "axes.linewidth": 0.8,
                "axes.edgecolor": "#222222",
                "xtick.major.width": 0.8,
                "ytick.major.width": 0.8,
                "xtick.minor.width": 0.6,
                "ytick.minor.width": 0.6,
                "xtick.major.size": 3,
                "ytick.major.size": 3,
                "xtick.minor.size": 2,
                "ytick.minor.size": 2,
                "xtick.direction": "out",
                "ytick.direction": "out",
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "savefig.facecolor": "white",
                "savefig.edgecolor": "none",
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "svg.fonttype": "none",
            }
        )

    def _get_valid_edge_types(
        self,
        edge_type_map: Dict[str, int],
        lr_filter: Optional[List[str]] = None,
    ) -> List[int]:
        if lr_filter is None:
            return list(edge_type_map.values())
        return [edge_type_map[k] for k in lr_filter if k in edge_type_map]

    @staticmethod
    def _safe_sum(x: np.ndarray) -> float:
        return float(np.sum(x)) if x.size else 0.0

    @staticmethod
    def _nature_palette(n: int) -> List[str]:
        # More restrained, publication-friendly palette
        palette = [
            "#4E79A7",  # blue
            "#E15759",  # red
            "#59A14F",  # green
            "#F28E2B",  # orange
            "#76B7B2",  # teal
            "#B07AA1",  # purple
            "#EDC948",  # yellow
            "#9C755F",  # brown
            "#BAB0AC",  # grey
            "#2F6B9A",
            "#D37295",
            "#8CD17D",
        ]
        if n <= len(palette):
            return palette[:n]
        repeat = (n // len(palette)) + 1
        return (palette * repeat)[:n]

    @staticmethod
    def _format_axis(ax: plt.Axes) -> None:
        """
        Mirror axis for fractions:
        ticks shown as absolute percentages on both sides.
        """
        ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(
            FuncFormatter(
                lambda y, _: "0" if abs(y) < 1e-10 else f"{abs(y) * 100:.0f}%"
            )
        )

    @staticmethod
    def _break_ligand_receptor_label(label: str) -> str:
        """
        Split label at the first separator so display is 'Ligand\\nReceptor'.
        Used for last-level stacked bars where names are real LR pairs.
        """
        for sep in ["_", "-", "–", "|", "/"]:
            if sep in label:
                parts = label.split(sep, 1)
                if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                    return f"{parts[0].strip()}\n{parts[1].strip()}"
        return label

    @staticmethod
    def _wrap_lr_label(label: str, max_line_len: int = 14, max_lines: int = 2) -> str:
        """
        Wrap LR labels to reduce overlap.
        Prefer splitting at '-', '_', '|', '/', ':'.
        """
        label = label.replace(":", "–")

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
        split_idx = midpoint
        for shift in range(0, min(8, len(label) // 2)):
            right = midpoint + shift
            left = midpoint - shift
            if right < len(label) and label[right] in [".", "–", "-", "_", "/"]:
                split_idx = right + 1
                break
            if left > 0 and label[left] in [".", "–", "-", "_", "/"]:
                split_idx = left + 1
                break

        line1 = label[:split_idx].strip()
        line2 = label[split_idx:].strip()
        if len(line2) > max_line_len - 1:
            line2 = line2[: max_line_len - 1] + "…"
        return f"{line1}\n{line2}"

    def plot(
        self,
        *,
        graph: np.ndarray,
        edge_type_map: Dict[str, int],
        adata: AnnData,
        cell_type_key: str = "cell_type",
        lr_filter: Optional[List[str]] = None,
        top_n: Optional[int] = None,
        threshold: float = 0.0,
        order_by: str = "total",
        save_path: Union[str, Path] = "lr_sender_receiver_stacked_bar.svg",
        figsize: Tuple[int, int] = (7, 6),
        dpi: int = 600,
        top_k_cell_types: int = 10,
        include_others: bool = True,
        break_lr_labels: bool = False,
    ) -> None:
        self._setup_nature_style()

        if cell_type_key not in adata.obs:
            raise ValueError(f"cell_type_key '{cell_type_key}' not found in adata.obs.")

        if graph.ndim != 3:
            raise ValueError(
                f"`graph` must be a 3D array of shape (n_cells, n_cells, n_edge_types), "
                f"but got shape {graph.shape}."
            )

        if graph.shape[0] != graph.shape[1]:
            raise ValueError(
                f"`graph` must have shape (n_cells, n_cells, n_edge_types), "
                f"but got shape {graph.shape}."
            )

        if graph.shape[0] != adata.n_obs:
            raise ValueError(
                f"Mismatch between graph cell dimension ({graph.shape[0]}) "
                f"and adata.n_obs ({adata.n_obs})."
            )

        valid_edge_types = self._get_valid_edge_types(edge_type_map, lr_filter)
        if len(valid_edge_types) == 0:
            print("Warning: No valid edge types found")
            return

        # ------------------------------------------------------------------
        # 1. Select LR pairs by total communication
        # ------------------------------------------------------------------
        lr_contributions: Dict[str, Dict[str, object]] = {}
        for lr_name, edge_idx in edge_type_map.items():
            if edge_idx not in valid_edge_types:
                continue
            if edge_idx < 0 or edge_idx >= graph.shape[2]:
                continue

            edge_probs = graph[:, :, edge_idx]
            if threshold > 0.0:
                edge_probs = edge_probs.copy()
                edge_probs[edge_probs < threshold] = 0.0

            n_cells = edge_probs.shape[0]
            non_diag = ~np.eye(n_cells, dtype=bool)
            total = float(edge_probs[non_diag].sum())

            if total > 0:
                lr_contributions[lr_name] = {
                    "total": total,
                    "edge_probs": edge_probs,
                }

        if not lr_contributions:
            print("Warning: No LR pairs with positive contributions found")
            return

        sorted_by_total = sorted(
            lr_contributions.items(),
            key=lambda x: (-float(x[1]["total"]), str(x[0])),
        )
        if top_n is not None:
            sorted_by_total = sorted_by_total[: int(top_n)]

        if str(order_by).lower() in ("edge_index", "index", "node", "node_id"):
            sorted_lr = sorted(
                sorted_by_total,
                key=lambda x: int(edge_type_map.get(str(x[0]), 1_000_000_000)),
            )
        else:
            sorted_lr = sorted_by_total

        lr_names = [name for name, _ in sorted_lr]
        n_lr = len(lr_names)

        # ------------------------------------------------------------------
        # 2. Choose top cell types by total sender + receiver contribution
        # ------------------------------------------------------------------
        cell_types = adata.obs[cell_type_key].astype(str).to_numpy()
        unique_cell_types = np.unique(cell_types)
        ct_totals: Dict[str, float] = {ct: 0.0 for ct in unique_cell_types}

        for ct in unique_cell_types:
            src_mask = cell_types == ct
            tgt_mask = src_mask
            total = 0.0

            for _, lr_data in sorted_lr:
                edge_probs = lr_data["edge_probs"]
                within = edge_probs[src_mask, :][:, tgt_mask]
                sender = self._safe_sum(edge_probs[src_mask, :]) - self._safe_sum(within)
                receiver = self._safe_sum(edge_probs[:, tgt_mask]) - self._safe_sum(within)
                total += sender + receiver

            ct_totals[ct] = total

        sorted_ct = sorted(ct_totals.items(), key=lambda x: x[1], reverse=True)

        top_cell_types = [
            ct
            for ct, _ in sorted_ct[: max(1, int(top_k_cell_types))]
            if ct_totals.get(ct, 0.0) > 0
        ]

        other_cell_types = [ct for ct, _ in sorted_ct if ct not in set(top_cell_types)]

        cell_type_bins = list(top_cell_types)
        if include_others and other_cell_types:
            cell_type_bins.append("Others")

        # ------------------------------------------------------------------
        # 3. Build sender / receiver matrices
        # ------------------------------------------------------------------
        n_ct = len(cell_type_bins)
        sender_mat = np.zeros((n_lr, n_ct), dtype=np.float64)
        receiver_mat = np.zeros((n_lr, n_ct), dtype=np.float64)

        for li, (_, lr_data) in enumerate(sorted_lr):
            edge_probs = lr_data["edge_probs"]

            for ci, ct in enumerate(cell_type_bins):
                if ct == "Others":
                    masks = [(cell_types == o) for o in other_cell_types]
                    if not masks:
                        continue
                    src_mask = np.logical_or.reduce(masks)
                    tgt_mask = src_mask
                else:
                    src_mask = cell_types == ct
                    tgt_mask = src_mask

                within = edge_probs[src_mask, :][:, tgt_mask]
                sender_mat[li, ci] = (
                    self._safe_sum(edge_probs[src_mask, :]) - self._safe_sum(within)
                )
                receiver_mat[li, ci] = (
                    self._safe_sum(edge_probs[:, tgt_mask]) - self._safe_sum(within)
                )

        # ------------------------------------------------------------------
        # 4. Normalize to fractions
        # ------------------------------------------------------------------
        sender_totals = sender_mat.sum(axis=1, keepdims=True)
        receiver_totals = receiver_mat.sum(axis=1, keepdims=True)

        sender_totals_safe = np.where(sender_totals > 0, sender_totals, 1.0)
        receiver_totals_safe = np.where(receiver_totals > 0, receiver_totals, 1.0)

        sender_mat = sender_mat / sender_totals_safe
        receiver_mat = receiver_mat / receiver_totals_safe

        # ------------------------------------------------------------------
        # 5. Colors
        # ------------------------------------------------------------------
        main_cts = [c for c in cell_type_bins if c != "Others"]
        ct_colors = self._nature_palette(len(main_cts))

        color_map: Dict[str, str] = {}
        for i, ct in enumerate(main_cts):
            color_map[ct] = ct_colors[i]

        if "Others" in cell_type_bins:
            color_map["Others"] = "#C9C9C9"

        # ------------------------------------------------------------------
        # 6. LR label display
        # ------------------------------------------------------------------
        if n_lr <= 6:
            max_line_len = 18
            rotation = 0
            x_fontsize = 7.8
            label_ha = "center"
            width = 0.78
        elif n_lr <= 10:
            max_line_len = 16
            rotation = 25
            x_fontsize = 7.2
            label_ha = "right"
            width = 0.76
        elif n_lr <= 16:
            max_line_len = 14
            rotation = 38
            x_fontsize = 6.8
            label_ha = "right"
            width = 0.74
        else:
            max_line_len = 12
            rotation = 45
            x_fontsize = 6.4
            label_ha = "right"
            width = 0.70

        if break_lr_labels:
            display_lr = [self._break_ligand_receptor_label(n) for n in lr_names]
        else:
            display_lr = [self._wrap_lr_label(n, max_line_len=max_line_len) for n in lr_names]

        # ------------------------------------------------------------------
        # 7. Plot
        # ------------------------------------------------------------------
        x = np.arange(n_lr, dtype=float)

        fig, ax = plt.subplots(
            1,
            1,
            figsize=figsize,
            dpi=max(int(dpi), 600),
        )

        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.set_axisbelow(True)

        # Spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.75)
        ax.spines["bottom"].set_linewidth(0.75)
        ax.spines["left"].set_color("#333333")
        ax.spines["bottom"].set_color("#333333")

        # Grid
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.55, linestyle="-", alpha=1.0)
        ax.grid(axis="x", visible=False)

        # Ticks
        ax.tick_params(
            axis="both",
            colors="#222222",
            length=3.0,
            width=0.7,
            pad=2.5,
        )

        # Center line
        ax.axhline(0.0, color="#222222", linewidth=0.95, zorder=5)

        # Sender bars
        pos_bottom = np.zeros(n_lr, dtype=np.float64)
        legend_handles = []
        legend_labels = []

        for ci, ct in enumerate(cell_type_bins):
            vals = sender_mat[:, ci]
            bars = ax.bar(
                x,
                vals,
                bottom=pos_bottom,
                width=width,
                color=color_map.get(ct, "#4E79A7"),
                edgecolor="white",
                linewidth=0.45,
                alpha=1.0,
                zorder=3,
                label=ct,
            )
            pos_bottom += vals
            if len(bars) > 0:
                legend_handles.append(bars[0])
                legend_labels.append(ct)

        # Receiver bars
        neg_bottom = np.zeros(n_lr, dtype=np.float64)
        for ci, ct in enumerate(cell_type_bins):
            vals = receiver_mat[:, ci]
            ax.bar(
                x,
                -vals,
                bottom=-neg_bottom,
                width=width,
                color=color_map.get(ct, "#4E79A7"),
                edgecolor="white",
                linewidth=0.45,
                alpha=1.0,
                zorder=3,
            )
            neg_bottom += vals

        # ------------------------------------------------------------------
        # 8. Axis format
        # ------------------------------------------------------------------
        ax.set_ylim(-1.02, 1.02)
        ax.set_xlim(-0.55, n_lr - 0.45)
        ax.margins(x=0.01)

        # Mirror-style y tick labels
        yticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
        yticklabels = ["100%", "50%", "0", "50%", "100%"]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, color="#222222")
        ax.tick_params(axis="y", labelsize=7.4, pad=2)

        ax.set_ylabel("Communication fraction", labelpad=6, color="#222222")

        ax.set_xticks(x)
        ax.set_xticklabels(
            display_lr,
            rotation=rotation,
            ha=label_ha,
            color="#222222",
            linespacing=0.95,
        )
        ax.tick_params(axis="x", labelsize=x_fontsize, pad=3)

        # Sender / Receiver labels
        ax.text(
            0.01,
            0.985,
            "Sender",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.3,
            color="#222222",
        )
        ax.text(
            0.01,
            0.015,
            "Receiver",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.3,
            color="#222222",
        )

        # ------------------------------------------------------------------
        # 9. Legend at top
        # ------------------------------------------------------------------
        if n_ct >= 1:
            if n_ct <= 4:
                ncol = n_ct
            elif n_ct <= 8:
                ncol = 4
            else:
                ncol = 5

            leg = ax.legend(
                legend_handles,
                legend_labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.03),
                frameon=False,
                borderaxespad=0.0,
                handlelength=1.1,
                handletextpad=0.45,
                labelspacing=0.40,
                columnspacing=0.90,
                ncol=ncol,
            )

            for txt in leg.get_texts():
                txt.set_color("#222222")
                txt.set_fontsize(7.0)

            legend_rows = int(np.ceil(n_ct / ncol))
        else:
            legend_rows = 0

        # ------------------------------------------------------------------
        # 10. Layout
        # ------------------------------------------------------------------
        extra_bottom = 0.0
        extra_top = 0.0
        if rotation >= 38:
            extra_bottom += 0.05
        elif rotation >= 25:
            extra_bottom += 0.03

        extra_top += 0.045 * max(legend_rows, 1)

        fig.subplots_adjust(
            left=0.12,
            right=0.98,
            top=max(0.78, 0.95 - extra_top),
            bottom=min(0.23 + extra_bottom, 0.42),
        )

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if path_wants_svg(save_path):
            configure_matplotlib_svg_for_illustrator()
        plt.savefig(
            save_path,
            dpi=max(int(dpi), 600),
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
            pad_inches=0.03,
        )
        plt.close(fig)
        print(f"Sender/receiver stacked bar saved to {save_path}")