"""Domain-level visualizations: stacked bar of cell type composition per region (Nature style)."""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from utils.viz.matplotlib_svg import configure_matplotlib_svg_for_illustrator, path_wants_svg
import pandas as pd
import matplotlib.pyplot as plt
from anndata import AnnData


def load_domain_from_csv(
    adata: AnnData,
    domain_path: Union[str, Path],
    cell_id_column: str = "cell_id",
    domain_column: str = "cluster",
    domain_obs_key: str = "domain",
) -> None:
    """
    Load domain labels from a CSV and merge into adata.obs.
    CSV must have a cell ID column (matching adata.obs_names) and a domain label column.
    Unmatched cells get label "unknown".
    """
    path = Path(domain_path)
    if not path.exists():
        raise FileNotFoundError(f"domain_path not found: {path}")
    df = pd.read_csv(path)
    if cell_id_column not in df.columns or domain_column not in df.columns:
        raise ValueError(
            f"domain CSV must have columns '{cell_id_column}' and '{domain_column}'; got {list(df.columns)}"
        )
    df = df.set_index(cell_id_column)
    domain_series = df[domain_column].astype(str).reindex(adata.obs_names)
    if domain_series.isna().all():
        raise ValueError("No overlap between adata.obs_names and domain CSV cell ids")
    domain_series = domain_series.fillna("unknown")
    adata.obs[domain_obs_key] = pd.Categorical(domain_series.values)


class DomainVisualizer:
    """Nature-style domain visualizations (e.g. stacked bar of cell types per region)."""

    def __init__(self) -> None:
        pass

    def _setup_nature_style(self) -> None:
        """Apply Nature-style rcParams (match CCC heatmap/stacked bar conventions)."""
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Helvetica", "Arial", "sans-serif"],
            "font.size": 8,
            "axes.linewidth": 0.75,
            "axes.edgecolor": "#000000",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
        })

    def plot_domain_cell_type_stacked_bar(
        self,
        adata: AnnData,
        domain_key: str = "domain",
        cell_type_key: str = "cell_type",
        domain_path: Optional[Union[str, Path]] = None,
        domain_file_cell_id_column: str = "cell_id",
        domain_file_domain_column: str = "cluster",
        top_n_cell_types: Optional[int] = 15,
        save_path: Union[str, Path] = "domain_cell_type_stacked_bar.svg",
        figsize: Tuple[float, float] = (10, 5),
        dpi: int = 600,
        legend_loc: str = "outside right",
    ) -> None:
        """
        Plot stacked bar chart: one bar per domain/region, stacked by cell type.
        If domain_path is set, load domain labels from CSV and merge into adata.obs["domain"] first.

        Notes
        -----
        - Bar height equals total cells per domain (raw counts).
        - Styling is optimized toward a cleaner, Nature-like publication figure.
        """
        from pathlib import Path
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        # -----------------------------
        # 1. Load external domain labels if provided
        # -----------------------------
        if domain_path is not None:
            load_domain_from_csv(
                adata,
                domain_path=Path(domain_path),
                cell_id_column=domain_file_cell_id_column,
                domain_column=domain_file_domain_column,
                domain_obs_key="domain",
            )
            domain_key = "domain"

        if domain_key not in adata.obs:
            raise ValueError(
                f"domain_key '{domain_key}' not found in adata.obs "
                f"(set domain_path or ensure column exists)"
            )
        if cell_type_key not in adata.obs:
            raise ValueError(f"cell_type_key '{cell_type_key}' not found in adata.obs")

        # -----------------------------
        # 2. Clean inputs
        # -----------------------------
        obs_df = adata.obs[[domain_key, cell_type_key]].copy()
        obs_df = obs_df.dropna(subset=[domain_key, cell_type_key])

        domains = obs_df[domain_key].astype(str)
        cell_types = obs_df[cell_type_key].astype(str)

        if len(domains) == 0:
            raise ValueError("No valid observations remain after removing NA values.")

        # Crosstab: rows = domains, cols = cell types
        ct = pd.crosstab(domains, cell_types)

        # Preserve categorical order if domain column is categorical and ordered;
        # otherwise sort by total cells descending
        if pd.api.types.is_categorical_dtype(adata.obs[domain_key]):
            cat = adata.obs[domain_key].cat
            ordered_domains = [str(x) for x in cat.categories if str(x) in ct.index]
            if len(ordered_domains) > 0:
                ct = ct.loc[ordered_domains]
            else:
                ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]
        else:
            ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]

        # -----------------------------
        # 3. Collapse minor cell types
        # -----------------------------
        if top_n_cell_types is not None and ct.shape[1] > top_n_cell_types:
            col_sum = ct.sum(axis=0).sort_values(ascending=False)
            keep = col_sum.head(top_n_cell_types).index.tolist()
            others = ct.drop(columns=keep).sum(axis=1)
            ct = ct[keep].copy()
            ct["Others"] = others

        # Total cells per domain (we always plot raw counts)
        plot_df = ct.copy()

        categories = plot_df.columns.tolist()
        x_labels = plot_df.index.tolist()
        n_bars = len(x_labels)

        # -----------------------------
        # 4. Nature-like styling
        # -----------------------------
        # Prefer a clean sans-serif style
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
        })

        # A restrained, publication-friendly palette
        # Extended enough for many categories; "Others" gets neutral gray
        nature_palette = [
            "#4E79A7", "#E15759", "#59A14F", "#F28E2B", "#76B7B2",
            "#B07AA1", "#EDC948", "#9C755F", "#BAB0AC", "#2F6B9A",
            "#D37295", "#8CD17D", "#B6992D", "#499894", "#86BCB6",
            "#A0CBE8", "#FFBE7D", "#FF9D9A", "#79706E", "#D4A6C8"
        ]

        color_map = {}
        color_idx = 0
        for c in categories:
            if c.lower() == "others":
                color_map[c] = "#BDBDBD"
            else:
                color_map[c] = nature_palette[color_idx % len(nature_palette)]
                color_idx += 1

        # Optional call to your own style setup if needed
        if hasattr(self, "_setup_nature_style"):
            self._setup_nature_style()

        _dpi = 600 if dpi is None else max(dpi, 600)
        fig, ax = plt.subplots(figsize=figsize, dpi=_dpi, facecolor="white")
        ax.set_facecolor("white")

        # -----------------------------
        # 5. Draw stacked bars
        # -----------------------------
        x = np.arange(n_bars)
        bottom = np.zeros(n_bars, dtype=float)

        for cat in categories:
            vals = plot_df[cat].values.astype(float)
            ax.bar(
                x,
                vals,
                bottom=bottom,
                width=0.72,
                label=cat,
                color=color_map[cat],
                edgecolor="white",   # cleaner than black outlines
                linewidth=0.45,
                alpha=1.0,
                zorder=3,
            )
            bottom += vals

        # -----------------------------
        # 6. Axes formatting
        # -----------------------------
        # Adaptive x tick rotation for readability
        if n_bars <= 8:
            rotation = 0
            ha = "center"
        elif n_bars <= 16:
            rotation = 30
            ha = "right"
        else:
            rotation = 45
            ha = "right"

        ax.set_xticks(x)
        ax.set_xticklabels(
            x_labels,
            fontsize=8,
            rotation=rotation,
            ha=ha,
            color="black",
        )

        ax.set_xlabel("Region / Domain", fontsize=9, color="black", labelpad=5)
        ax.set_ylabel("Number of cells", fontsize=9, color="black", labelpad=5)

        ax.set_xlim(-0.55, n_bars - 0.45)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

        # Subtle horizontal grid only
        ax.grid(axis="y", linestyle="-", linewidth=0.5, color="#D9D9D9", alpha=0.7, zorder=0)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)

        # Clean spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.spines["left"].set_color("black")
        ax.spines["bottom"].set_color("black")

        ax.tick_params(axis="x", labelsize=8, colors="black", width=0.8, length=3, pad=2)
        ax.tick_params(axis="y", labelsize=8, colors="black", width=0.8, length=3, pad=2)

        # Headroom
        ymax = float(bottom.max()) if len(bottom) > 0 else 1.0
        ax.set_ylim(0, ymax * 1.04)

        # -----------------------------
        # 7. Legend
        # -----------------------------
        if len(categories) > 1:
            # More compact column logic
            if len(categories) <= 10:
                ncol = 1
            elif len(categories) <= 20:
                ncol = 2
            else:
                ncol = 3

            common_legend_kwargs = dict(
                fontsize=7.5,
                frameon=False,         # cleaner, more journal-like
                borderpad=0.2,
                labelspacing=0.35,
                handlelength=1.1,
                handleheight=0.8,
                handletextpad=0.45,
                columnspacing=0.8,
                ncol=ncol,
            )

            if legend_loc == "upper right":
                legend = ax.legend(
                    loc="upper right",
                    **common_legend_kwargs,
                )
            else:
                legend = ax.legend(
                    loc="upper left",
                    bbox_to_anchor=(1.01, 1.0),
                    borderaxespad=0.0,
                    **common_legend_kwargs,
                )

            for text in legend.get_texts():
                text.set_color("black")

        # -----------------------------
        # 8. Layout and save
        # -----------------------------
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Slightly different layout handling depending on legend placement
        if legend_loc == "outside right" and len(categories) > 1:
            plt.tight_layout(rect=[0, 0, 0.84, 1], pad=0.6)
        else:
            plt.tight_layout(pad=0.6)

        if path_wants_svg(save_path):
            configure_matplotlib_svg_for_illustrator()
        plt.savefig(
            save_path,
            dpi=_dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
            pad_inches=0.03,
        )
        plt.close(fig)
        print(f"Domain cell-type stacked bar chart saved to {save_path}")