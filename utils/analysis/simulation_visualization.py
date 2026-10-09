"""Side-by-side simulation ground truth and predicted incoming signals."""
from __future__ import annotations


import json

from pathlib import Path


import matplotlib as mpl

import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm

import numpy as np

import pandas as pd


SCENARIOS = ("domains", "gradients", "heterogeneous")

SCENARIO_LABELS = {
    "domains": "Domains",
    "gradients": "Gradients",
    "heterogeneous": "Heterogeneous",
}

FIGURE_SIZE = (7.6, 5.7)

THRESHOLD = 0.5

TRUTH_CMAP = LinearSegmentedColormap.from_list(
    "truth_blue", ["#EFF3F7", "#B5C9DE", "#5D88B2", "#244E78"]
)

PRED_CMAP = LinearSegmentedColormap.from_list(
    "prediction_coral", ["#F8F4F2", "#F4BE99", "#D97465", "#8D3056"]
)


STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
    "font.size": 7.3,
    "axes.titlesize": 8.0,
    "axes.labelsize": 7.0,
    "xtick.labelsize": 6.1,
    "ytick.labelsize": 6.1,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
}

def cell_source_table(data: dict) -> pd.DataFrame:
    coords = data["coords"]
    truth = data["truth"]
    prediction = data["prediction"]
    return pd.DataFrame({
        "scenario": data["scenario"],
        "seed": data["seed"],
        "dataset_id": data["dataset_id"],
        "pair_id": data["pair_id"],
        "ligand_receptor": data["name"],
        "cell_id": data["cell_ids"],
        "x": coords[:, 0],
        "y": coords[:, 1],
        "truth_incoming_edges": np.asarray(truth.sum(axis=0)).ravel().astype(int),
        "predicted_incoming_strength": np.asarray(prediction.sum(axis=0)).ravel(),
        "predicted_incoming_edges": np.asarray(prediction.getnnz(axis=0)).ravel(),
        "prediction_threshold": data["threshold"],
    })

def shared_spatial_limits(datasets: list[dict]) -> tuple[float, float]:
    """Use one square data extent so all six axes have identical dimensions."""
    lower = min(float(data["coords"].min()) for data in datasets) - 0.7
    upper = max(float(data["coords"].max()) for data in datasets) + 0.7
    return lower, upper

def _draw_spatial(axis: plt.Axes, coords: np.ndarray, values: np.ndarray,
                  norm: Normalize, cmap: mpl.colors.Colormap,
                  limits: tuple[float, float]) -> None:
    axis.set_xlim(*limits)
    axis.set_ylim(*limits)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_facecolor("#FBFCFD")
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#8492A1")
        spine.set_linewidth(0.55)
    axis.scatter(coords[:, 0], coords[:, 1], s=8.0, marker="s",
                 c="#E4E9EE", edgecolors="none", zorder=1)
    positive = values > 0
    if positive.any():
        axis.scatter(coords[positive, 0], coords[positive, 1], s=9.0,
                     marker="s", c=values[positive], cmap=cmap, norm=norm,
                     edgecolors="none", zorder=2)

def render(datasets: list[dict], cells: pd.DataFrame, dpi: int,
           spatial_limits: tuple[float, float]) -> plt.Figure:
    truth_max = max(1, int(cells.truth_incoming_edges.max()))
    raw_pred_max = max(1.0, float(cells.predicted_incoming_strength.max()))
    pred_ticks = mpl.ticker.MaxNLocator(
        nbins=2, steps=[1, 2, 5, 10]
    ).tick_values(0, raw_pred_max)
    pred_max = float(pred_ticks[-1])
    truth_norm = Normalize(vmin=0, vmax=truth_max)
    prediction_norm = PowerNorm(gamma=0.62, vmin=0, vmax=pred_max)

    with mpl.rc_context(STYLE):
        figure, axes = plt.subplots(2, 3, figsize=FIGURE_SIZE, dpi=dpi)
        figure.subplots_adjust(left=0.075, right=0.880, top=0.865,
                               bottom=0.100, wspace=0.130, hspace=0.190)
        for column, data in enumerate(datasets):
            table = cells.loc[cells.dataset_id.eq(data["dataset_id"])]
            coords = data["coords"]
            _draw_spatial(
                axes[0, column], coords,
                table.truth_incoming_edges.to_numpy(dtype=float),
                truth_norm, TRUTH_CMAP, spatial_limits,
            )
            _draw_spatial(
                axes[1, column], coords,
                table.predicted_incoming_strength.to_numpy(dtype=float),
                prediction_norm, PRED_CMAP, spatial_limits,
            )
            position = axes[0, column].get_position()
            center = (position.x0 + position.x1) / 2
            figure.text(
                center, 0.945,
                f"{SCENARIO_LABELS[data['scenario']]}  ·  {data['pair_id']}",
                ha="center", va="center", fontsize=8.2,
                fontweight="bold", color="#25364B",
            )

        for row, label in enumerate(("Ground truth", "CellSTIC")):
            position = axes[row, 0].get_position()
            figure.text(
                0.027, (position.y0 + position.y1) / 2, label,
                rotation=90, ha="center", va="center", fontsize=7.8,
                fontweight="bold", color="#25364B",
            )

        for row, norm, cmap, ticks, label in (
            (0, truth_norm, TRUTH_CMAP,
             sorted(set([0, truth_max // 2, truth_max])),
             "True incoming edges / cell"),
            (1, prediction_norm, PRED_CMAP, pred_ticks,
             "Predicted incoming strength / cell"),
        ):
            position = axes[row, -1].get_position()
            colorbar_axis = figure.add_axes((0.905, position.y0, 0.014, position.height))
            colorbar = figure.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=colorbar_axis, orientation="vertical", ticks=ticks,
            )
            colorbar.ax.tick_params(length=2, width=0.5, pad=2, labelsize=6.0)
            colorbar.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.0f"))
            colorbar.outline.set_edgecolor("#8492A1")
            colorbar.outline.set_linewidth(0.5)
            figure.text(
                0.975, (position.y0 + position.y1) / 2, label,
                rotation=90, ha="center", va="center",
                fontsize=6.4, color="#38485B",
            )
        return figure

from .result_io import load_result, lr_matrices, clean_matrix
from .figure_export import save_show

def ccc_maps(run_root, pair_ids, output, *, seed=11, threshold=.5):
    datasets = []
    for scenario, pair_id in pair_ids.items():
        result = load_result(Path(run_root) / f'{scenario}_{seed}/out/out.h5ad', truth=True)
        row = result.uns['lr_candidates'].set_index('pair_id').loc[pair_id]
        pair = f'{row.ligand}:{row.receptor}'
        import json
        truth_key = json.loads(str(result.uns['ground_truth_obsp_keys_json']))[pair_id]
        truth = clean_matrix(result.obsp[truth_key], remove_self=True)
        truth.data[:] = 1
        prediction = clean_matrix(lr_matrices(result)[pair], threshold=threshold, remove_self=True)
        datasets.append(dict(scenario=scenario, seed=seed, dataset_id=result.uns['dataset_id'],
            coords=np.asarray(result.obsm['spatial']), cell_ids=result.obs_names.to_numpy(),
            pair_id=pair_id, name=pair, truth=truth, prediction=prediction, threshold=threshold))
    cells = pd.concat([cell_source_table(data) for data in datasets], ignore_index=True)
    fig = render(datasets, cells, 600, shared_spatial_limits(datasets))
    save_show(fig, output, 'ccc_truth_prediction', exact_size=True)
    cells.to_csv(Path(output) / 'ccc_map_source.csv', index=False)
    return cells
