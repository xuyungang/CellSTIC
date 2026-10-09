"""Visium HD tissue coverage, receiver scores and native PLA images."""
from __future__ import annotations


from dataclasses import dataclass

from pathlib import Path

import anndata as ad

import h5py

import matplotlib as mpl

import matplotlib.pyplot as plt

from matplotlib.colors import Normalize


import numpy as np

import pandas as pd

from PIL import Image

from scipy import ndimage


FIGURE_MM = (180.0, 130.0)

SAMPLES = ("A1", "D1")

PLA_PAIRS = (
    ("A1", "Dll1_Notch1", "Dll1–Notch1"),
    ("A1", "Fgf8_Fgfr1", "Fgf8–Fgfr1"),
    ("D1", "Efna3_Epha5", "Efna3–Epha5"),
    ("D1", "Igf2_Igf2r", "Igf2–Igf2r"),
    ("D1", "Spp1_Cd44", "Spp1–Cd44"),
)

QC_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.2,
    "axes.titlesize": 6.8,
    "axes.linewidth": 0.5,
    "axes.labelsize": 6.0,
    "xtick.labelsize": 5.3,
    "ytick.labelsize": 6.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
}

@dataclass
class Sample:
    name: str
    image: np.ndarray
    bins: pd.DataFrame
    n_genes: int
    scale: float
    microns_per_pixel: float
    tissue_bounds: tuple[float, float, float, float]
    view_limits: tuple[float, float, float, float] | None = None

def read_detected_genes(path: Path, n_bins: int) -> np.ndarray:
    """Count positive genes per bin from raw CSR values in bounded row blocks."""
    genes = np.empty(n_bins, dtype=np.int32)
    with h5py.File(path) as handle:
        matrix = handle["X"]
        if matrix.attrs.get("encoding-type") != "csr_matrix":
            raise ValueError(f"Expected a CSR count matrix: {path}")
        indptr = matrix["indptr"][:]
        if len(indptr) != n_bins + 1 or np.any(np.diff(indptr) < 0):
            raise ValueError(f"Invalid CSR row pointers: {path}")
        for start in range(0, n_bins, 2048):
            stop = min(start + 2048, n_bins)
            offsets = indptr[start:stop + 1] - indptr[start]
            values = matrix["data"][indptr[start]:indptr[stop]]
            if not np.issubdtype(values.dtype, np.integer) or np.any(values < 0):
                raise ValueError(f"Expected nonnegative integer RNA counts: {path}")
            if np.any(values == 0):
                cumulative = np.empty(len(values) + 1, dtype=np.int64)
                cumulative[0] = 0
                np.cumsum(values > 0, dtype=np.int64, out=cumulative[1:])
                genes[start:stop] = np.diff(cumulative[offsets])
            else:
                genes[start:stop] = np.diff(offsets)
    return genes

def tissue_bounds(image: np.ndarray, scale: float) -> tuple[float, float, float, float]:
    """Locate the main H&E tissue component; preserve its complete bounding box."""
    rgb = np.asarray(image, dtype=float)
    if rgb.max() > 1:
        rgb /= 255
    mask = ((rgb.max(axis=2) - rgb.min(axis=2)) > 0.10) & (rgb.mean(axis=2) < 0.90)
    mask = ndimage.binary_closing(mask, iterations=2)
    components, count = ndimage.label(mask)
    if count == 0:
        raise ValueError("No H&E tissue component found")
    areas = np.bincount(components.ravel())
    areas[0] = 0
    rows, columns = np.where(components == areas.argmax())
    return (columns.min() / scale, (columns.max() + 1) / scale,
            rows.min() / scale, (rows.max() + 1) / scale)

def load_sample(data_root: Path, name: str, *, include_gene_counts: bool = True) -> Sample:
    path = data_root / "h5ad" / f"{name}.h5ad"
    data = ad.read_h5ad(path, backed="r")
    try:
        if data.X.shape[1] != data.n_vars or "spatial" not in data.obsm:
            raise ValueError(f"Missing count or spatial data: {path}")
        image_record = next(iter(data.uns["spatial"].values()))
        image = np.asarray(image_record["images"]["hires"]).copy()
        scale = float(image_record["scalefactors"]["tissue_hires_scalef"])
        lowres_scale = float(image_record["scalefactors"]["tissue_lowres_scalef"])
        bounds = tissue_bounds(image_record["images"]["lowres"], lowres_scale)
        microns_per_pixel = float(image_record["scalefactors"]["microns_per_pixel"])
        coords = np.asarray(data.obsm["spatial"], dtype=float)
        barcodes = data.obs_names.astype(str).to_numpy()
        n_genes = data.n_vars
    finally:
        data.file.close()

    if not np.isfinite(coords).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid spatial coordinates: {path}")
    bins = pd.DataFrame({
        "sample": name,
        "barcode": barcodes,
        "x_fullres_px": coords[:, 0],
        "y_fullres_px": coords[:, 1],
    })
    if include_gene_counts:
        bins["detected_genes"] = read_detected_genes(path, len(barcodes))
    print(f"{name}: {len(bins):,} spatial bins", flush=True)
    return Sample(name, image, bins, n_genes, scale, microns_per_pixel, bounds)

def set_view_limits(samples: dict[str, Sample]) -> None:
    """Use square views with the same physical span for both samples."""
    physical_side = 1.14 * max(
        max(sample.tissue_bounds[1] - sample.tissue_bounds[0],
            sample.tissue_bounds[3] - sample.tissue_bounds[2]) * sample.microns_per_pixel
        for sample in samples.values()
    )
    for sample in samples.values():
        x0, x1, y0, y1 = sample.tissue_bounds
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half = physical_side / sample.microns_per_pixel / 2
        sample.view_limits = (cx - half, cx + half, cy - half, cy + half)
        sample.bins["in_display_crop"] = (
            sample.bins.x_fullres_px.between(cx - half, cx + half)
            & sample.bins.y_fullres_px.between(cy - half, cy + half)
        )

def add_axis(figure: plt.Figure, x: float, y: float, w: float, h: float) -> plt.Axes:
    width, height = FIGURE_MM
    return figure.add_axes((x / width, y / height, w / width, h / height))


def frame_axis(axis: plt.Axes) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.45)
        spine.set_color("#9BA8AD")

def scale_bar(axis: plt.Axes, sample: Sample, color: str = "#223640") -> None:
    x0, x1, _, _ = sample.view_limits
    fraction = 1000 / ((x1 - x0) * sample.microns_per_pixel)
    axis.plot((0.92 - fraction, 0.92), (0.065, 0.065), transform=axis.transAxes,
              color=color, linewidth=1.2, solid_capstyle="butt")
    text = axis.text(0.92 - fraction / 2, 0.10, "1 mm", transform=axis.transAxes,
                     color=color, ha="center", va="bottom", fontsize=4.8)
    if color != "white":
        text.set_bbox({"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.3})

def view_axis(axis: plt.Axes, sample: Sample) -> None:
    x0, x1, y0, y1 = sample.view_limits
    axis.set_xlim(x0, x1)
    axis.set_ylim(y1, y0)
    axis.set_aspect("equal", adjustable="box")
    frame_axis(axis)

def draw_he(figure: plt.Figure, samples: dict[str, Sample]) -> None:
    for index, name in enumerate(SAMPLES):
        sample = samples[name]
        x = 9 + index * 40
        axis = add_axis(figure, x, 83, 34, 34)
        x0, x1, y0, y1 = sample.view_limits
        left, top = np.floor(np.array([x0, y0]) * sample.scale).astype(int)
        right, bottom = np.ceil(np.array([x1, y1]) * sample.scale).astype(int)
        if min(left, top) < 0 or right > sample.image.shape[1] or bottom > sample.image.shape[0]:
            raise ValueError(f"Tissue crop falls outside the H&E image for {name}")
        axis.imshow(sample.image[top:bottom, left:right], interpolation="bilinear",
                    extent=(left / sample.scale, right / sample.scale,
                            bottom / sample.scale, top / sample.scale))
        view_axis(axis, sample)
        scale_bar(axis, sample)
        axis.set_title(name, loc="left", pad=3, color="#223640", fontweight="bold")
        axis.text(0.5, -0.055, f"{len(sample.bins):,} bins", transform=axis.transAxes,
                  ha="center", va="top", fontsize=6, color="#526873")

def draw_genes(figure: plt.Figure, samples: dict[str, Sample]) -> None:
    all_counts = np.concatenate([samples[name].bins.detected_genes.to_numpy() for name in SAMPLES])
    vmax = float(np.percentile(all_counts, 99))
    norm = Normalize(vmin=0, vmax=vmax)
    cmap = mpl.cm.get_cmap("viridis")
    for index, name in enumerate(SAMPLES):
        sample = samples[name]
        x = 99 + index * 40
        axis = add_axis(figure, x, 83, 34, 34)
        axis.scatter(
            sample.bins.x_fullres_px, sample.bins.y_fullres_px,
            c=sample.bins.detected_genes, cmap=cmap, norm=norm,
            s=0.09, marker="s", linewidths=0, rasterized=True,
        )
        view_axis(axis, sample)
        scale_bar(axis, sample)
        axis.set_title(name, loc="left", pad=3, color="#223640", fontweight="bold")
    # One shared scale makes the two samples directly comparable.
    color_axis = add_axis(figure, 123, 79, 48, 1.8)
    colorbar = figure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                               cax=color_axis, orientation="horizontal")
    colorbar.set_ticks((0, 1000, 2000, vmax))
    colorbar.set_ticklabels(("0", "1,000", "2,000", f"{vmax:,.0f}"))
    colorbar.ax.tick_params(labelsize=5, length=1.6, pad=1)
    colorbar.outline.set_linewidth(0.4)
    figure.text(120 / FIGURE_MM[0], 79.9 / FIGURE_MM[1], "Genes / bin",
                ha="right", va="center", fontsize=5.5, color="#526873")


from matplotlib.colors import LinearSegmentedColormap


FIGURE_SIZE = (3.55, 2.62)

PLA_COLOR = "#477AA8"

PREDICTION_COLOR = "#D87352"

CELLSTIC_COLOR = "#2E5C8A"

TEXT_COLOR = "#263A46"

PREDICTION_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 4.8,
    "axes.titlesize": 4.6,
    "axes.labelsize": 4.2,
    "xtick.labelsize": 3.4,
    "ytick.labelsize": 3.5,
    "axes.linewidth": 0.45,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
}

def rank_positive(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.zeros(values.shape, dtype=float)
    keep = np.isfinite(values) & (values > 0)
    if keep.any():
        result[keep] = pd.Series(values[keep]).rank(
            method="average", pct=True
        ).to_numpy()
    return result

def signal_cmap(color: str) -> LinearSegmentedColormap:
    if color == PLA_COLOR:
        colors = ["#E9EDF0", "#D8E3EC", "#8EACC6", color]
    else:
        colors = ["#E9EDF0", "#F0DDD5", "#E3A68B", color]
    return LinearSegmentedColormap.from_list("signal_rank", colors, N=256)

def pair_extent(frame: pd.DataFrame) -> tuple[float, float, float, float]:
    x0, x1 = np.quantile(frame["x"], (0.002, 0.998))
    y0, y1 = np.quantile(frame["y"], (0.002, 0.998))
    xpad = 0.015 * (x1 - x0)
    ypad = 0.015 * (y1 - y0)
    return x0 - xpad, x1 + xpad, y0 - ypad, y1 + ypad

def draw_spatial_map(
    ax: plt.Axes,
    frame: pd.DataFrame,
    value: str,
    extent: tuple[float, float, float, float],
    color: str,
) -> None:
    ordered = frame.iloc[np.argsort(frame[value].to_numpy())]
    ax.scatter(
        ordered["x"], ordered["y"], c=ordered[value],
        cmap=signal_cmap(color), vmin=0, vmax=1,
        s=0.22, linewidths=0, rasterized=True,
    )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[3], extent[2])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#F7F9FA")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#A8B2B9")
        spine.set_linewidth(0.38)

from .dataset_analysis import receiver_scores
from .figure_export import save_show

def e12_histology(data_root, output):
    """Show the original H&E images in matched tissue views with 1-mm scale bars.

    Images and scale factors come from the assembled input metadata; RNA count
    matrices are not loaded or scanned for this image-only preview.
    """
    samples = {name: load_sample(Path(data_root), name, include_gene_counts=False)
               for name in SAMPLES}
    set_view_limits(samples)
    with mpl.rc_context(QC_STYLE):
        fig = plt.figure(figsize=(FIGURE_MM[0]/25.4, FIGURE_MM[1]/25.4))
        draw_he(fig, samples)
        save_show(fig, output, 'e12_histology')

def e12_quality(data_root, output):
    samples = {name: load_sample(Path(data_root), name) for name in SAMPLES}
    set_view_limits(samples)
    with mpl.rc_context(QC_STYLE):
        fig = plt.figure(figsize=(FIGURE_MM[0]/25.4, FIGURE_MM[1]/25.4))
        for x, title in [(2, 'H&E · E12.5 mouse embryo'), (92, 'Spatial RNA · 16 μm bins')]:
            fig.text((x+6)/FIGURE_MM[0], 128/FIGURE_MM[1], title, fontsize=7,
                     fontweight='bold', color='#223640', va='top')
        draw_he(fig, samples)
        draw_genes(fig, samples)
        save_show(fig, output, 'e12_anatomy_and_rna')
    pd.concat([s.bins for s in samples.values()]).to_csv(Path(output)/'e12_qc_source.csv', index=False)

def e12_predictions(results, output):
    frames = []
    with mpl.rc_context(PREDICTION_STYLE):
        fig = plt.figure(figsize=FIGURE_SIZE)
        for left, (sample, pair_id, label) in zip(np.linspace(.070,.798,5), PLA_PAIRS):
            result = results[sample]
            pair = pair_id.replace('_', ':')
            xy = np.asarray(result.obsm['spatial'])
            frame = pd.DataFrame({'x':xy[:,0], 'y':xy[:,1], 'cellstic_rank':receiver_scores(result,[pair],rank=True)[pair]})
            draw_spatial_map(fig.add_axes([left,.505,.160,.145]),frame,'cellstic_rank',pair_extent(frame),PREDICTION_COLOR)
            fig.text(left+.08,.70,label+'\n'+sample,ha='center',va='center',fontsize=3.55,fontweight='bold',color=TEXT_COLOR,linespacing=.92)
            frames.append(frame.assign(sample=sample,lr_pair=pair))
        save_show(fig,output,'e12_lr_predictions')
    pd.concat(frames).to_csv(Path(output)/'e12_prediction_source.csv',index=False)

def e12_raw_pla(data_root, results, output):
    with mpl.rc_context(QC_STYLE):
        fig = plt.figure(figsize=(180/25.4,42/25.4))
        for i,(sample,pair_id,label) in enumerate(PLA_PAIRS):
            ligand,receptor=pair_id.split('_')
            table=results[sample].uns['pla_pairs']
            relative=table.loc[table.ligand.eq(ligand)&table.receptor.eq(receptor),'image_file'].iloc[0]
            old_limit=Image.MAX_IMAGE_PIXELS
            try:
                Image.MAX_IMAGE_PIXELS=None
                with Image.open(Path(data_root)/str(relative)) as raw:
                    small=raw.convert('F');small.thumbnail((900,900));pixels=np.asarray(small)
            finally:
                Image.MAX_IMAGE_PIXELS=old_limit
            ax=fig.add_axes([(10+i*33.5)/180,6/42,28.5/180,28.5/42])
            ax.imshow(pixels,cmap='magma',vmin=0,vmax=np.percentile(pixels,99.5) or 1,interpolation='bilinear')
            frame_axis(ax)
            ax.set_title(f'{sample} · {label}',loc='left',pad=2.8,fontsize=5.7,color='#223640')
        save_show(fig,output,'e12_raw_pla')
