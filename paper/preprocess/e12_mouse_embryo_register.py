#!/usr/bin/env python3
"""Register the five stitched PLA images to the corresponding Visium HD H&E.

The saved affine maps original PLA pixel centres to a 23,216 x 13,108
reference frame derived from the full-resolution Space Ranger H&E image.
Registration uses anatomy (the tissue outline), never PLA signal or model
predictions. Adjacent sections can differ locally, so the automated result
requires anatomical review before it is used as 16-um ground truth.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
from pathlib import Path

import cv2
import numpy as np
import tifffile
from scipy.optimize import minimize


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "e12_mouse_embryo"
REFERENCE_SIZE = (23216, 13108)  # width, height; reference used in the paper
PREVIEW_WIDTH = 1200
PAIRS = {
    "A1": {
        "accession": "GSM8829020",
        "images": {
            "Dll1-Notch1": "A1_fused_DLL-NOTCH_ch02.png",
            "Fgf8-Fgfr1": "A1_fused_Fgf8-Fgfr1_CH2.png",
        },
    },
    "D1": {
        "accession": "GSM8829021",
        "images": {
            "Efna3-Epha5": "D1_fused_Efna3-Epha5_brain_CH2.png",
            "Igf2-Igf2r": "D1_fused_Igf-Igf2r_CH2.png",
            "Spp1-Cd44": "D1_fused_Spp1-Cd44_CH2.png",
        },
    },
}


def _resize_matrix(source_size: tuple[int, int], target_size: tuple[int, int]) -> np.ndarray:
    """Map zero-based source pixel centres through cv2.resize geometry."""
    sx = target_size[0] / source_size[0]
    sy = target_size[1] / source_size[1]
    return np.array(
        [[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]],
        dtype=np.float64,
    )


def _largest_component(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = cv2.morphologyEx(
        mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count < 2:
        raise ValueError("No tissue component detected")
    index = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == index).astype(np.uint8), stats[index]


def _read_hires(dataset_dir: Path, sample: str) -> np.ndarray:
    accession = PAIRS[sample]["accession"]
    path = (
        dataset_dir
        / "raw"
        / "visium_hd"
        / f"{accession}_{sample}-tissue_hires_image.png.gz"
    )
    with gzip.open(path, "rb") as stream:
        image = cv2.imdecode(np.frombuffer(stream.read(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    return image


def _fullres_size(dataset_dir: Path, sample: str) -> tuple[int, int]:
    """Read the JPEG dimensions without expanding its 2.4-billion-pixel raster."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    accession = PAIRS[sample]["accession"]
    path = (
        dataset_dir
        / "raw"
        / "visium_hd"
        / f"{accession}_{sample}-tissue_fullres_image.jpg.gz"
    )
    with gzip.open(path, "rb") as stream:
        with Image.open(stream) as image:
            return image.size


def _he_preview(hires: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height = round(hires.shape[0] * PREVIEW_WIDTH / hires.shape[1])
    preview = cv2.resize(hires, (PREVIEW_WIDTH, height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
    saturation = cv2.cvtColor(preview, cv2.COLOR_BGR2HSV)[:, :, 1]
    tissue, box = _largest_component((saturation > 17) & (gray < 242))
    return preview, tissue, box


def _pla_preview(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    height = round(image.shape[0] * PREVIEW_WIDTH / image.shape[1])
    preview = cv2.resize(image, (PREVIEW_WIDTH, height), interpolation=cv2.INTER_AREA)
    border = np.concatenate(
        (preview[:40].ravel(), preview[-40:].ravel(), preview[:, :40].ravel(), preview[:, -40:].ravel())
    )
    background = float(np.median(border))
    p90 = float(np.percentile(preview, 90))
    threshold = background + 0.18 * (p90 - background)
    tissue, box = _largest_component(preview > threshold)
    return preview, tissue, box, background, p90, threshold


def _affine_parameters(
    values: np.ndarray, moving_center: np.ndarray, fixed_center: np.ndarray
) -> np.ndarray:
    sx, sy = np.exp(values[:2])
    angle = values[2]
    cosine, sine = np.cos(angle), np.sin(angle)
    linear = np.array([[sx * cosine, -sy * sine], [sx * sine, sy * cosine]])
    translation = fixed_center + values[3:] - linear @ moving_center
    return np.column_stack((linear, translation)).astype(np.float32)


def _fit_outline(
    moving: np.ndarray, moving_box: np.ndarray, fixed: np.ndarray, fixed_box: np.ndarray
) -> tuple[np.ndarray, float, float]:
    fixed_center = np.array(
        [fixed_box[0] + fixed_box[2] / 2, fixed_box[1] + fixed_box[3] / 2]
    )
    moving_center = np.array(
        [moving_box[0] + moving_box[2] / 2, moving_box[1] + moving_box[3] / 2]
    )
    sx = fixed_box[2] / moving_box[2]
    sy = fixed_box[3] / moving_box[3]
    initial = np.array([np.log(sx), np.log(sy), 0, 0, 0], dtype=float)
    fixed_bool = fixed.astype(bool)

    def iou(values: np.ndarray) -> float:
        aligned = cv2.warpAffine(
            moving,
            _affine_parameters(values, moving_center, fixed_center),
            (fixed.shape[1], fixed.shape[0]),
            flags=cv2.INTER_NEAREST,
        ).astype(bool)
        return float(np.logical_and(fixed_bool, aligned).sum() / np.logical_or(fixed_bool, aligned).sum())

    def objective(values: np.ndarray) -> float:
        return 1 - iou(values) + 0.02 * abs(values[0] - values[1])

    bounds = [
        (np.log(sx * 0.75), np.log(sx * 1.25)),
        (np.log(sy * 0.75), np.log(sy * 1.25)),
        (-0.5, 0.5),
        (-80, 80),
        (-80, 80),
    ]
    initial_iou = iou(initial)
    fit = minimize(
        objective,
        initial,
        method="Powell",
        bounds=bounds,
        options={"maxiter": 70, "ftol": 1e-5},
    )
    return _affine_parameters(fit.x, moving_center, fixed_center), initial_iou, iou(fit.x)


def _preview_overlays(
    he: np.ndarray,
    he_tissue: np.ndarray,
    pla_preview: np.ndarray,
    pla_tissue: np.ndarray,
    affine: np.ndarray,
    background: float,
    p90: float,
    output_dir: Path,
) -> None:
    size = (he.shape[1], he.shape[0])
    aligned = cv2.warpAffine(pla_preview, affine, size, flags=cv2.INTER_LINEAR)
    aligned_mask = cv2.warpAffine(pla_tissue, affine, size, flags=cv2.INTER_NEAREST)
    he_gray = cv2.cvtColor(he, cv2.COLOR_BGR2GRAY).astype(np.float32)
    he_density = np.clip((245 - he_gray) * 1.5, 0, 255).astype(np.uint8)
    pla_density = np.clip(
        (aligned.astype(np.float32) - background) * 255 / max(p90 - background, 1),
        0,
        255,
    ).astype(np.uint8)
    anatomy = np.zeros_like(he)
    anatomy[:, :, 0] = he_density
    anatomy[:, :, 2] = he_density
    anatomy[:, :, 1] = pla_density
    cv2.imwrite(str(output_dir / "anatomy_overlay.png"), anatomy)

    outline = he.copy()
    fixed_contours, _ = cv2.findContours(he_tissue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    moving_contours, _ = cv2.findContours(aligned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(outline, fixed_contours, -1, (170, 40, 210), 2)
    cv2.drawContours(outline, moving_contours, -1, (20, 175, 20), 2)
    cv2.imwrite(str(output_dir / "outline_overlay.png"), outline)


def register_one(
    dataset_dir: Path,
    output_root: Path,
    sample: str,
    pair: str,
    he_hires: np.ndarray,
    he_preview: np.ndarray,
    he_tissue: np.ndarray,
    he_box: np.ndarray,
    fullres_size: tuple[int, int],
    write_image: bool,
) -> None:
    filename = PAIRS[sample]["images"][pair]
    source = dataset_dir / "raw" / "pla" / filename
    output_dir = output_root / sample / pair.replace("-", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    pla = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if pla is None or pla.ndim != 2 or pla.dtype != np.uint16:
        raise ValueError(f"Expected a 16-bit grayscale PLA image: {source}")
    original_size = (pla.shape[1], pla.shape[0])
    preview, tissue, _, background, p90, threshold = _pla_preview(pla)

    # All five images were visually checked against their sample H&E images.
    # A horizontal reflection is required; rotation is not. This agreement
    # across each sample also resolves near-ties in outline-only scores.
    tissue = np.fliplr(tissue).copy()
    moving_preview = np.fliplr(preview).copy()
    tissue, box = _largest_component(tissue)
    affine, initial_iou, fitted_iou = _fit_outline(tissue, box, he_tissue, he_box)
    _preview_overlays(
        he_preview, he_tissue, moving_preview, tissue, affine, background, p90, output_dir
    )

    # Original PLA -> PLA preview -> mirrored preview -> H&E preview ->
    # original H&E -> paper-sized H&E reference. All are zero-based pixel-centre
    # coordinates. The matrix therefore also maps any raw PLA pixel directly to
    # the reference image used for the exported TIFF.
    source_to_preview = _resize_matrix(original_size, (preview.shape[1], preview.shape[0]))
    reflect = np.array(
        [[-1, 0, preview.shape[1] - 1], [0, 1, 0], [0, 0, 1]], dtype=np.float64
    )
    preview_affine = np.vstack((affine, [0, 0, 1])).astype(np.float64)
    fullres_to_preview = _resize_matrix(fullres_size, (he_preview.shape[1], he_preview.shape[0]))
    source_to_he_fullres = np.linalg.inv(fullres_to_preview) @ preview_affine @ reflect @ source_to_preview
    fullres_to_reference = _resize_matrix(fullres_size, REFERENCE_SIZE)
    source_to_reference = fullres_to_reference @ source_to_he_fullres
    fullres_to_hires = _resize_matrix(
        fullres_size, (he_hires.shape[1], he_hires.shape[0])
    )
    source_to_hires = fullres_to_hires @ source_to_he_fullres

    # A larger overlay exposes local anatomical errors that are invisible in
    # the 1200-pixel outline preview (roughly one preview pixel per 16-um bin).
    aligned_hires = cv2.warpAffine(
        pla,
        source_to_hires[:2],
        (he_hires.shape[1], he_hires.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderValue=0,
    )
    he_gray = cv2.cvtColor(he_hires, cv2.COLOR_BGR2GRAY).astype(np.float32)
    he_density = np.clip((245 - he_gray) * 1.5, 0, 255).astype(np.uint8)
    pla_density = np.clip(
        (aligned_hires.astype(np.float32) - background) * 255 / max(p90 - background, 1),
        0,
        255,
    ).astype(np.uint8)
    hires_overlay = np.empty_like(he_hires)
    hires_overlay[:, :, 0] = he_density
    hires_overlay[:, :, 1] = pla_density
    hires_overlay[:, :, 2] = he_density
    cv2.imwrite(
        str(output_dir / "anatomy_overlay_hires.jpg"),
        hires_overlay,
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )
    del aligned_hires, he_gray, he_density, pla_density, hires_overlay

    metadata = {
        "sample": sample,
        "lr_pair": pair,
        "source_image": str(source.relative_to(dataset_dir)),
        "reference": "Visium HD H&E, rescaled from full-resolution Space Ranger image",
        "source_size_xy": original_size,
        "he_fullres_size_xy": fullres_size,
        "he_hires_size_xy": (he_hires.shape[1], he_hires.shape[0]),
        "reference_size_xy": REFERENCE_SIZE,
        "orientation": "horizontal_reflection",
        "pla_background_preview": background,
        "pla_tissue_threshold_preview": threshold,
        "outline_iou_before_optimization": initial_iou,
        "outline_iou_after_optimization": fitted_iou,
        "pla_to_he_fullres": source_to_he_fullres.tolist(),
        "pla_to_he_hires": source_to_hires.tolist(),
        "pla_to_reference": source_to_reference.tolist(),
        "he_fullres_to_reference": fullres_to_reference.tolist(),
        "qc_status": "automatic_anatomical_alignment; 16um_accuracy_unverified",
    }
    (output_dir / "registration.json").write_text(json.dumps(metadata, indent=2) + "\n")

    if write_image:
        aligned = cv2.warpAffine(
            pla,
            source_to_reference[:2].astype(np.float64),
            REFERENCE_SIZE,
            flags=cv2.INTER_LINEAR,
            borderValue=0,
        )
        valid = cv2.warpAffine(
            np.ones(pla.shape, dtype=np.uint8),
            source_to_reference[:2].astype(np.float64),
            REFERENCE_SIZE,
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )
        tifffile.imwrite(output_dir / "registered_pla.tif", aligned, compression="deflate")
        tifffile.imwrite(output_dir / "valid_mask.tif", valid, compression="deflate")
        del aligned, valid
    print(f"{sample} {pair}: outline IoU {initial_iou:.3f} -> {fitted_iou:.3f}", flush=True)
    del pla
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sample", choices=("A1", "D1"), nargs="+")
    parser.add_argument("--pair", help="Process only one ligand-receptor pair")
    parser.add_argument("--preview-only", action="store_true", help="Skip full-resolution TIFF export")
    args = parser.parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_root = (args.output_dir or dataset_dir / "registered").resolve()
    for sample in args.sample or ("A1", "D1"):
        he_hires = _read_hires(dataset_dir, sample)
        he_preview, he_tissue, he_box = _he_preview(he_hires)
        fullres_size = _fullres_size(dataset_dir, sample)
        for pair in PAIRS[sample]["images"]:
            if args.pair and args.pair != pair:
                continue
            register_one(
                dataset_dir,
                output_root,
                sample,
                pair,
                he_hires,
                he_preview,
                he_tissue,
                he_box,
                fullres_size,
                not args.preview_only,
            )
        del he_hires, he_preview, he_tissue
        gc.collect()


if __name__ == "__main__":
    main()
