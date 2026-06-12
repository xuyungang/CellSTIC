# Axolotl Telencephalon — Regeneration

## Data Source

**CNGB SciRAID** (STDS0000056):

- Regeneration: https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000056/stomics/Regeneration.h5ad

In this repository, regeneration time points are stored in a **single** AnnData at `raw/rna.h5ad` (not separate per-stage folders).

## Statistics

Bundled `raw/rna.h5ad`: **78,123 cells × 16,379 genes**, telencephalon only.

| Stage folder | `obs['Batch']` | # Cells |
|---|---|---:|
| `regeneration_2` | Injury_2DPI_rep1_SS200000147BL_D5 | 7,668 |
| `regeneration_5` | Injury_5DPI_rep1_SS200000147BL_D2 | 8,106 |
| `regeneration_10` | Injury_10DPI_rep1_SS200000147BL_B5 | 9,440 |
| `regeneration_15` | Injury_15DPI_rep4_FP200000266TR_E4 | 9,676 |
| `regeneration_20` | Injury_20DPI_rep2_SS200000147BL_B4 | 11,319 |
| `regeneration_30` | Injury_30DPI_rep2_FP200000264BL_A6 | 9,252 |
| `regeneration_60` | Injury_60DPI_rep3_FP200000264BL_A6 | 10,964 |
| `regeneration_control` | Injury_control_FP200000239BL_E3 | 11,698 |

Key fields in the bundled file:

- **`obs['Batch']`** — sample batch ID (mapped to stage folders in the notebook)
- **`obs['Annotation']`** — cell-type annotation (27 types)
- **`obsm['spatial']`** — spot coordinates

## Raw data layout

Use **`data/axolotl_regene/`** as the dataset root.

- **`raw/rna.h5ad`** — combined regeneration Stereo-seq RNA

## Processing pipeline

See **`notebook/axolotl_regene.ipynb`**. The notebook (shared structure with `axolotl_develop.ipynb`):

1. Loads `raw/rna.h5ad` once (deduplicate obs names, normalize pipe-formatted gene names, ensure spatial coords)
2. Subsets by `obs['Batch']` using `BATCH_TO_STAGE`
3. Preprocesses each stage: QC, HVG (3000), normalize (1e4), log1p, PCA (500), spatial distances; writes `obs['stage']` / `obs['organ']`
4. Runs `run_cellstic` per stage → `result/<stage>/cellstic_result.h5ad`
5. Runs `TimeSequenceAnalysis` via `result_root` (figures under `time_series/`)

Ligand–receptor pairs defined manually in the notebook:

| Ligand | Receptor |
|---|---|
| WNT7B | FZD5 |

## Result layout

```
data/axolotl_regene/
├── raw/
│   └── rna.h5ad
├── model/<stage>/              # e.g. regeneration_10, regeneration_control
├── result/<stage>/             # cellstic_result.h5ad per stage
└── time_series/                # SVG figures from TimeSequenceAnalysis
```

After a full run, `result/` contains **8** stage folders.

## Tutorial

See **`notebook/axolotl_regene.ipynb`**. Start Jupyter with the repository root as the working directory.
