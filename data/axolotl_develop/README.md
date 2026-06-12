# Axolotl Telencephalon — Development

## Data Source

**CNGB SciRAID** (STDS0000056):

- Development: https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000056/stomics/Development.h5ad

In this repository, development batches are stored in a **single** AnnData at `raw/rna.h5ad` (not separate per-stage folders).

## Statistics

Bundled `raw/rna.h5ad`: **36,198 cells × 12,704 genes**, telencephalon only.

| Stage folder | `obs['Batch']` | # Cells |
|---|---|---:|
| `develop_44` | Stage44_telencephalon_rep2_FP200000239BL_E4 | 1,477 |
| `develop_54` | Stage54_telencephalon_rep2_DP8400015649BRD6_2 | 2,929 |
| `develop_57` | Stage57_telencephalon_rep2_DP8400015649BRD5_1 | 4,410 |
| `develop_Juvenile` | Injury_control_FP200000239BL_E3 | 11,698 |
| `develop_Adult` | Batch1_Adult_telencephalon_rep2_DP8400015234BLA3_1 | 8,243 |
| `develop_Metamorphosed` | Meta_telencephalon_rep1_DP8400015234BLB2_1 | 7,441 |

Key fields in the bundled file:

- **`obs['Batch']`** — sample batch ID (mapped to stage folders in the notebook)
- **`obs['Annotation']`** — cell-type annotation (33 types)
- **`obsm['spatial']`** — spot coordinates

## Raw data layout

Use **`data/axolotl_develop/`** as the dataset root.

- **`raw/rna.h5ad`** — combined development Stereo-seq RNA

## Processing pipeline

See **`notebook/axolotl_develop.ipynb`**. The notebook (shared structure with `axolotl_regene.ipynb`):

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
data/axolotl_develop/
├── raw/
│   └── rna.h5ad
├── model/<stage>/              # e.g. develop_44, develop_Adult
├── result/<stage>/             # cellstic_result.h5ad per stage
└── time_series/                # SVG figures from TimeSequenceAnalysis
```

After a full run, `result/` contains **6** stage folders.

## Tutorial

See **`notebook/axolotl_develop.ipynb`**. Start Jupyter with the repository root as the working directory.
