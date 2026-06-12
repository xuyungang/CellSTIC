# Mouse Embryo

## Data Source

**Publication.** Chen, A., Liao, S., Cheng, M., Ma, K., Wu, L., Lai, Y., ... & Wang, J. (2022). Spatiotemporal transcriptomic atlas of mouse organogenesis using DNA nanoball-patterned arrays. *Cell*, 185(10), 1777-1792.

**CNGB SciRAID** (STDS0000058). Stage-wise downloads (E9.5–E16.5):

- E9.5: https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/E9.5_E1S1.MOSTA.h5ad
- E10.5: https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/E10.5_E1S1.MOSTA.h5ad
- E11.5: https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/E11.5_E1S1.MOSTA.h5ad
- E12.5: https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/E12.5_E1S1.MOSTA.h5ad
- E13.5: https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/E13.5_E1S1.MOSTA.h5ad
- E14.5: https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/E14.5_E1S1.MOSTA.h5ad
- E15.5: https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/E15.5_E1S1.MOSTA.h5ad
- E16.5: https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/E16.5_E1S1.MOSTA.h5ad

In this repository, selected stages and organ domains are merged into a **single** AnnData at `raw/rna.h5ad` (not separate per-stage folders).

## Statistics

Bundled `raw/rna.h5ad`: **118,265 cells × 29,678 genes**.

| `obs['stage']` | `obs['annotation']` | # Cells |
|---|---|---:|
| 9.5 | Brain | 1,518 |
| 9.5 | Liver | 98 |
| 10.5 | Brain | 2,816 |
| 10.5 | Liver | 195 |
| 11.5 | Brain | 5,794 |
| 11.5 | Liver | 754 |
| 12.5 | Brain | 11,525 |
| 12.5 | Liver | 1,429 |
| 13.5 | Brain | 12,707 |
| 13.5 | Liver | 2,750 |
| 14.5 | Brain | 17,715 |
| 14.5 | Liver | 6,015 |
| 15.5 | Brain | 17,071 |
| 15.5 | Liver | 6,337 |
| 16.5 | Brain | 17,374 |
| 16.5 | Liver | 14,167 |

Key fields in the bundled file:

- **`obs['stage']`** — embryonic stage (`9.5` … `16.5`)
- **`obs['annotation']`** — organ domain (`Brain`, `Liver`)
- **`obsm['spatial']`** — spot coordinates
- **`layers['count']` / `layers['counts']`** — raw count matrices
- **`X`** — sparse expression matrix (CSR, float32)

## Raw data layout

Use **`data/mouse_embryo/`** as the dataset root.

- **`raw/rna.h5ad`** — combined Stereo-seq RNA for all stages and organ domains

## Processing pipeline

See **`notebook/mouse_embryo.ipynb`**. The notebook:

1. Loads `raw/rna.h5ad` once
2. Loops over every `obs['stage']` × `obs['annotation']` pair
3. Preprocesses each subset: QC, HVG (3000), normalize (1e4), log1p, PCA (500), spatial distances
4. Annotates cell types with CellTypist (`mouse_brain` for Brain, `mouse_liver` for Liver)
5. Runs `run_cellstic` per pair and writes `result/<stage>_<organ>/cellstic_result.h5ad`
6. Runs `TimeSequenceAnalysis` via `result_root`, loading all per-run result files (figures under `time_series/`)

Ligand–receptor pairs used in the notebook:

| Ligand | Receptor |
|---|---|
| F2 | F2r |
| Lpar3 | Adgre5 |
| Nts | Sort1 |
| Plg | Pard3 |
| Thbs4 | Cd36 |

## Result layout

```
data/mouse_embryo/
├── raw/
│   └── rna.h5ad
├── model/<stage>_<organ>/         # e.g. 9.5_Brain, 10.5_Liver
├── result/<stage>_<organ>/        # cellstic_result.h5ad (CCI + spatial + cell_type)
├── analysis/                     # optional downstream analysis outputs
└── time_series/                  # SVG figures from TimeSequenceAnalysis
```

After a full run, `result/` contains **16** folders (8 stages × 2 organs).

## Tutorial

See **`notebook/mouse_embryo.ipynb`**. Start Jupyter with the repository root as the working directory.
