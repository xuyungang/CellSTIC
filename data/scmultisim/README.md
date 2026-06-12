# scMultiSim (re1–re8)

## Data Source

Li, H., Zhang, Z., Squires, M. et al. scMultiSim: simulation of single-cell multi-omics and spatial data guided by gene regulatory networks and cell–cell interactions. *Nat Methods* 22, 982–993 (2025). https://doi.org/10.1038/s41592-025-02651-0

Simulated spatial multi-omics replicates **re1–re8** (400 cells each).

## Statistics

| Item | Description |
|------|-------------|
| Species | Simulate |
| Modality | RNA, ATAC |
| # Cells | 400 |
| RNA (genes) | 120 |
| ATAC (peaks) | 360 |

## Raw data layout

Use **`data/scmultisim/`** as the dataset root: one folder per replicate **`re1` … `re8`**.

Each replicate stores pre-packaged AnnData under **`raw/`**:

- `raw/rna.h5ad` — RNA counts, spatial coordinates, cell types, and CCC metadata in `uns`
- `raw/atac.h5ad` — ATAC counts and spatial coordinates

Ground truth and LR metadata live in **`rna.h5ad`** (not separate CSV/HDF5 files):

| Location | Content |
|----------|---------|
| `rna.uns["ccc_gt"]` | Ground-truth CCC labels |
| `rna.uns["ligand_receptor_map"]` | LR pair → channel names |
| `rna.uns["pair_type_constraints"]` | Allowed sender/receiver cell-type pairs per LR |
| `rna.uns["lr_table"]` | LR table used in simulation |
| `rna.obsm["spatial"]` | Spot coordinates |
| `rna.obs["cell_type"]` | Cell-type labels |

Preprocessing (PCA / LSI, spatial distance matrix) is performed in **`notebook/scmultisim.ipynb`** (Step 3); there is no dedicated loader module for this dataset.

## Replicate layout

Running the tutorial creates model checkpoints, results, and figures under the chosen replicate folder. Set `RE_NUM` in the notebook to switch replicates.

```
data/scmultisim/
├── re1/
│   ├── raw/
│   │   ├── rna.h5ad
│   │   └── atac.h5ad
│   ├── model/              # cellstic_model.pth (after training)
│   ├── result/             # cellstic_result.h5ad
│   └── analysis/           # figures and metrics CSVs
├── re2/
│   └── raw/                # rna.h5ad, atac.h5ad only (run notebook to generate outputs)
└── re8/
    └── ...
```

## Tutorial

See **`notebook/scmultisim.ipynb`**. Start Jupyter with the repository root as the working directory.
