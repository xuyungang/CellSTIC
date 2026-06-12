# Mouse Brain

## Data Source

Guo, P., Mao, L., Chen, Y. et al. Multiplexed spatial mapping of chromatin features, transcriptome and proteins in tissues. *Nat Methods* 22, 520–529 (2025). https://doi.org/10.1038/s41592-024-02576-0

## Statistics

| Item       | Description |
|------------|-------------|
| Species    | Mouse       |
| Modality   | RNA, ADT, ATAC, H3K27ac, H3K27me3 |
| # Cells    | 10,000      |
| RNA (genes) | 48,440    |
| ADT (proteins) | 131     |
| ATAC (peaks) | 21,091    |
| H3K27ac (peaks) | 16,835 |
| H3K27me3 (peaks) | 6,979  |

## Raw data layout

Use **`data/mouse_brain/`** as the dataset root. The tutorial [`notebook/mouse_brain.ipynb`](../../notebook/mouse_brain.ipynb) reads five aligned AnnData objects directly from **`raw/`**:

- `raw/rna.h5ad`
- `raw/adt.h5ad`
- `raw/atac.h5ad`
- `raw/h3k27ac.h5ad`
- `raw/h3k27me3.h5ad`

Each file should share the same cell barcodes (`obs_names`), include `obsm['spatial']`, and RNA should include `obs['cell_type']`. Preprocessing (HVG, PCA/LSI, spatial distance matrix) is done in the notebook; no cached `preprocess/` step is required.

Ligand–receptor pairs are passed manually in the notebook configuration (default: `Cd200–Cd200r4`, `Col6a3–Sdc4`, `Penk–Oprk1`).

```
data/mouse_brain/
├── raw/
│   ├── rna.h5ad
│   ├── adt.h5ad
│   ├── atac.h5ad
│   ├── h3k27ac.h5ad
│   └── h3k27me3.h5ad
├── model/            # trained CellSTIC checkpoints
├── result/           # cellstic_result.h5ad
└── analysis/         # figures (lr_spatial/, spatial_heatmaps/, lr_pair_stacked_bar.svg, …)
```
