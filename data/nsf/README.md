# Spatial Multi-modal (NSF)

## Data Source

Long, Y., Ang, K.S., Sethi, R. et al. Deciphering spatial domains from spatial multi-omics with SpatialGlue. *Nat Methods* 21, 1658–1667 (2024). https://doi.org/10.1038/s41592-024-02316-4

## Statistics

| Item       | Description |
|------------|-------------|
| Species    | Simulate    |
| Modality   | RNA, ATAC, ADT |
| # Cells    | 1,296       |
| RNA (genes) | 1,000      |
| ATAC (peaks) | 1,000     |
| ADT (proteins) | 100     |

## Raw data layout

Use **`data/nsf/`** as the dataset root. Place three aligned AnnData objects and ground-truth labels under **`raw/`**:

- `raw/rna/adata_RNA.h5ad`
- `raw/adt/adata_ADT.h5ad`
- `raw/atac/adata_ATAC.h5ad`
- `raw/gt/spatial_factors.npy` (ground-truth spatial factors for evaluation)

Each AnnData should share the same cell barcodes (`obs_names`) and include `obsm['spatial']`. Preprocessing (HVG, LSI, PCA/CLR, spatial distance matrix) is done in the analysis notebook on every run; no `preprocess/` cache is required.

Ligand–receptor pairs are passed manually in the notebook configuration when running CellSTIC.

```
data/nsf/
├── raw/
│   ├── rna/adata_RNA.h5ad
│   ├── adt/adata_ADT.h5ad
│   ├── atac/adata_ATAC.h5ad
│   └── gt/spatial_factors.npy
├── model/            # trained CellSTIC checkpoints
├── result/           # cellstic_result.h5ad
└── analysis/         # figures
```
