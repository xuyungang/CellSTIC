# Spatial Multi-modal 

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

Use **`data/nsf/`** in this CellSTIC repository as the dataset root. `utils.loader.load_nsf` requires under **`raw/`**:

- `rna/adata_RNA.h5ad`
- `adt/adata_ADT.h5ad`
- `atac/adata_ATAC.h5ad`
- `gt/spatial_factors.npy` (ground-truth labels for evaluation)

```
data/nsf/
├── raw/
│   ├── rna/adata_RNA.h5ad
│   ├── adt/adata_ADT.h5ad
│   ├── atac/adata_ATAC.h5ad
│   └── gt/spatial_factors.npy
├── preprocess/       # preprocessed_*.h5ad
├── config/
├── model/
└── output/
```
