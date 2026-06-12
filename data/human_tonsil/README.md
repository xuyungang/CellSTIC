# Human Tonsil

## Data Source

Liu, Y., DiStasio, M., Su, G. et al. High-plex protein and whole transcriptome co-mapping at cellular resolution with spatial CITE-seq. *Nat Biotechnol* 41, 1405–1409 (2023). https://doi.org/10.1038/s41587-023-01676-0

## Statistics

| Item     | Description |
|----------|-------------|
| Species  | Human       |
| Modality | RNA, ADT    |
| # Cells  | 2,492       |
| RNA (genes) | 28,417  |
| ADT (proteins) | 283   |

## Raw data layout

Use **`data/human_tonsil/`** as the dataset root. Place two aligned AnnData objects under **`raw/`**:

- `raw/rna.h5ad`
- `raw/adt.h5ad`

Each file should share the same spot barcodes (`obs_names`), include `obsm['spatial']`, and spot coordinates can be parsed from barcodes of the form `{x}x{y}`. This dataset has no cell-type annotation in `obs`. Preprocessing (gene filtering, HVG, PCA/CLR, spatial distance matrix) is done in the analysis notebook on every run; no `preprocess/` cache is required.

Original GEO TSVs (optional, for building the h5ad files): `rna/GSM6578062_humantonsil_RNA.tsv`, `adt/GSM6578071_humantonsil_protein.tsv`.

```
data/human_tonsil/
├── raw/
│   ├── rna.h5ad
│   └── adt.h5ad
├── model/            # trained CellSTIC checkpoints
├── result/           # cellstic_result.h5ad
└── analysis/         # figures
```
