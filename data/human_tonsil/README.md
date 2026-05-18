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

Use **`data/human_tonsil/`** in this CellSTIC repository as the dataset root. `raw/` must include modality subfolders. `utils.loader.load_human_tonsil` reads TSVs from `raw/rna/` and `raw/adt/` by default (`rna/GSM6578062_humantonsil_RNA.tsv`, `adt/GSM6578071_humantonsil_protein.tsv` in the loader), or you can place pre-built `rna.h5ad` and `adt.h5ad` under `preprocess/` to skip TSV assembly.

```
data/human_tonsil/
├── raw/
│   ├── rna/          # e.g. GSM6578062_humantonsil_RNA.tsv
│   ├── adt/          # e.g. GSM6578071_humantonsil_protein.tsv
│   ├── spatial/      # if required by your workflow
│   └── domain/       # optional; include markers/ etc. as in the reference tree
├── preprocess/
├── config/
├── model/
└── output/
```
