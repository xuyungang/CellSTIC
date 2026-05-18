# Human Skin

## Data Source

Liu, Y., DiStasio, M., Su, G. et al. High-plex protein and whole transcriptome co-mapping at cellular resolution with spatial CITE-seq. *Nat Biotechnol* 41, 1405–1409 (2023). https://doi.org/10.1038/s41587-023-01676-0

## Statistics

| Item     | Description |
|----------|-------------|
| Species  | Human       |
| Modality | RNA, ADT    |
| # Cells  | 1,691       |
| RNA (genes) | 15,486  |
| ADT (proteins) | 283   |

## Raw data layout

Use **`data/human_skin/`** in this CellSTIC repository as the dataset root. `utils.loader.load_human_skin` reads TSVs from `raw/rna/` and `raw/adt/` (`rna/GSM6578065_humanskin_RNA.tsv`, `adt/GSM6578074_humanskin_protein.tsv`), or supply `rna.h5ad` and `adt.h5ad` under `preprocess/`.

```
data/human_skin/
├── raw/
│   ├── rna/          # e.g. GSM6578065_humanskin_RNA.tsv
│   ├── adt/          # e.g. GSM6578074_humanskin_protein.tsv
│   ├── spatial/      # optional; e.g. extra spatial sidecars (coords default from TSV index in loader)
│   └── domain/       # optional; markers/ etc. as in the reference tree
├── preprocess/
├── config/
├── model/
└── output/
```
