# Human Lymph Node

## Data Source

Long, Y., Ang, K.S., Sethi, R. et al. Deciphering spatial domains from spatial multi-omics with SpatialGlue. *Nat Methods* 21, 1658–1667 (2024). https://doi.org/10.1038/s41592-024-02316-4

## Statistics

| Item     | Description |
|----------|-------------|
| Species  | Human       |
| Modality | RNA, ADT    |
| # Cells  | 3,484       |
| RNA (genes) | 17,922  |
| ADT (proteins) | 31   |

## Raw data layout

Use **`data/human_lymph_node/`** as the dataset root. The tutorial [`notebook/human_lymph_node.ipynb`](../../notebook/human_lymph_node.ipynb) reads two aligned AnnData objects from **`raw/`**:

- `raw/rna.h5ad`
- `raw/adt.h5ad`

Each file should share the same cell barcodes (`obs_names`), include `obsm['spatial']`, and RNA should include `obs['cell_type']`. Preprocessing (HVG, PCA/CLR, spatial distance matrix) is done in the notebook on every run; no `preprocess/` cache is used.

Ligand–receptor pairs are passed manually in the notebook configuration (26 pairs; see the `ligand_receptor_map` dict in Step 2).

```
data/human_lymph_node/
├── raw/
│   ├── rna.h5ad
│   └── adt.h5ad
├── model/            # trained CellSTIC checkpoints
├── result/           # cellstic_result.h5ad
└── analysis/         # figures (modality_umap/, cell_type_domain/, tree_level/, …)
```
