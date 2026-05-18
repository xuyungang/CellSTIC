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

Mirror the directory tree below under **`data/human_lymph_node/`** in this CellSTIC repository (`raw/`, `preprocess/`, `config/`, `model/`, `output/`, etc.). Filenames must match what `utils.loader.load_human_lymph_node` expects (see constants in that module).

```
data/human_lymph_node/
├── raw/
│   ├── rna/          # e.g. GSM8195494_A1LN_matrix.mtx.gz, GSM8195494_A1LN_features.tsv.gz
│   ├── adt/          # e.g. GSM8195498_A1_LN_Protein.h5ad, GSM8195498_A1LN_isotype_normalization_factors.csv.gz
│   ├── spatial/      # e.g. GSM8195494_A1LN_tissue_positions.csv
│   ├── l-r/LR.csv    # ligand–receptor table
│   └── type/         # e.g. cell_type_annotations_mapped.csv
├── preprocess/
├── config/
├── model/
└── output/
```

Set the loader’s `raw_path` to **`data/human_lymph_node/raw`**.
