# CCC Multi-modal(re1–re8)

## Data Source

Li, H., Zhang, Z., Squires, M. et al. scMultiSim: simulation of single-cell multi-omics and spatial data guided by gene regulatory networks and cell–cell interactions. *Nat Methods* 22, 982–993 (2025). https://doi.org/10.1038/s41592-025-02651-0

Simulated spatial multi-omics data re1–re8.

## Statistics

| Item       | Description |
|------------|-------------|
| Species    | Simulate    |
| Modality   | RNA, ATAC   |
| # Cells    | 400         |
| RNA (genes) | 120        |
| ATAC (peaks) | 360       |

## Raw data layout

Use **`data/scmultisim/`** in this CellSTIC repository: one folder per replicate **`re1` … `re8`**, following the tree below.

`utils.loader.load_scmultisim` **`raw_path`** should be **`data/scmultisim/re<N>/raw/`**, containing:

- `rna/tran_count.csv`
- `atac/atac_count.csv`
- `spatial/coord.csv`
- `l-r/LR.csv`
- `gt/label.h5` (ground truth)
- optional `meta/cell.csv` (cell types, etc.)

```
data/scmultisim/
├── re1/
│   ├── raw/
│   │   ├── rna/          # e.g. tran_count.csv
│   │   ├── atac/         # e.g. atac_count.csv
│   │   ├── spatial/      # e.g. coord.csv
│   │   ├── l-r/          # e.g. LR.csv
│   │   ├── gt/           # e.g. label.h5
│   │   └── meta/         # optional; e.g. cell.csv
│   ├── preprocess/
│   ├── config/
│   ├── model/
│   └── output/
├── re2/
│   └── ...
└── re8/
```
