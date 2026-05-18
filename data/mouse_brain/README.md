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

Use **`data/mouse_brain/`** in this CellSTIC repository as the dataset root. `utils.loader.load_mouse_brain` expects 10x-style or exported multimodal inputs under **`raw/`**. PeakMatrix CSVs may also live under matching subfolders in `preprocess/` (see `docs/mouse_brain.ipynb`).

```
data/mouse_brain/
├── raw/
│   ├── rna/          # e.g. 10x matrix.mtx.gz (+ features.tsv.gz, barcodes.tsv.gz); or preprocess/rna.h5ad
│   ├── adt/          # e.g. matrix.mtx.gz, features.tsv.gz, barcodes.tsv.gz
│   ├── atac/         # e.g. GSM8494157_5M_20um_ATAC_PeakMatrix.csv (loader reads preprocess/atac/ by default)
│   ├── h3k27ac/      # e.g. GSM8494157_5M_20um_H3K27ac_PeakMatrix.csv
│   ├── h3k27me3/     # e.g. GSM8494157_5M_20um_H3K27me3_PeakMatrix.csv
│   ├── spatial/      # e.g. tissue_positions_list.csv
│   ├── domain/       # optional; auxiliary domain assets
│   ├── l-r/LR.csv    # optional ligand–receptor table
│   └── type/         # optional; e.g. rna_10xWholeMouseBrain(CCN20230722)_CorrelationMapping_UTC_1769068788450.csv
├── preprocess/       # cached h5ad, GeneScore / PeakMatrix CSVs, etc.
├── config/
├── model/
└── output/
```
