# Mouse Embryonic

## Data Source

**Publication.** Chen, A., Liao, S., Cheng, M., Ma, K., Wu, L., Lai, Y., ... & Wang, J. (2022). Spatiotemporal transcriptomic atlas of mouse organogenesis using DNA nanoball-patterned arrays. *Cell*, 185(10), 1777-1792.

**CNGB SciRAID.** Data are from CNGB SciRAID (STDS0000058). Download via:

- E9.5: https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/E9.5_E1S1.MOSTA.h5ad
- E10.5: https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/E10.5_E1S1.MOSTA.h5ad
- E11.5: https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/E11.5_E1S1.MOSTA.h5ad
- E12.5: https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/E12.5_E1S1.MOSTA.h5ad
- E13.5: https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/E13.5_E1S1.MOSTA.h5ad
- E14.5: https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/E14.5_E1S1.MOSTA.h5ad
- E15.5: https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/E15.5_E1S1.MOSTA.h5ad
- E16.5: https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/E16.5_E1S1.MOSTA.h5ad

## Statistics

| Dataset | Platform | Species | # Cells | Features |
|---|---|---|---:|---|
| Mouse embryonic E9.5 | Stereo-seq | Mouse | 5,913 | RNA: 25,568 |
| Mouse embryonic E10.5 | Stereo-seq | Mouse | 18,408 | RNA: 25,201 |
| Mouse embryonic E11.5 | Stereo-seq | Mouse | 30,124 | RNA: 26,854 |
| Mouse embryonic E12.5 | Stereo-seq | Mouse | 51,365 | RNA: 27,810 |
| Mouse embryonic E13.5 | Stereo-seq | Mouse | 77,369 | RNA: 28,408 |
| Mouse embryonic E14.5 | Stereo-seq | Mouse | 102,519 | RNA: 28,463 |
| Mouse embryonic E15.5 | Stereo-seq | Mouse | 113,350 | RNA: 28,798 |
| Mouse embryonic E16.5 | Stereo-seq | Mouse | 121,767 | RNA: 28,204 |

## Raw data layout

Under **`data/mouse_embryo/`** in this CellSTIC repository, use one folder per stage **`data/mouse_embryo/<stage>/`** (e.g. `9.5` … `16.5`), each with `raw/`, `preprocess/` (per organ `Brain` / `Liver`), `config/`, `model/`, `output/`, etc.

- **`raw/`**: stage-level Stereo-seq RNA **`*.h5ad`** (see `utils.loader.load_mouse_embryo`; exclude cache name `preprocessed_RNA_filtered.h5ad`).
- **Ligand–receptor table**: commonly **`data/mouse_embryo/LR.csv`** (matches `lr_path = work_dir.parent / "LR.csv"` in `docs/mouse_embryo.ipynb`), or `raw/l-r/LR.csv` (loader tries both).

```
data/mouse_embryo/
├── LR.csv            # optional; recommended at embryo root
├── 14.5/
│   ├── raw/          # stage raw h5ad
│   ├── preprocess/Brain|Liver/
│   ├── config/
│   ├── model/Brain|Liver/
│   └── output/...
├── 15.5/
│   └── ...
└── ...
```

For cross-stage summaries (e.g. CCI time courses), you can add an optional mirror under **`data/mouse_embryo_cci/`** with `raw/<stage>/Brain|Liver/` (same convention as the main embryo tree).
