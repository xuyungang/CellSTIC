# Axolotl telencephalon

## Data Source

**CNGB SciRAID.** Data are from CNGB SciRAID (STDS0000056). Download via:

- Development: https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000056/stomics/Development.h5ad
- Regeneration: https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000056/stomics/Regeneration.h5ad

## Statistics

| Item | Description |
|---|---|
| Species | Axolotl |
| Modality | RNA (spatial transcriptomics) |
| Settings | Development, Regeneration |
| # Spots / Cells | TBD |
| # Genes | TBD |

## Raw data layout

Under **`data/axolotl_telencephalon/`** in this CellSTIC repository, you may keep the two **combined** `.h5ad` files under `raw/`, and (optionally) split them into per-stage folders (e.g. `develop_44`, `regeneration_10`) for single-sample training/inference.

- **`raw/`**: one or more stage-level RNA `*.h5ad` (see `utils.loader.load_axolotl_telencephalon`; exclude cache name `preprocessed_RNA.h5ad`).
- **Ligand–receptor table**: recommended at **`data/axolotl_telencephalon/LR.csv`** (matches `lr_path = work_dir.parent / "LR.csv"` in the axolotl notebook runner).
- **Spatial coordinates requirement**: the loader expects `adata.obsm["spatial"]`; if missing, it requires `adata.obs[["x","y"]]`.

```
data/axolotl_telencephalon/
├── LR.csv
├── raw/
│   ├── Development.h5ad
│   └── Regeneration.h5ad
├── develop_44/
│   ├── raw/                      # per-stage input h5ad(s)
│   ├── preprocess/               # cache + preprocessing artifacts
│   ├── config/
│   ├── model/
│   └── output/...
├── develop_54/
│   └── ...
├── regeneration_10/
│   └── ...
└── ...
```

If you run the notebook pipeline with a per-stage `work_dir` (e.g. `data/axolotl_telencephalon/develop_44`), it will read:

- raw input from `data/axolotl_telencephalon/develop_44/raw/*.h5ad`
- `LR.csv` from `data/axolotl_telencephalon/LR.csv`
